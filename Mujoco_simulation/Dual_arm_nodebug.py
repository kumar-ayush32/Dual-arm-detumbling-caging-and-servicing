import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
import queue
from dataclasses import dataclass, field
from typing import Optional

XML_PATH = "Custom_arm.xml"
ALL_JOINTS_TMPL     = ["joint_hindarm_{}", "joint_forearm_{}", "joint_hand_{}"]
ACTUATOR_NAMES_TMPL = ["hindarm_ctrl_{}", "forearm_ctrl_{}", "hand_ctrl_{}"]
EE_SITE_NAME_TMPL   = "EE_site_{}"
SHOULDER_JOINT_TMPL = "joint_hindarm_{}"
ARM_IDS = ["a", "b"]

TARGET_A = [-0.15, -0.55]
TARGET_B = [0.15, -0.55]

FREQUENCY       = 3000
IK_QUEUE_SIZE   = 3000000
REPLAN_INTERVAL = 2

V_CRUISE    = 0.15
ALPHA_CURVE = 0.10
DAMP_RADIUS = 0.05
T_MIN       = 0.1
T_MAX       = 4.0

L1 = 0.2875
L2 = 0.2040
L3 = 0.1860
MAX_REACH = L1 + L2 + L3
IK_PREFILL_FRAMES = 150
EE_VEL_ALPHA = 0.6
BLEND_FRAMES    = int(0.05 * FREQUENCY)   # 50 ms worth of blending

# DATA CLASSES
@dataclass
class TrajectoryPlan:
    cartesian_traj   : np.ndarray
    angle_traj       : np.ndarray
    dt               : float
    N                : int
    current_pos      : np.ndarray
    target_pos       : np.ndarray
    target_angle_rad : float
    total_time       : float
    frequency        : int

@dataclass
class ArmState:
    arm_id             : str
    replan_interval    : float
    plan               : Optional[TrajectoryPlan]   = None
    ik_queue           : Optional[queue.Queue]       = None
    ik_status          : dict = field(default_factory=dict)
    worker             : Optional[threading.Thread]  = None
    frame              : int   = 0
    sim_time_at_replan : float = 0.0
    current_q          : Optional[np.ndarray] = None
    last_replan        : float = 0.0

    pending_plan       : Optional[TrajectoryPlan]  = None
    pending_iq         : Optional[queue.Queue]     = None
    pending_ist        : Optional[dict]            = None
    pending_worker     : Optional[threading.Thread] = None
    pending_st         : float = 0.0

    ee_vel_filtered    : Optional[np.ndarray] = None
    fk_origin          : Optional[np.ndarray] = None

# MOVING TARGET THREAD
class MovingTargetThread(threading.Thread):
    _CORNER_OFFSET = {
        0: (-1, -1, -1),
        1: (+1, +1, +1),
    }

    def __init__(self, model, data, data_lock, arm_index,
                 update_interval=0.05):
        super().__init__(daemon=True)
        self._model           = model
        self._data            = data
        self._data_lock       = data_lock
        self._update_interval = update_interval
        self._stop_event      = threading.Event()
        self._pos_lock        = threading.Lock()

        self._bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Target")
        self._gid = next(i for i in range(model.ngeom)
                         if model.geom_bodyid[i] == self._bid)

        sx, sy, sz = model.geom_size[self._gid][:3]
        dx, dy, dz = self._CORNER_OFFSET[arm_index]
        self._local_off = np.array([dx * sx, dy * sy, dz * sz])

        with self._data_lock:
            corner, phi = self._compute()
        self._current_corner = corner
        self._current_phi    = phi

    def _compute(self):
        geom_pos = self._data.geom_xpos[self._gid].copy()
        geom_rot = self._data.geom_xmat[self._gid].reshape(3, 3)
        world_corner = geom_pos + geom_rot @ self._local_off
        corner_xy    = world_corner[:2].copy()
        centre_xy = geom_pos[:2].copy()
        direction = corner_xy - centre_xy
        phi = np.arctan2(direction[0], direction[1])
        return corner_xy, phi

    def run(self):
        while not self._stop_event.is_set():
            with self._data_lock:
                corner, phi = self._compute()
            with self._pos_lock:
                self._current_corner = corner
                self._current_phi    = phi
            time.sleep(self._update_interval)

    def get_target(self):
        """Returns (corner_xy, phi_rad) — both computed from the same snapshot."""
        with self._pos_lock:
            return self._current_corner.copy(), float(self._current_phi)

    def stop(self):
        self._stop_event.set()

# MAP BUILDERS
def build_maps(model, arm_id):
    all_joints     = [j.format(arm_id) for j in ALL_JOINTS_TMPL]
    actuator_names = [a.format(arm_id) for a in ACTUATOR_NAMES_TMPL]

    all_joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                       for i in range(model.njnt)]
    all_act_names   = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                       for i in range(model.nu)]

    qpos_adr = {}
    for j in all_joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if jid < 0:
            raise RuntimeError(f"Joint '{j}' not found.\nAvailable: {all_joint_names}")
        qpos_adr[j] = model.jnt_qposadr[jid]

    act_id = {}
    for a in actuator_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        if aid < 0:
            raise RuntimeError(f"Actuator '{a}' not found.\nAvailable: {all_act_names}")
        act_id[a] = aid

    return qpos_adr, act_id

def find_ee_site(model, arm_id):
    name = EE_SITE_NAME_TMPL.format(arm_id)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid >= 0:
        print(f"[EE site] Arm-{arm_id}: '{name}' (id={sid})")
        return sid
    all_sites = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or ""
                 for i in range(model.nsite)]
    raise RuntimeError(f"Could not find EE site '{name}' for arm '{arm_id}'.\n"
                       f"Sites available: {all_sites}")

def find_shoulder_joint(model, arm_id):
    name = SHOULDER_JOINT_TMPL.format(arm_id)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise RuntimeError(f"Could not find shoulder joint '{name}' for arm '{arm_id}'.")
    print(f"[Shoulder joint] Arm-{arm_id}: '{name}' (id={jid})")
    return jid

def read_actual_q(data, qpos_adr, arm_id):
    return np.array([
        data.qpos[qpos_adr[f"joint_hindarm_{arm_id}"]],
        data.qpos[qpos_adr[f"joint_forearm_{arm_id}"]],
        data.qpos[qpos_adr[f"joint_hand_{arm_id}"]],
    ])

def ee_world_velocity(model, data, ee_site_id, qpos_adr, arm_id):
    nv   = model.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
    vel = np.zeros(nv)
    for jname in [f"joint_hindarm_{arm_id}",
                  f"joint_forearm_{arm_id}",
                  f"joint_hand_{arm_id}"]:
        jid    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        dofadr = model.jnt_dofadr[jid]
        vel[dofadr] = data.qvel[dofadr]
    return (jacp @ vel)[[0, 1]]   # world-frame XY velocity

def solve_ik_trig(target_xz, phi_world, is_arm_b, elbow_up=True):
    target_x, target_z = target_xz

    x_wrist = target_x + L3 * np.sin(phi_world)
    z_wrist = target_z + L3 * np.cos(phi_world)

    cos_q2 = (x_wrist**2 + z_wrist**2 - L1**2 - L2**2) / (2.0 * L1 * L2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)

    if elbow_up and is_arm_b:
        q2 = -np.arccos(cos_q2)
    else:
        q2 =  np.arccos(cos_q2)

    alpha = np.arctan2(-x_wrist, -z_wrist)
    beta  = np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    q1 = alpha - beta
    q3 = phi_world - q1 - q2
    return np.array([q1, q2, q3])

def phi_world_from_q(q):
    return q[0] + q[1] + q[2]

def fk_local_from_q(q):
    q1, q2, q3 = q
    x = -L1 * np.sin(q1) - L2 * np.sin(q1 + q2) - L3 * np.sin(q1 + q2 + q3)
    y = -L1 * np.cos(q1) - L2 * np.cos(q1 + q2) - L3 * np.cos(q1 + q2 + q3)
    return np.array([x, y])

def _quintic_coeffs(p0, pf, v0, T):
    """Zero-final-velocity quintic coefficients for a scalar or 1-D array."""
    T3 = T**3; T4 = T**4; T5 = T**5
    dp = pf - p0
    a0 = p0
    a1 = v0
    a2 = np.zeros_like(dp) if hasattr(dp, '__len__') else 0.0
    a3 = (10*dp - T*(6*v0)) / T3
    a4 = (-15*dp + T*(8*v0)) / T4
    a5 = (6*dp  - T*(3*v0)) / T5
    return a0, a1, a2, a3, a4, a5

def _quintic_eval(coeffs, t):
    a0, a1, a2, a3, a4, a5 = coeffs
    return a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5

def _quintic_blend(start_pos, target_pos, start_phi, target_phi,
                   start_vel, start_phi_vel, total_time, frequency):
    if start_vel is None:
        start_vel = np.zeros(2)
    start_vel = np.asarray(start_vel, dtype=float)
    T  = float(total_time)
    ts = np.linspace(0, T, int(T * frequency))

    pos_coeffs = _quintic_coeffs(start_pos, target_pos, start_vel, T)
    traj = np.array([_quintic_eval(pos_coeffs, ti) for ti in ts])

    phi_coeffs = _quintic_coeffs(start_phi, target_phi, start_phi_vel, T)
    angle_traj = np.array([_quintic_eval(phi_coeffs, ti) for ti in ts])

    return traj, angle_traj

def compute_total_time(start, target, start_vel):
    dist = float(np.linalg.norm(np.asarray(target) - np.asarray(start)))
    if dist < 1e-4:
        return T_MIN
    T_cruise  = dist / V_CRUISE
    v_mag     = float(np.linalg.norm(start_vel))
    T_vel_cap = (ALPHA_CURVE * dist / (0.135 * v_mag)) if v_mag > 1e-4 else T_MAX
    return float(np.clip(min(T_cruise, T_vel_cap), T_MIN, T_MAX))

def damp_start_velocity(start, target, start_vel):
    dist = float(np.linalg.norm(np.asarray(target) - np.asarray(start)))
    if dist >= DAMP_RADIUS:
        return start_vel
    return start_vel * (dist / DAMP_RADIUS) ** 2

def plan_trajectory(current_pos, target_pos, start_phi_rad, target_phi_rad,
                    start_vel=None, start_phi_vel=0.0,
                    frequency=FREQUENCY) -> TrajectoryPlan:
    """All positions in SHOULDER-LOCAL X-Y space. Caller converts world→local first."""
    current_pos = np.asarray(current_pos, dtype=float)
    target_pos  = np.asarray(target_pos,  dtype=float)
    if start_vel is None:
        start_vel = np.zeros(2)
    start_vel = np.asarray(start_vel, dtype=float)
    start_vel  = damp_start_velocity(current_pos, target_pos, start_vel)
    total_time = compute_total_time(current_pos, target_pos, start_vel)

    cartesian_traj, angle_traj = _quintic_blend(
        current_pos, target_pos, start_phi_rad, target_phi_rad,
        start_vel, start_phi_vel, total_time, frequency,
    )
    dt   = 1.0 / frequency
    N    = len(cartesian_traj)
    dist = float(np.linalg.norm(target_pos - current_pos))
    print(f"[Plan] {N} pts | dist={dist:.3f}m | T={total_time:.2f}s | "
          f"local {np.round(current_pos,4).tolist()} → {np.round(target_pos,4).tolist()} | "
          f"phi {np.degrees(start_phi_rad):.1f}° → {np.degrees(target_phi_rad):.1f}°")
    return TrajectoryPlan(
        cartesian_traj=cartesian_traj, angle_traj=angle_traj, dt=dt, N=N,
        current_pos=current_pos, target_pos=target_pos,
        target_angle_rad=target_phi_rad, total_time=total_time, frequency=frequency,
    )

# IK WORKER THREAD
class IKWorker(threading.Thread):
    def __init__(self, cartesian_traj_local, angle_traj, ik_queue, status, arm_id,
                 elbow_up=True):
        super().__init__(daemon=True)
        self.cartesian_traj_local = cartesian_traj_local.copy()
        self.angle_traj           = angle_traj.copy()
        self.ik_queue             = ik_queue
        self.status               = status
        self.is_arm_b             = (arm_id == ARM_IDS[1])
        self.elbow_up             = elbow_up
        self._solver = solve_ik_trig

    def run(self):
        N = len(self.cartesian_traj_local)
        try:
            for i in range(N):
                if self.status.get("cancel", False):
                    return
                q = self._solver(self.cartesian_traj_local[i], self.angle_traj[i],
                                 self.is_arm_b, elbow_up=self.elbow_up)
                try:
                    self.ik_queue.put(q, timeout=1.0)
                except queue.Full:
                    if self.status.get("cancel", False):
                        return
        except Exception as exc:
            print(f"[IKWorker] EXCEPTION in IK solve: {exc}")
            self.status["error"] = str(exc)
        finally:
            self.status["done"] = True

# REPLAN HELPER
def _start_new_plan(model, data, ee_site_id, shoulder_joint_id, qpos_adr, arm_id,
                    target_pos, target_phi_rad,
                    ee_vel_filtered=None):
    actual_q = read_actual_q(data, qpos_adr, arm_id)
    ee_world       = data.site_xpos[ee_site_id][[0, 1]].copy()
    ee_local_start = fk_local_from_q(actual_q)
    fk_origin      = ee_world - ee_local_start
    target_local   = np.asarray(target_pos, dtype=float) - fk_origin

    reach_dist = float(np.linalg.norm(target_local))
    if reach_dist > MAX_REACH:
        print(f"[WARNING][Replan-{arm_id}] target {reach_dist:.3f} m from FK origin "
              f"(max {MAX_REACH:.3f} m) → UNREACHABLE")

    start_phi_rad = phi_world_from_q(actual_q)
    start_phi_vel = 0.0
    elbow_up      = actual_q[1] >= 0
    start_vel_local = ee_vel_filtered if ee_vel_filtered is not None else np.zeros(2)
    v_cap = 2.0 * V_CRUISE
    v_mag = float(np.linalg.norm(start_vel_local))
    if v_mag > v_cap:
        start_vel_local = start_vel_local * (v_cap / v_mag)

    world_dist = float(np.linalg.norm(np.asarray(target_pos, dtype=float) - ee_world))
    print(f"[Replan-{arm_id}] actual_q={np.round(np.degrees(actual_q), 2).tolist()}°")
    print(f"[Replan-{arm_id}] fk_origin={np.round(fk_origin,4).tolist()} "
          f"ee_local(FK)={np.round(ee_local_start,4).tolist()}")
    print(f"[Replan-{arm_id}] target world={np.asarray(target_pos).tolist()} "
          f"local(IK frame)={np.round(target_local,4).tolist()} world_dist={world_dist:.4f}m")
    print(f"[Replan-{arm_id}] phi {np.degrees(start_phi_rad):.1f}° → "
          f"{np.degrees(target_phi_rad):.1f}°  elbow_up={elbow_up}")
    print(f"[Replan-{arm_id}] start_vel_local={np.round(start_vel_local,4).tolist()}")
    plan = plan_trajectory(
        current_pos=ee_local_start,
        target_pos=target_local,
        start_phi_rad=start_phi_rad,
        target_phi_rad=target_phi_rad,
        start_vel=start_vel_local,
        start_phi_vel=start_phi_vel,
        frequency=FREQUENCY,
    )

    ik_queue  = queue.Queue(maxsize=IK_QUEUE_SIZE)
    ik_status = {"done": False, "cancel": False}
    worker    = IKWorker(plan.cartesian_traj, plan.angle_traj, ik_queue, ik_status,
                         arm_id, elbow_up=elbow_up)
    worker.start()
    return plan, ik_queue, ik_status, worker, 0, data.time, fk_origin

def _cancel_worker(state: ArmState):
    """Cancel the running IK worker and drain its queue (if any)."""
    if state.ik_status:
        state.ik_status["cancel"] = True
    if state.worker is not None and state.worker.is_alive():
        state.worker.join(timeout=0.5)
    if state.ik_queue is not None:
        while not state.ik_queue.empty():
            try:
                state.ik_queue.get_nowait()
            except queue.Empty:
                break

def _cancel_pending(state: ArmState):
    """Cancel a pending (pre-filling) IK worker if one exists."""
    if state.pending_ist:
        state.pending_ist["cancel"] = True
    if state.pending_worker is not None and state.pending_worker.is_alive():
        state.pending_worker.join(timeout=0.5)
    if state.pending_iq is not None:
        while not state.pending_iq.empty():
            try:
                state.pending_iq.get_nowait()
            except queue.Empty:
                break
    state.pending_plan   = None
    state.pending_iq     = None
    state.pending_ist    = None
    state.pending_worker = None
    state.pending_st     = 0.0

# TRAJECTORY VISUALISER
_VIS_MAX_SEGS = 200
_COLOURS = {
    ARM_IDS[0]: {
        "done"   : np.array([0.55, 0.55, 0.55, 0.50], dtype=np.float32),
        "pending": np.array([0.20, 0.90, 0.30, 0.90], dtype=np.float32),
        "current": np.array([1.00, 1.00, 1.00, 1.00], dtype=np.float32),
        "target" : np.array([0.95, 0.20, 0.20, 1.00], dtype=np.float32),
    },
    ARM_IDS[1]: {
        "done"   : np.array([0.40, 0.40, 0.60, 0.50], dtype=np.float32),
        "pending": np.array([0.20, 0.85, 0.95, 0.90], dtype=np.float32),
        "current": np.array([1.00, 1.00, 0.00, 1.00], dtype=np.float32),
        "target" : np.array([1.00, 0.50, 0.00, 1.00], dtype=np.float32),
    },
}
_EYE = np.eye(3, dtype=np.float64).flatten()

def _add_segment(scn, p1_xy, p2_xy, rgba, z_height=0.0, width_px=2.0):
    if scn.ngeom >= scn.maxgeom - 4:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE,
                        np.zeros(3, np.float64), np.zeros(3, np.float64), _EYE, rgba)
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width_px,
                         np.array([p1_xy[0], p1_xy[1], z_height], np.float64),
                         np.array([p2_xy[0], p2_xy[1], z_height], np.float64))
    scn.ngeom += 1

def _add_sphere(scn, pos_xy, rgba, z_height=0.0, size=0.008):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([size]*3, np.float64),
                        np.array([pos_xy[0], pos_xy[1], z_height], np.float64),
                        _EYE, rgba)
    scn.ngeom += 1

def draw_trajectories(viewer, plan_a, frame_a, plan_b, frame_b,
                      fk_origin_a, fk_origin_b,
                      shoulder_joint_a_z, shoulder_joint_b_z,
                      pending_plan_a=None, pending_plan_b=None):
    if not hasattr(viewer, "user_scn"):
        return
    scn = viewer.user_scn
    scn.ngeom = 0

    _PENDING_PREVIEW = {
        ARM_IDS[0]: np.array([0.20, 0.90, 0.30, 0.35], dtype=np.float32),
        ARM_IDS[1]: np.array([0.20, 0.85, 0.95, 0.35], dtype=np.float32),
    }

    for plan, frame, arm_id, fk_orig, z_h, pp in (
        (plan_a, frame_a, ARM_IDS[0], fk_origin_a, shoulder_joint_a_z, pending_plan_a),
        (plan_b, frame_b, ARM_IDS[1], fk_origin_b, shoulder_joint_b_z, pending_plan_b),
    ):
        col = _COLOURS[arm_id]

        if pp is not None:
            pp_world = pp.cartesian_traj + fk_orig
            step = max(1, len(pp_world) // _VIS_MAX_SEGS)
            for i in range(0, len(pp_world) - 1, step):
                _add_segment(scn, pp_world[i], pp_world[i+1],
                             _PENDING_PREVIEW[arm_id], z_height=z_h, width_px=1.5)
            _add_sphere(scn, pp_world[-1], col["target"], z_height=z_h, size=0.012)

        if plan is None:
            continue
        traj_local = plan.cartesian_traj
        traj_world = traj_local + fk_orig

        N          = plan.N
        n_exec     = max(frame - 1, 0)
        n_rem      = max(N - 1 - frame, 0)
        total_segs = n_exec + n_rem
        if total_segs == 0:
            _add_sphere(scn, traj_world[0], col["current"], z_height=z_h, size=0.010)
            continue
        ratio   = min(1.0, _VIS_MAX_SEGS / total_segs)
        step_ex = max(1, int(1.0 / ratio))
        step_rm = max(1, int(1.0 / ratio))
        for i in range(0, min(frame, N-1), step_ex):
            _add_segment(scn, traj_world[i], traj_world[i+1], col["done"],
                         z_height=z_h, width_px=1.5)
        for i in range(max(frame, 0), N-1, step_rm):
            _add_segment(scn, traj_world[i], traj_world[i+1], col["pending"],
                         z_height=z_h, width_px=3.0)

        _add_sphere(scn, traj_world[min(frame, N-1)], col["current"],
                    z_height=z_h, size=0.010)
        _add_sphere(scn, traj_world[-1], col["target"],
                    z_height=z_h, size=0.012)

def _tick_arm(state: ArmState, model, data, qpos_adr, act_id, ee_site_id, shoulder_joint_id,
              target_thread: MovingTargetThread):
    arm_id  = state.arm_id
    sim_now = data.time

    raw_vel = ee_world_velocity(model, data, ee_site_id, qpos_adr, arm_id)
    if state.ee_vel_filtered is None:
        state.ee_vel_filtered = raw_vel.copy()
    else:
        state.ee_vel_filtered = (EE_VEL_ALPHA * raw_vel
                                 + (1.0 - EE_VEL_ALPHA) * state.ee_vel_filtered)

    if sim_now - state.last_replan >= state.replan_interval:
        state.last_replan = sim_now
        new_target, target_phi_rad = target_thread.get_target()
        print(f"\n[_tick_arm] Arm-{arm_id} replan → {new_target.tolist()} "
              f"phi={np.degrees(target_phi_rad):.1f}°")

        _cancel_pending(state)

        plan, iq, ist, wk, fr, st, new_fk_origin = _start_new_plan(
            model, data, ee_site_id, shoulder_joint_id, qpos_adr, arm_id,
            target_pos=new_target, target_phi_rad=target_phi_rad,
            ee_vel_filtered=state.ee_vel_filtered,
        )
        state.fk_origin      = new_fk_origin
        state.pending_plan   = plan
        state.pending_iq     = iq
        state.pending_ist    = ist
        state.pending_worker = wk
        state.pending_st     = st

    if state.pending_plan is not None:
        p_done      = state.pending_ist.get("done", False)
        p_cancelled = state.pending_ist.get("cancel", False)
        p_error     = "error" in state.pending_ist

        if p_cancelled or p_error:
            if p_error:
                print(f"[_tick_arm] Arm-{arm_id} pending IK worker failed: "
                      f"{state.pending_ist.get('error')} — discarding plan")
            _cancel_pending(state)
        else:
            ready = state.pending_iq.qsize() >= IK_PREFILL_FRAMES
            if not ready and p_done:
                ready = not state.pending_iq.empty()
            if not ready and state.pending_plan is not None:
                age = data.time - state.pending_st
                if age > state.pending_plan.total_time * 2.0 + 0.5:
                    print(f"[_tick_arm] Arm-{arm_id} pending plan TIMEOUT "
                          f"(age={age:.2f}s) — force-promoting")
                    ready = not state.pending_iq.empty()

            if ready:
                _cancel_worker(state)
                state.plan               = state.pending_plan
                state.ik_queue           = state.pending_iq
                state.ik_status          = state.pending_ist
                state.worker             = state.pending_worker
                state.frame              = 0
                state.sim_time_at_replan = data.time

                state.pending_plan   = None
                state.pending_iq     = None
                state.pending_ist    = None
                state.pending_worker = None
                print(f"[_tick_arm] Arm-{arm_id} new plan activated (prefill satisfied)")

    if state.plan is not None:
        elapsed_sim  = data.time - state.sim_time_at_replan
        target_frame = min(int(elapsed_sim / state.plan.dt), state.plan.N - 1)
        step_limit = max(1, int(model.opt.timestep / state.plan.dt) + 2)
        consumed = 0
        while state.frame <= target_frame and consumed < step_limit:
            try:
                q_new = state.ik_queue.get_nowait()
                if state.current_q is not None and state.frame < BLEND_FRAMES:
                    alpha = state.frame / max(BLEND_FRAMES - 1, 1)
                    q_new = (1.0 - alpha) * state.current_q + alpha * q_new

                state.current_q = q_new
                state.frame    += 1
                consumed       += 1
            except queue.Empty:
                break
        if state.frame >= state.plan.N:
            state.frame = state.plan.N - 1
        plan_done = (state.frame >= state.plan.N - 1
                     and state.ik_queue.empty()
                     and (state.ik_status.get("done", False)
                          or not (state.worker and state.worker.is_alive())))
        if plan_done and state.pending_plan is None:
            state.last_replan = data.time

    if state.current_q is not None:
        act_names = [a.format(arm_id) for a in ACTUATOR_NAMES_TMPL]
        for name, val in zip(act_names, state.current_q):
            data.ctrl[act_id[name]] = val
    return state

# EXECUTE MOTION
def execute_motion(model, data, qpos_adr_a, act_id_a, ee_site_a, shoulder_joint_a,
                   qpos_adr_b, act_id_b, ee_site_b, shoulder_joint_b,
                   target_thread_a: MovingTargetThread,
                   target_thread_b: MovingTargetThread,
                   replan_interval: float,
                   data_lock: threading.Lock):
    mujoco.mj_forward(model, data)
    counter = 0

    def _init_arm(arm_id, ee_site_id, shoulder_joint_id, qpos_adr, act_id, t_thread):
        tpos, phi = t_thread.get_target()
        print(f"[Init-{arm_id}] corner={np.round(tpos,4).tolist()} phi={np.degrees(phi):.1f}°")

        plan, iq, ist, wk, fr, st, init_fk_origin = _start_new_plan(
            model, data, ee_site_id, shoulder_joint_id, qpos_adr, arm_id,
            target_pos=tpos, target_phi_rad=phi,
            ee_vel_filtered=None,
        )
        actual_q = read_actual_q(data, qpos_adr, arm_id)
        for name, val in zip([a.format(arm_id) for a in ACTUATOR_NAMES_TMPL], actual_q):
            data.ctrl[act_id[name]] = val

        return ArmState(
            arm_id=arm_id, replan_interval=replan_interval,
            plan=plan, ik_queue=iq, ik_status=ist, worker=wk,
            frame=fr, sim_time_at_replan=st,
            current_q=actual_q,
            last_replan=data.time,
            ee_vel_filtered=np.zeros(2),
            fk_origin=init_fk_origin,
        )

    state_a = _init_arm(ARM_IDS[0], ee_site_a, shoulder_joint_a, qpos_adr_a, act_id_a,
                         target_thread_a)
    state_b = _init_arm(ARM_IDS[1], ee_site_b, shoulder_joint_b, qpos_adr_b, act_id_b,
                         target_thread_b)

    t0_wall = time.perf_counter()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            with data_lock:
                state_a = _tick_arm(state_a, model, data,
                                     qpos_adr_a, act_id_a, ee_site_a, shoulder_joint_a,
                                     target_thread_a)
                state_b = _tick_arm(state_b, model, data,
                                     qpos_adr_b, act_id_b, ee_site_b, shoulder_joint_b,
                                     target_thread_b)

                mujoco.mj_step(model, data)

                shoulder_a = data.xanchor[shoulder_joint_a][[0, 1]].copy()
                shoulder_b = data.xanchor[shoulder_joint_b][[0, 1]].copy()
                shoulder_a_z = float(data.xanchor[shoulder_joint_a][2])
                shoulder_b_z = float(data.xanchor[shoulder_joint_b][2])

                ee_a = data.site_xpos[ee_site_a].copy()
                ee_b = data.site_xpos[ee_site_b].copy()
                sim_time = data.time

            fk_orig_a = state_a.fk_origin if state_a.fk_origin is not None else shoulder_a
            fk_orig_b = state_b.fk_origin if state_b.fk_origin is not None else shoulder_b
            draw_trajectories(viewer,
                              state_a.plan, state_a.frame,
                              state_b.plan, state_b.frame,
                              fk_orig_a, fk_orig_b,
                              shoulder_a_z, shoulder_b_z,
                              pending_plan_a=state_a.pending_plan,
                              pending_plan_b=state_b.pending_plan)
            viewer.sync()

            sleep_t = sim_time - (time.perf_counter() - t0_wall)
            if sleep_t > 0:
                time.sleep(sleep_t)

            q_a  = np.degrees(state_a.current_q) if state_a.current_q is not None else np.zeros(3)
            q_b  = np.degrees(state_b.current_q) if state_b.current_q is not None else np.zeros(3)
            phi_a_now = (state_a.plan.angle_traj[min(state_a.frame, state_a.plan.N-1)]
                         if state_a.plan else 0.0)
            phi_b_now = (state_b.plan.angle_traj[min(state_b.frame, state_b.plan.N-1)]
                         if state_b.plan else 0.0)
            counter += 1
            if counter > 100:
                print(f"[Sim] t={sim_time:.3f}s  "
                      f"A: EE=({ee_a[0]:.3f},{ee_a[1]:.3f}) "
                      f"phi={np.degrees(phi_a_now):.1f}deg "
                      f"q=[{q_a[0]:.1f} {q_a[1]:.1f} {q_a[2]:.1f}]°")
                print(f"[Sim] t={sim_time:.3f}s  "
                      f"B: EE=({ee_b[0]:.3f},{ee_b[1]:.3f}) "
                      f"phi={np.degrees(phi_b_now):.1f}deg "
                      f"q=[{q_b[0]:.1f} {q_b[1]:.1f} {q_b[2]:.1f}]°")
                counter = 0
                for i in range(model.njnt):
                    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    qpos_addr  = model.jnt_qposadr[i]
                    print(f"{joint_name:20s} : "
                          f"{data.qpos[qpos_addr]: .6f} rad "
                          f"({np.degrees(data.qpos[qpos_addr]): .2f} deg)")

    print("\n[Execute motion] Viewer closed.")

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    qpos_adr_a, act_id_a = build_maps(model, ARM_IDS[0])
    qpos_adr_b, act_id_b = build_maps(model, ARM_IDS[1])

    ee_site_a        = find_ee_site(model, ARM_IDS[0])
    ee_site_b        = find_ee_site(model, ARM_IDS[1])
    shoulder_joint_a = find_shoulder_joint(model, ARM_IDS[0])
    shoulder_joint_b = find_shoulder_joint(model, ARM_IDS[1])

    mujoco.mj_forward(model, data)

    ee_a_world = data.site_xpos[ee_site_a][[0, 1]].copy()
    ee_b_world = data.site_xpos[ee_site_b][[0, 1]].copy()
    print(f"[Main] EE-A: {ee_a_world}  → target_A = {TARGET_A}")
    print(f"[Main] EE-B: {ee_b_world}  → target_B = {TARGET_B}")

    shoulder_a_world = data.xanchor[shoulder_joint_a][[0, 1]].copy()
    shoulder_b_world = data.xanchor[shoulder_joint_b][[0, 1]].copy()
    dist_a = float(np.linalg.norm(np.asarray(TARGET_A) - shoulder_a_world))
    dist_b = float(np.linalg.norm(np.asarray(TARGET_B) - shoulder_b_world))
    print(f"[Main] Shoulder-A: {shoulder_a_world.tolist()}  "
          f"dist={dist_a:.3f}m (max {MAX_REACH:.3f}m) "
          f"{'→ UNREACHABLE!' if dist_a > MAX_REACH else '→ OK'}")
    print(f"[Main] Shoulder-B: {shoulder_b_world.tolist()}  "
          f"dist={dist_b:.3f}m (max {MAX_REACH:.3f}m) "
          f"{'→ UNREACHABLE!' if dist_b > MAX_REACH else '→ OK'}")

    data_lock  = threading.Lock()
    target_thread_a = MovingTargetThread(model, data, data_lock, arm_index=1)
    target_thread_b = MovingTargetThread(model, data, data_lock, arm_index=0)
    target_thread_a.start()
    target_thread_b.start()

    try:
        execute_motion(
            model=model, data=data,
            qpos_adr_a=qpos_adr_a, act_id_a=act_id_a,
            ee_site_a=ee_site_a, shoulder_joint_a=shoulder_joint_a,
            qpos_adr_b=qpos_adr_b, act_id_b=act_id_b,
            ee_site_b=ee_site_b, shoulder_joint_b=shoulder_joint_b,
            target_thread_a=target_thread_a,
            target_thread_b=target_thread_b,
            replan_interval=REPLAN_INTERVAL,
            data_lock=data_lock,
        )
    finally:
        target_thread_a.stop()
        target_thread_b.stop()
        print("[Main] Target threads stopped.")

if __name__ == "__main__":
    main()