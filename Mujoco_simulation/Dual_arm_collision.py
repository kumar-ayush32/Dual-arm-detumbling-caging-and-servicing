import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
import queue
from dataclasses import dataclass, field
from typing import Optional

XML_PATH = "Custom_arm.xml"
DOF3_JOINTS_TMPL    = ["joint_hindarm_{}", "joint_forearm_{}", "joint_hand_{}"]
ALL_JOINTS_TMPL     = ["joint_hindarm_{}", "joint_forearm_{}", "joint_hand_{}"]
ACTUATOR_NAMES_TMPL = ["hindarm_ctrl_{}", "forearm_ctrl_{}", "hand_ctrl_{}"]
EE_SITE_NAME_TMPL   = "EE_site_{}"
SHOULDER_JOINT_TMPL = "joint_hindarm_{}"
ARM_IDS = ["a", "b"]

TARGET_A = [-0.15, -0.55]
TARGET_B = [0.15, -0.55]
TARGET_ANGLE_DEG_A = 90.0
TARGET_ANGLE_DEG_B = -90.0

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

# MOVING TARGET THREAD
class MovingTargetThread(threading.Thread):
    _CORNER_OFFSET = {
        0: (-1, -1, -1),   # arm 0 → bottom-left-back corner
        1: (+1, +1, +1),   # arm 1 → top-right-front corner
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
            self._current_corner = self._compute_corner()

    def _compute_corner(self):
        geom_pos = self._data.geom_xpos[self._gid].copy()
        geom_rot = self._data.geom_xmat[self._gid].reshape(3, 3)
        world_corner = geom_pos + geom_rot @ self._local_off
        return world_corner[:2].copy()

    def run(self):
        while not self._stop_event.is_set():
            with self._data_lock:
                corner_xy = self._compute_corner()
            with self._pos_lock:
                self._current_corner = corner_xy
            time.sleep(self._update_interval)

    def get_target(self):
        with self._pos_lock:
            return self._current_corner.copy()

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

def _quintic_angle_vel_at(plan: TrajectoryPlan, frame: int) -> float:
    n = plan.N
    if n < 2:
        return 0.0
    f = min(max(frame, 0), n - 1)
    if f == 0:
        return (plan.angle_traj[1] - plan.angle_traj[0]) / plan.dt
    if f == n - 1:
        return (plan.angle_traj[-1] - plan.angle_traj[-2]) / plan.dt
    return (plan.angle_traj[f+1] - plan.angle_traj[f-1]) / (2.0 * plan.dt)

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
        self.elbow_up             = elbow_up   # Bug 5 fix
        self._solver = solve_ik_trig

    def run(self):
        N = len(self.cartesian_traj_local)
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
        self.status["done"] = True

# REPLAN HELPER
def _start_new_plan(model, data, ee_site_id, shoulder_joint_id, qpos_adr, arm_id,
                    target_pos, target_phi_rad,
                    current_q, current_frame, old_plan):
    shoulder_world = data.xanchor[shoulder_joint_id][[0, 1]].copy()
    ee_world       = data.site_xpos[ee_site_id][[0, 1]].copy()
    ee_local       = ee_world - shoulder_world
    target_local   = np.asarray(target_pos, dtype=float) - shoulder_world

    reach_dist = float(np.linalg.norm(target_local))
    if reach_dist > MAX_REACH:
        print(f"[WARNING][Replan-{arm_id}] target {reach_dist:.3f} m from shoulder "
              f"(max {MAX_REACH:.3f} m) → UNREACHABLE")

    start_vel_world = ee_world_velocity(model, data, ee_site_id, qpos_adr, arm_id)

    if old_plan is not None and current_frame < old_plan.N:
        start_phi_rad = old_plan.angle_traj[current_frame]
        start_phi_vel = _quintic_angle_vel_at(old_plan, current_frame)
    elif current_q is not None:
        start_phi_rad = phi_world_from_q(current_q)
        start_phi_vel = 0.0
    else:
        actual_q      = read_actual_q(data, qpos_adr, arm_id)
        start_phi_rad = phi_world_from_q(actual_q)
        start_phi_vel = 0.0

    if current_q is not None:
        elbow_up = current_q[1] >= 0
    else:
        elbow_up = True

    print(f"[Replan-{arm_id}] EE world={ee_world.tolist()} local={ee_local.tolist()}")
    print(f"[Replan-{arm_id}] target world={np.asarray(target_pos).tolist()} "
          f"local={target_local.tolist()}")
    print(f"[Replan-{arm_id}] shoulder={shoulder_world.tolist()}")
    print(f"[Replan-{arm_id}] phi {np.degrees(start_phi_rad):.1f}° → "
          f"{np.degrees(target_phi_rad):.1f}°  elbow_up={elbow_up}")

    plan = plan_trajectory(
        current_pos=ee_local,
        target_pos=target_local,
        start_phi_rad=start_phi_rad,
        target_phi_rad=target_phi_rad,
        start_vel=start_vel_world,
        start_phi_vel=start_phi_vel,
        frequency=FREQUENCY,
    )

    ik_queue  = queue.Queue(maxsize=IK_QUEUE_SIZE)
    ik_status = {"done": False, "cancel": False}
    worker    = IKWorker(plan.cartesian_traj, plan.angle_traj, ik_queue, ik_status,
                         arm_id, elbow_up=elbow_up)
    worker.start()
    return plan, ik_queue, ik_status, worker, 0, data.time

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
                      shoulder_a, shoulder_b,
                      shoulder_joint_a_z, shoulder_joint_b_z):

    if not hasattr(viewer, "user_scn"):
        return
    scn = viewer.user_scn
    scn.ngeom = 0

    for plan, frame, arm_id, shoulder, z_h in (
        (plan_a, frame_a, ARM_IDS[0], shoulder_a, shoulder_joint_a_z),
        (plan_b, frame_b, ARM_IDS[1], shoulder_b, shoulder_joint_b_z),
    ):
        if plan is None:
            continue
        traj_local = plan.cartesian_traj
        traj_world = traj_local + shoulder

        N          = plan.N
        col        = _COLOURS[arm_id]
        n_exec     = max(frame - 1, 0)
        n_rem      = max(N - 1 - frame, 0)
        total_segs = n_exec + n_rem
        if total_segs == 0:
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

# PER-ARM TICK
def _tick_arm(state: ArmState, model, data, qpos_adr, act_id, ee_site_id, shoulder_joint_id,
              target_thread: MovingTargetThread, target_phi_rad: float):
    arm_id = state.arm_id
    now    = time.perf_counter()

    if now - state.last_replan >= state.replan_interval:
        state.last_replan = now
        new_target = target_thread.get_target()
        print(f"\n[_tick_arm] Arm-{arm_id} replan → {new_target.tolist()}")

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

        plan, iq, ist, wk, fr, st = _start_new_plan(
            model, data, ee_site_id, shoulder_joint_id, qpos_adr, arm_id,
            target_pos=new_target, target_phi_rad=target_phi_rad,
            current_q=state.current_q, current_frame=state.frame,
            old_plan=state.plan,
        )
        state.plan               = plan
        state.ik_queue           = iq
        state.ik_status          = ist
        state.worker             = wk
        state.frame              = 0
        state.sim_time_at_replan = st

    if state.plan is not None:
        elapsed_sim  = data.time - state.sim_time_at_replan
        target_frame = min(int(elapsed_sim / state.plan.dt), state.plan.N - 1)

        step_limit = int(model.opt.timestep / state.plan.dt) + 2
        consumed = 0
        while state.frame <= target_frame and consumed < step_limit:
            try:
                q = state.ik_queue.get_nowait()
                state.current_q = q
                state.frame    += 1
                consumed       += 1
            except queue.Empty:
                break

        if state.frame >= state.plan.N:
            state.frame = state.plan.N - 1

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
                   target_angle_deg_a: float, target_angle_deg_b: float,
                   replan_interval: float,
                   data_lock: threading.Lock):
    phi_a = np.deg2rad(target_angle_deg_a)
    phi_b = np.deg2rad(target_angle_deg_b)
    mujoco.mj_forward(model, data)
    counter = 0

    def _init_arm(arm_id, ee_site_id, shoulder_joint_id, qpos_adr, act_id, t_thread, phi):
        tpos = t_thread.get_target()
        # BUG 4 FIX: Read the actual joint configuration from qpos instead of
        # zeroing current_q.  Zeroing forced the position servo to drive joints
        # to zero on the very first tick, causing a startup impulse when the
        # arm's XML specifies a non-zero initial pose.
        current_q = read_actual_q(data, qpos_adr, arm_id)

        plan, iq, ist, wk, fr, st = _start_new_plan(
            model, data, ee_site_id, shoulder_joint_id, qpos_adr, arm_id,
            target_pos=tpos, target_phi_rad=phi,
            current_q=current_q, current_frame=0, old_plan=None,
        )
        # BUG 4 FIX: Do NOT zero data.ctrl here.  Initialise the actuators from
        # the actual configuration so there is no initial impulse toward zero.
        for name, val in zip([a.format(arm_id) for a in ACTUATOR_NAMES_TMPL], current_q):
            data.ctrl[act_id[name]] = val

        return ArmState(
            arm_id=arm_id, replan_interval=replan_interval,
            plan=plan, ik_queue=iq, ik_status=ist, worker=wk,
            frame=fr, sim_time_at_replan=st,
            current_q=current_q,              # Bug 4 fix: real q, not zeros
            last_replan=time.perf_counter(),
        )

    state_a = _init_arm(ARM_IDS[0], ee_site_a, shoulder_joint_a, qpos_adr_a, act_id_a,
                         target_thread_a, phi_a)
    state_b = _init_arm(ARM_IDS[1], ee_site_b, shoulder_joint_b, qpos_adr_b, act_id_b,
                         target_thread_b, phi_b)

    t0_wall = time.perf_counter()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # BUG 2 FIX: Wrap mj_step and all data.* reads in data_lock so the
            # MovingTargetThread never reads geom_xpos / geom_xmat while mj_step
            # is writing them.  Without the lock, a partially updated geom_xmat
            # produced a nonsense rotation matrix → the target position jumped to
            # a wrong world location with no physical cause.
            # BUG 7 FIX: _tick_arm (→ _start_new_plan) also reads data.xanchor,
            # data.site_xpos, data.qvel, and data.time; holding data_lock here
            # covers those reads as well.
            with data_lock:
                state_a = _tick_arm(state_a, model, data,
                                     qpos_adr_a, act_id_a, ee_site_a, shoulder_joint_a,
                                     target_thread_a, phi_a)
                state_b = _tick_arm(state_b, model, data,
                                     qpos_adr_b, act_id_b, ee_site_b, shoulder_joint_b,
                                     target_thread_b, phi_b)

                mujoco.mj_step(model, data)

                shoulder_a = data.xanchor[shoulder_joint_a][[0, 1]].copy()
                shoulder_b = data.xanchor[shoulder_joint_b][[0, 1]].copy()
                # BUG 10 FIX: Capture actual Z of each shoulder for visualisation.
                shoulder_a_z = float(data.xanchor[shoulder_joint_a][2])
                shoulder_b_z = float(data.xanchor[shoulder_joint_b][2])

                ee_a = data.site_xpos[ee_site_a].copy()
                ee_b = data.site_xpos[ee_site_b].copy()
                sim_time = data.time

            # Draw trajectories outside the lock — viewer.sync() does not touch
            # data, so releasing the lock first shortens the critical section.
            draw_trajectories(viewer,
                              state_a.plan, state_a.frame,
                              state_b.plan, state_b.frame,
                              shoulder_a, shoulder_b,
                              shoulder_a_z, shoulder_b_z)
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
                    qpos_addr = model.jnt_qposadr[i]
                    print(f"{joint_name:20s} : "
                        f"{data.qpos[qpos_addr]: .6f} rad "
                        f"({np.degrees(data.qpos[qpos_addr]): .2f} deg)"
                    )

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

    # BUG 2 FIX: Create one shared data_lock and pass it to execute_motion so
    # the main loop wraps mj_step (and all data reads) in the same lock that
    # MovingTargetThread uses when reading geom_xpos / geom_xmat.
    data_lock  = threading.Lock()
    target_thread_a = MovingTargetThread(model, data, data_lock, arm_index=0)
    target_thread_b = MovingTargetThread(model, data, data_lock, arm_index=1)
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
            target_angle_deg_a=TARGET_ANGLE_DEG_A,
            target_angle_deg_b=TARGET_ANGLE_DEG_B,
            replan_interval=REPLAN_INTERVAL,
            data_lock=data_lock,          # Bug 2 fix: pass lock to main loop
        )
    finally:
        target_thread_a.stop()
        target_thread_b.stop()
        print("[Main] Target threads stopped.")

if __name__ == "__main__":
    main()