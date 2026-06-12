import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
import queue
from dataclasses import dataclass, field
from typing import Optional

XML_PATH = "main_space_dual.xml"
DOF3_JOINTS_TMPL    = ["joint_hindarm_{}", "joint_forearm_{}", "joint_hand_{}"]
ALL_JOINTS_TMPL     = ["joint_hindarm_{}", "joint_forearm_{}", "joint_hand_{}"]
ACTUATOR_NAMES_TMPL = ["hindarm_ctrl_{}", "forearm_ctrl_{}", "hand_ctrl_{}"]
ARM_IDS = ["a", "b"]

TARGET_A = [0, 0.55]
TARGET_B = [0, 0.55]
TARGET_ANGLE_DEG_A = 45.0
TARGET_ANGLE_DEG_B = 135.0

FREQUENCY       = 3000
IK_QUEUE_SIZE   = 30000     # ≥ T_MAX * FREQUENCY
REPLAN_INTERVAL = 2

# Trajectory shape
V_CRUISE    = 0.15
ALPHA_CURVE = 0.10
DAMP_RADIUS = 0.05
T_MIN       = 0.1
T_MAX       = 4.0

# Arm link lengths
L1 = 0.22112
L2 = 0.12750 + 0.09500
L3 = 0.06500

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
    """All mutable runtime state for one arm."""
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
    def __init__(self, initial_pos, arm, update_interval=0.5):
        super().__init__(daemon=True)
        self._pos             = np.asarray(initial_pos, dtype=float).copy()
        self._lock            = threading.Lock()
        self._update_interval = update_interval
        self.arm = arm
        self._stop_event      = threading.Event()

    def _compute_next_position(self, current_pos, elapsed):
        # return current_pos.copy()
        
        # Example: slow circular motion — uncomment to test
        if self.arm == ARM_IDS[0]:
            cx, cz = TARGET_A
            r = 0.05
            return np.array([cx + r * np.cos(0.3 * elapsed),
                            cz + r * np.sin(0.3 * elapsed)])
        cx, cz = TARGET_B
        r = 0.05
        return np.array([cx - r * np.cos(0.3 * elapsed),
                        cz - r * np.sin(0.3 * elapsed)])

    def run(self):
        start = time.perf_counter()
        while not self._stop_event.is_set():
            elapsed = time.perf_counter() - start
            new_pos = self._compute_next_position(self._pos.copy(), elapsed)
            with self._lock:
                self._pos = np.asarray(new_pos, dtype=float)
            time.sleep(self._update_interval)

    def get_target(self):
        with self._lock:
            return self._pos.copy()

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

def find_ee(model, arm_id):
    for name in [f"gripper_{arm_id}_Gripper",
                 f"gripper_Gripper_{arm_id}", f"Gripper_{arm_id}"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            print(f"[EE] Arm-{arm_id}: '{name}' (id={bid})")
            return bid
    all_bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
                  for i in range(model.nbody)]
    for name in all_bodies:
        if name and "ripper" in name and name.endswith(f"_{arm_id}"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                print(f"[EE] Arm-{arm_id}: scan -> '{name}' (id={bid})")
                return bid
    raise RuntimeError(f"Could not find gripper body for arm '{arm_id}'.\n"
                       f"Bodies: {all_bodies}")

def find_hindarm(model, arm_id):
    for name in [f"hindarm_{arm_id}_Hindarm", f"hindarm_Hindarm_{arm_id}"]:
        hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if hid >= 0:
            print(f"[Hind-arm] Arm-{arm_id}: '{name}' (id={hid})")
            return hid
    all_bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
                  for i in range(model.nbody)]
    for name in all_bodies:
        if name and "indarm" in name and name.endswith(f"_{arm_id}"):
            hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if hid >= 0:
                print(f"[Hind-arm] Arm-{arm_id}: scan -> '{name}' (id={hid})")
                return hid
    raise RuntimeError(f"Could not find hindarm body for arm '{arm_id}'.")

def read_actual_q(data, qpos_adr, arm_id):
    return np.array([
        data.qpos[qpos_adr[f"joint_hindarm_{arm_id}"]],
        data.qpos[qpos_adr[f"joint_forearm_{arm_id}"]],
        data.qpos[qpos_adr[f"joint_hand_{arm_id}"]],
    ])

def ee_world_velocity(model, data, ee_id, qpos_adr, arm_id):
    nv   = model.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, ee_id)
    vel = np.zeros(nv)
    for jname in [f"joint_hindarm_{arm_id}",
                  f"joint_forearm_{arm_id}",
                  f"joint_hand_{arm_id}"]:
        jid    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        dofadr = model.jnt_dofadr[jid]
        vel[dofadr] = data.qvel[dofadr]
    return (jacp @ vel)[[0, 2]]

# IK SOLVERS
def solve_ik_trig(target_xz, phi_world, elbow_up=True):
    target_x, target_z = target_xz
    x_wrist = target_x - L3 * np.sin(phi_world)
    z_wrist = target_z - L3 * np.cos(phi_world)

    cos_q2 = (x_wrist**2 + z_wrist**2 - L1**2 - L2**2) / (2.0 * L1 * L2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)

    if elbow_up:
        q2 = np.arccos(cos_q2)
    else:
        q2 = -np.arccos(cos_q2)

    alpha = np.arctan2(x_wrist, z_wrist)
    beta = np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    q1 = alpha - beta

    # End-effector orientation constraint
    q3 = phi_world - q1 - q2
    return np.array([q1, q2, q3])

def phi_world_from_q(q):
    return q[0] + q[1] + q[2]

# QUINTIC TRAJECTORY
def _quintic_blend(start_pos, target_pos, start_phi, target_phi,
                   start_vel, start_phi_vel, total_time, frequency):
    if start_vel is None:
        start_vel = np.zeros(2)
    start_vel = np.asarray(start_vel, dtype=float)
    T = float(total_time)
    T3 = T**3;  T4 = T**4;  T5 = T**5

    def _qc(p0, pf, v0):
        dp = pf - p0
        return (p0, v0, 0.0,
                (10*dp - T*(6*v0)) / T3,
                (-15*dp + T*(8*v0)) / T4,
                (6*dp  - T*(3*v0)) / T5)

    def _qe(c, t):
        a0, a1, a2, a3, a4, a5 = c
        return a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5

    t  = np.linspace(0, T, int(T * frequency))
    dp = target_pos - start_pos
    a0, a1, a2 = start_pos, start_vel, np.zeros(2)
    a3 = (10*dp - T*(6*start_vel)) / T3
    a4 = (-15*dp + T*(8*start_vel)) / T4
    a5 = (6*dp  - T*(3*start_vel)) / T5
    traj = (a0 + np.outer(t, a1) + np.outer(t**2, a2) +
            np.outer(t**3, a3) + np.outer(t**4, a4) + np.outer(t**5, a5))

    phi_c      = _qc(start_phi, target_phi, start_phi_vel)
    angle_traj = np.array([_qe(phi_c, ti) for ti in t])
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
    print(f"[Plan Trajectory] {N} pts | Dist={dist:.3f}m | T={total_time:.2f}s | "
          f"pos: {np.round(current_pos,4).tolist()} -> {np.round(target_pos,4).tolist()} | "
          f"phi: {np.degrees(start_phi_rad):.1f} -> {np.degrees(target_phi_rad):.1f}°")
    return TrajectoryPlan(
        cartesian_traj=cartesian_traj, angle_traj=angle_traj, dt=dt, N=N,
        current_pos=current_pos, target_pos=target_pos,
        target_angle_rad=target_phi_rad, total_time=total_time, frequency=frequency,
    )

# IK WORKER THREAD
class IKWorker(threading.Thread):
    def __init__(self, cartesian_traj_local, angle_traj, ik_queue, status, arm_id):
        super().__init__(daemon=True)
        if arm_id == ARM_IDS[0]:
            self.cartesian_traj_local = cartesian_traj_local
        else:
            self.cartesian_traj_local = cartesian_traj_local.copy()
            self.cartesian_traj_local[:, 0] *= -1
        self.angle_traj           = angle_traj
        self.ik_queue             = ik_queue
        self.status               = status
        self._solver = solve_ik_trig

    def run(self):
        N = len(self.cartesian_traj_local)
        for i in range(N):
            if self.status.get("cancel", False):
                return
            q = self._solver(self.cartesian_traj_local[i], self.angle_traj[i])
            try:
                self.ik_queue.put(q, timeout=1.0)
            except queue.Full:
                if self.status.get("cancel", False):
                    return
        self.status["done"] = True

# REPLAN HELPER
def _start_new_plan(model, data, ee_id, hindarm_id, qpos_adr, arm_id,
                    target_pos, target_phi_rad,
                    current_q, current_frame, old_plan):
    """
    Build a new plan and launch IKWorker.

    Returns a 6-tuple whose last element is now data.time (simulation time
    when the plan was created) instead of time.perf_counter().  This is stored
    in ArmState.sim_time_at_replan and used for frame synchronisation.
    """
    current_pos = data.xpos[ee_id][[0, 2]].copy()

    if old_plan is not None and current_frame < old_plan.N:
        start_phi_rad = old_plan.angle_traj[current_frame]
        start_phi_vel = _quintic_angle_vel_at(old_plan, current_frame)
    elif current_q is not None:
        phi_fn        = phi_world_from_q
        start_phi_rad = phi_fn(current_q)
        start_phi_vel = 0.0
    else:
        actual_q      = read_actual_q(data, qpos_adr, arm_id)
        phi_fn        = phi_world_from_q
        start_phi_rad = phi_fn(actual_q)
        start_phi_vel = 0.0

    start_vel = ee_world_velocity(model, data, ee_id, qpos_adr, arm_id)
    plan = plan_trajectory(
        current_pos=current_pos, target_pos=target_pos,
        start_phi_rad=start_phi_rad, target_phi_rad=target_phi_rad,
        start_vel=start_vel, start_phi_vel=start_phi_vel,
        frequency=FREQUENCY,
    )

    shifted_frame = data.xpos[hindarm_id][[0, 2]].copy()
    local_traj    = plan.cartesian_traj - shifted_frame

    print(f"[Replan-{arm_id}] EE={current_pos.tolist()} vel={start_vel.tolist()}")
    print(f"[Replan-{arm_id}] phi: {np.degrees(start_phi_rad):.1f}° → "
          f"{np.degrees(target_phi_rad):.1f}°  "
          f"(vel {np.degrees(start_phi_vel):.1f}°/s)")

    ik_queue  = queue.Queue(maxsize=IK_QUEUE_SIZE)
    ik_status = {"done": False, "cancel": False}
    worker    = IKWorker(local_traj, plan.angle_traj, ik_queue, ik_status, arm_id)
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

def _add_segment(scn, p1_xz, p2_xz, rgba, width_px=2.0):
    if scn.ngeom >= scn.maxgeom - 4:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE,
                        np.zeros(3, np.float64), np.zeros(3, np.float64), _EYE, rgba)
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width_px,
                         np.array([p1_xz[0], 0.0, p1_xz[1]], np.float64),
                         np.array([p2_xz[0], 0.0, p2_xz[1]], np.float64))
    scn.ngeom += 1

def _add_sphere(scn, pos_xz, rgba, size=0.008):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([size]*3, np.float64),
                        np.array([pos_xz[0], 0.0, pos_xz[1]], np.float64),
                        _EYE, rgba)
    scn.ngeom += 1

def draw_trajectories(viewer, plan_a, frame_a, plan_b, frame_b):
    if not hasattr(viewer, "user_scn"):
        return
    scn = viewer.user_scn
    scn.ngeom = 0
    for plan, frame, arm_id in ((plan_a, frame_a, ARM_IDS[0]), (plan_b, frame_b, ARM_IDS[1])):
        if plan is None:
            continue
        traj       = plan.cartesian_traj
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
            _add_segment(scn, traj[i], traj[i+1], col["done"], 1.5)
        for i in range(max(frame, 0), N-1, step_rm):
            _add_segment(scn, traj[i], traj[i+1], col["pending"], 3.0)
        _add_sphere(scn, traj[min(frame, N-1)], col["current"], 0.010)
        _add_sphere(scn, traj[-1],              col["target"],  0.012)

# PER-ARM TICK
def _tick_arm(state: ArmState, model, data, qpos_adr, act_id, ee_id, hindarm_id,
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
            model, data, ee_id, hindarm_id, qpos_adr, arm_id,
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

        while state.frame <= target_frame:
            try:
                q = state.ik_queue.get_nowait()
                state.current_q = q
                state.frame    += 1
            except queue.Empty:
                break

        if state.frame >= state.plan.N:
            state.frame = state.plan.N - 1

    # Apply actuator targets
    if state.current_q is not None:
        act_names = [a.format(arm_id) for a in ACTUATOR_NAMES_TMPL]
        for name, val in zip(act_names, state.current_q):
            data.ctrl[act_id[name]] = val
    return state

# EXECUTE MOTION
def execute_motion(model, data,
                   qpos_adr_a, act_id_a, ee_id_a, hindarm_id_a,
                   qpos_adr_b, act_id_b, ee_id_b, hindarm_id_b,
                   target_thread_a: MovingTargetThread,
                   target_thread_b: MovingTargetThread,
                   target_angle_deg_a: float,
                   target_angle_deg_b: float,
                   replan_interval: float):
    phi_a = np.deg2rad(target_angle_deg_a)
    phi_b = np.deg2rad(target_angle_deg_b)
    mujoco.mj_forward(model, data)

    # Initialise both arms
    def _init_arm(arm_id, ee_id, hindarm_id, qpos_adr, act_id, t_thread, phi):
        tpos = t_thread.get_target()
        plan, iq, ist, wk, fr, st = _start_new_plan(
            model, data, ee_id, hindarm_id, qpos_adr, arm_id,
            target_pos=tpos, target_phi_rad=phi,
            current_q=None, current_frame=0, old_plan=None,
        )
        for name in [a.format(arm_id) for a in ACTUATOR_NAMES_TMPL]:
            data.ctrl[act_id[name]] = 0.0
        return ArmState(
            arm_id=arm_id, replan_interval=replan_interval,
            plan=plan, ik_queue=iq, ik_status=ist, worker=wk,
            frame=fr, sim_time_at_replan=st,
            current_q=np.zeros(3),
            last_replan=time.perf_counter(),
        )
    state_a = _init_arm(ARM_IDS[0], ee_id_a, hindarm_id_a, qpos_adr_a, act_id_a,
                         target_thread_a, phi_a)
    state_b = _init_arm(ARM_IDS[1], ee_id_b, hindarm_id_b, qpos_adr_b, act_id_b,
                         target_thread_b, phi_b)
    
    t0_wall = time.perf_counter()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            state_a = _tick_arm(state_a, model, data,
                                 qpos_adr_a, act_id_a, ee_id_a, hindarm_id_a,
                                 target_thread_a, phi_a)
            state_b = _tick_arm(state_b, model, data,
                                 qpos_adr_b, act_id_b, ee_id_b, hindarm_id_b,
                                 target_thread_b, phi_b)

            mujoco.mj_step(model, data)
            draw_trajectories(viewer, state_a.plan, state_a.frame, state_b.plan, state_b.frame)
            viewer.sync()

            sleep_t = data.time - (time.perf_counter() - t0_wall)
            if sleep_t > 0:
                time.sleep(sleep_t)

            # Status readout
            ee_a  = data.xpos[ee_id_a]
            ee_b  = data.xpos[ee_id_b]
            tgt_a = target_thread_a.get_target()
            tgt_b = target_thread_b.get_target()
            q_a   = np.degrees(state_a.current_q) if state_a.current_q is not None else np.zeros(3)
            q_b   = np.degrees(state_b.current_q) if state_b.current_q is not None else np.zeros(3)
            phi_a_now = (state_a.plan.angle_traj[min(state_a.frame, state_a.plan.N-1)]
                         if state_a.plan else 0.0)
            phi_b_now = (state_b.plan.angle_traj[min(state_b.frame, state_b.plan.N-1)]
                         if state_b.plan else 0.0)
            print(f"\r[Sim] t={data.time:.3f}s  "
                f"A: EE=({ee_a[0]:.3f},{ee_a[2]:.3f}) "
                f"φ={np.degrees(phi_a_now):.1f}° "
                f"q=[{q_a[0]:.1f} {q_a[1]:.1f} {q_a[2]:.1f}]°"
            )
            print(f"\r[Sim] t={data.time:.3f}s  "
                f"B: EE=({ee_b[0]:.3f},{ee_b[2]:.3f})  "
                f"φ={np.degrees(phi_b_now):.1f}° "
                f"q=[{q_b[0]:.1f} {q_b[1]:.1f} {q_b[2]:.1f}]°"
            )
    print("\n[Execute motion] Viewer closed.")

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    qpos_adr_a, act_id_a = build_maps(model, ARM_IDS[0])
    qpos_adr_b, act_id_b = build_maps(model, ARM_IDS[1])
    ee_id_a      = find_ee(model, ARM_IDS[0])
    ee_id_b      = find_ee(model, ARM_IDS[1])
    hindarm_id_a = find_hindarm(model, ARM_IDS[0])
    hindarm_id_b = find_hindarm(model, ARM_IDS[1])

    mujoco.mj_forward(model, data)

    ee_a_world = data.xpos[ee_id_a][[0, 2]].copy()
    ee_b_world = data.xpos[ee_id_b][[0, 2]].copy()

    print(f"[Main] EE-A: {ee_a_world}  → target_A = {TARGET_A}")
    print(f"[Main] EE-B: {ee_b_world}  → target_B = {TARGET_B}")

    target_thread_a = MovingTargetThread(TARGET_A, ARM_IDS[0], update_interval=0.5)
    target_thread_b = MovingTargetThread(TARGET_B, ARM_IDS[1], update_interval=0.5)
    target_thread_a.start()
    target_thread_b.start()

    try:
        execute_motion(
            model=model, data=data,
            qpos_adr_a=qpos_adr_a, act_id_a=act_id_a,
            ee_id_a=ee_id_a, hindarm_id_a=hindarm_id_a,
            qpos_adr_b=qpos_adr_b, act_id_b=act_id_b,
            ee_id_b=ee_id_b, hindarm_id_b=hindarm_id_b,
            target_thread_a=target_thread_a,
            target_thread_b=target_thread_b,
            target_angle_deg_a=TARGET_ANGLE_DEG_A,
            target_angle_deg_b=TARGET_ANGLE_DEG_B,
            replan_interval=REPLAN_INTERVAL,
        )
    finally:
        target_thread_a.stop()
        target_thread_b.stop()
        print("[Main] Target threads stopped.")

if __name__ == "__main__":
    main()