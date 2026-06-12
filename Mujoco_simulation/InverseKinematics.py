import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
import queue
from dataclasses import dataclass

XML_PATH        = "main.xml"
DOF3_JOINTS     = ["joint_hindarm", "joint_forearm", "joint_hand"]
ALL_JOINTS      = ["joint_hip", "joint_hindarm", "joint_forearm", "joint_wrist", "joint_hand"]
ACTUATOR_NAMES  = ["hip_ctrl", "hindarm_ctrl", "forearm_ctrl", "wrist_ctrl", "hand_ctrl"]

FREQUENCY       = 300
IK_QUEUE_SIZE   = 9000   # Look-ahead buffer (>= T_MAX * FREQUENCY / 2)
REPLAN_INTERVAL = 30

V_CRUISE    = 0.15   # m/s
ALPHA_CURVE = 0.10   # fraction  (10 % max lateral deviation)
DAMP_RADIUS = 0.05   # metres
T_MIN       = 0.1    # seconds
T_MAX       = 4.0    # seconds

# TRAJECTORY PLAN
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

# MOVING TARGET THREAD
class MovingTargetThread(threading.Thread):
    """
    Simulates a moving target by updating its [X, Z] position over time.
    Thread-safe access via get_target().
    """
    def __init__(self, initial_pos, update_interval=0.5):
        super().__init__(daemon=True)
        self._pos             = np.asarray(initial_pos, dtype=float).copy()
        self._lock            = threading.Lock()
        self._update_interval = update_interval
        self._stop_event      = threading.Event()

    def _compute_next_position(self, current_pos: np.ndarray, elapsed: float) -> np.ndarray:
        """
        Replace this body with perception / object-tracking code when ready.
        """
        return current_pos.copy()

        # Example: Slow circular motion
        # cx, cz = 0.42482, 0.42089
        # r = 0.05
        # return np.array([cx + r * np.cos(0.3 * elapsed),
        #                   cz + r * np.sin(0.3 * elapsed)])

    def run(self):
        start = time.perf_counter()
        while not self._stop_event.is_set():
            elapsed = time.perf_counter() - start
            new_pos = self._compute_next_position(self._pos.copy(), elapsed)
            with self._lock:
                self._pos = np.asarray(new_pos, dtype=float)
            time.sleep(self._update_interval)

    def get_target(self) -> np.ndarray:
        with self._lock:
            return self._pos.copy()

    def stop(self):
        self._stop_event.set()

# MAP BUILDERS
def build_maps(model):
    qpos_adr = {}
    for j in ALL_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        qpos_adr[j] = model.jnt_qposadr[jid]

    act_id = {}
    for a in ACTUATOR_NAMES:
        act_id[a] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)

    return qpos_adr, act_id

def find_ee(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_Gripper")

def read_actual_q(data, qpos_adr):
    return np.array([
        data.qpos[qpos_adr["joint_hindarm"]],
        data.qpos[qpos_adr["joint_forearm"]],
        data.qpos[qpos_adr["joint_hand"]],
    ])

def ee_world_velocity(model, data, ee_id, qpos_adr):
    nv = model.nv
    jacp = np.zeros((3, nv))   # positional Jacobian, world frame
    jacr = np.zeros((3, nv))   # rotational (unused)
    mujoco.mj_jacBody(model, data, jacp, jacr, ee_id)

    # Collect qvel for the joints we control
    joint_names = ["joint_hindarm", "joint_forearm", "joint_hand"]
    vel = np.zeros(nv)
    for jname in joint_names:
        jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        dofadr = model.jnt_dofadr[jid]
        vel[dofadr] = data.qvel[dofadr]

    v_world = jacp @ vel   # (3,) world-frame velocity
    return v_world[[0, 2]]  # [Vx, Vz]

# IK SOLVER
def solve_ik_trig(target_xz, phi_world, elbow_up=True):
    L1 = 0.22112
    L2 = 0.12750 + 0.09500
    L3 = 0.06500

    target_x, target_z = target_xz
    phi_geom = np.pi / 2.0 - phi_world

    x_wrist = target_x - L3 * np.cos(phi_geom)
    z_wrist = target_z - L3 * np.sin(phi_geom)

    cos_q2 = (x_wrist**2 + z_wrist**2 - L1**2 - L2**2) / (2.0 * L1 * L2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    q2      = -np.arccos(cos_q2) if elbow_up else np.arccos(cos_q2)

    alpha = np.arctan2(z_wrist, x_wrist)
    beta  = np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    q1    = alpha - beta
    q3    = phi_geom - q1 - q2

    q1_mj = np.pi / 2.0 - q1
    q2_mj = -q2
    q3_mj = -q3

    return np.array([q1_mj, q2_mj, q3_mj])


def phi_world_from_q(q):
    """
    Recover phi_world (radians) from a MuJoCo joint solution.
    Inverse of the MuJoCo convention applied in solve_ik_trig.

    q : (3,)  [q1_mj, q2_mj, q3_mj]
    """
    q1_geom = np.pi / 2.0 - q[0]   # hindarm
    q2_geom = -q[1]                 # forearm
    q3_geom = q[2]   # hand  (matches q3_mj = q3 - pi/2)
    phi_geom = q1_geom + q2_geom + q3_geom
    return np.pi / 2.0 - phi_geom   # phi_world = π/2 − phi_geom

# QUINTIC TRAJECTORY
def _quintic_blend(
    start_pos,
    target_pos,
    start_phi,          # phi_world radians  ← actual current angle
    target_phi,         # phi_world radians  ← desired final angle
    start_vel,          # (2,) m/s, or None  ← actual current EE velocity
    start_phi_vel,      # rad/s              ← actual current angular velocity
    total_time,
    frequency,
):
    """
    Quintic (minimum-jerk) trajectory with full C2 boundary conditions.

    Boundary conditions
    -------------------
    Position : p(0) = start_pos,  p(T) = target_pos
    Velocity : p'(0)= start_vel,  p'(T)= 0   (arrive at rest)
    Accel    : p''(0)= 0,         p''(T)= 0

    Same BC structure is applied to the angle axis so hand pitch is also
    C2-continuous across re-plans.

    All angles are in phi_world RADIANS — no degree conversion inside here.

    Returns
    -------
    traj      : (N, 2) ndarray  positions [X, Z]
    angle_traj: (N,)   ndarray  phi_world in radians
    """
    if start_vel is None:
        start_vel = np.zeros(2)
    start_vel = np.asarray(start_vel, dtype=float)

    T  = float(total_time)
    T2 = T ** 2
    T3 = T ** 3
    T4 = T ** 4
    T5 = T ** 5

    # ── Quintic coefficients ─────────────────────────────────────────
    # Standard derivation with a2=0, a2''(T)=0:
    #   a0 = p0
    #   a1 = v0
    #   a2 = 0
    #   a3 = (10Δp − T·(6v0)) / T³
    #   a4 = (−15Δp + T·(8v0)) / T⁴
    #   a5 = (  6Δp − T·(3v0)) / T⁵

    def _quintic_coeffs_scalar(p0, pf, v0):
        dp = pf - p0
        return p0, v0, 0.0, \
               (10*dp - T*(6*v0)) / T3, \
               (-15*dp + T*(8*v0)) / T4, \
               (6*dp  - T*(3*v0)) / T5

    def _eval_quintic_scalar(coeffs, t):
        a0, a1, a2, a3, a4, a5 = coeffs
        return a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5

    def _eval_quintic_scalar_vel(coeffs, t):
        """First derivative for angular velocity tracking."""
        _, a1, a2, a3, a4, a5 = coeffs
        return a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4

    t = np.linspace(0, T, int(T * frequency))  # (N,)

    # -- Cartesian (vector) --
    dp = target_pos - start_pos                 # (2,)
    a0 = start_pos
    a1 = start_vel
    a2 = np.zeros(2)
    a3 = (10*dp - T*(6*start_vel)) / T3
    a4 = (-15*dp + T*(8*start_vel)) / T4
    a5 = (6*dp  - T*(3*start_vel)) / T5

    traj = (
        a0
        + np.outer(t,    a1)
        + np.outer(t**2, a2)
        + np.outer(t**3, a3)
        + np.outer(t**4, a4)
        + np.outer(t**5, a5)
    )   # (N, 2)

    # -- Angle (scalar quintic, same BC structure, velocity-continuous) --
    phi_coeffs = _quintic_coeffs_scalar(start_phi, target_phi, start_phi_vel)
    angle_traj = np.array([_eval_quintic_scalar(phi_coeffs, ti) for ti in t])  # (N,)

    return traj, angle_traj


def _quintic_angle_vel_at(plan: TrajectoryPlan, frame: int) -> float:
    """
    Return the angular velocity (rad/s, phi_world) at the given frame
    by finite-differencing the stored angle_traj.
    Used to seed start_phi_vel at re-plan time.
    """
    n = plan.N
    if n < 2:
        return 0.0
    f = min(max(frame, 0), n - 1)
    if f == 0:
        return (plan.angle_traj[1] - plan.angle_traj[0]) / plan.dt
    if f == n - 1:
        return (plan.angle_traj[-1] - plan.angle_traj[-2]) / plan.dt
    return (plan.angle_traj[f + 1] - plan.angle_traj[f - 1]) / (2.0 * plan.dt)


def compute_total_time(start, target, start_vel):
    """
    Return a distance-proportional trajectory duration that keeps the
    quintic arc below ALPHA_CURVE * dist regardless of initial speed.

    T = min(dist/V_CRUISE,  ALPHA_CURVE*dist / (0.135*|v0|))
    clamped to [T_MIN, T_MAX].
    The 0.135 coefficient is the empirical peak-lateral factor of the quintic.
    """
    dist = float(np.linalg.norm(np.asarray(target) - np.asarray(start)))
    if dist < 1e-4:
        return T_MIN
    T_cruise  = dist / V_CRUISE
    v_mag     = float(np.linalg.norm(start_vel))
    T_vel_cap = (ALPHA_CURVE * dist / (0.135 * v_mag)) if v_mag > 1e-4 else T_MAX
    return float(np.clip(min(T_cruise, T_vel_cap), T_MIN, T_MAX))


def damp_start_velocity(start, target, start_vel):
    """
    Smoothly taper start_vel to zero inside DAMP_RADIUS of the target.
    Scale = (dist / DAMP_RADIUS)^2  →  0 at dist=0, 1 at dist=DAMP_RADIUS.
    Prevents wide arcs when the arm is nearly at the goal.
    """
    dist = float(np.linalg.norm(np.asarray(target) - np.asarray(start)))
    if dist >= DAMP_RADIUS:
        return start_vel
    return start_vel * (dist / DAMP_RADIUS) ** 2


# ============================================================
# PUBLIC INTERFACE: plan_trajectory
# ============================================================

def plan_trajectory(
    current_pos,
    target_pos,
    start_phi_rad,        # phi_world radians  – ACTUAL current angle
    target_phi_rad,       # phi_world radians  – desired final angle
    start_vel   = None,   # (2,) m/s or None
    start_phi_vel = 0.0,  # rad/s
    frequency   = FREQUENCY,
) -> TrajectoryPlan:
    """
    Build a C2-continuous minimum-jerk trajectory with adaptive duration.
    total_time is computed from distance and start_vel via compute_total_time()
    so that short moves don't produce deep arcs.
    All angle parameters are in phi_world RADIANS.
    """
    current_pos = np.asarray(current_pos, dtype=float)
    target_pos  = np.asarray(target_pos,  dtype=float)
    if start_vel is None:
        start_vel = np.zeros(2)
    start_vel = np.asarray(start_vel, dtype=float)

    if current_pos.shape != (2,) or target_pos.shape != (2,):
        raise ValueError("current_pos and target_pos must both be (2,) arrays: [X, Z]")

    # Damp velocity near target, then compute a proportional duration
    start_vel  = damp_start_velocity(current_pos, target_pos, start_vel)
    total_time = compute_total_time(current_pos, target_pos, start_vel)

    cartesian_traj, angle_traj = _quintic_blend(
        start_pos    = current_pos,
        target_pos   = target_pos,
        start_phi    = start_phi_rad,
        target_phi   = target_phi_rad,
        start_vel    = start_vel,
        start_phi_vel= start_phi_vel,
        total_time   = total_time,
        frequency    = frequency,
    )

    dt = 1.0 / frequency
    N  = len(cartesian_traj)

    dist_m = float(np.linalg.norm(target_pos - current_pos))
    print(
        f"[plan_trajectory] {N} pts | dist={dist_m:.3f}m T={total_time:.2f}s | "
        f"pos: {np.round(current_pos,4).tolist()} -> {np.round(target_pos,4).tolist()} | "
        f"phi: {np.degrees(start_phi_rad):.1f}->{np.degrees(target_phi_rad):.1f}deg"
    )

    return TrajectoryPlan(
        cartesian_traj   = cartesian_traj,
        angle_traj       = angle_traj,     # phi_world radians
        dt               = dt,
        N                = N,
        current_pos      = current_pos,
        target_pos       = target_pos,
        target_angle_rad = target_phi_rad,
        total_time       = total_time,
        frequency        = frequency,
    )

# ============================================================
# IK WORKER THREAD
# ============================================================

class IKWorker(threading.Thread):
    """
    Background thread: pre-computes IK for every trajectory point
    and pushes (frame_index, q) tuples into ik_queue.
    angle_traj must be in phi_world RADIANS (passed straight to solve_ik_trig).
    """

    def __init__(self, cartesian_traj_local, angle_traj, ik_queue, status):
        super().__init__(daemon=True)
        self.cartesian_traj_local = cartesian_traj_local
        self.angle_traj           = angle_traj   # phi_world radians
        self.ik_queue             = ik_queue
        self.status               = status       # {"done": False, "cancel": False}

    def run(self):
        N = len(self.cartesian_traj_local)
        for i in range(N):
            if self.status.get("cancel", False):
                print(f"\n[IKWorker] Cancelled at frame {i}/{N}.")
                return
            q = solve_ik_trig(self.cartesian_traj_local[i], self.angle_traj[i])
            self.ik_queue.put((i, q))   # blocks when buffer full (back-pressure)

        self.status["done"] = True
        print(f"\n[IKWorker] All {N} frames computed.")

# ============================================================
# REPLAN HELPER
# ============================================================

def _start_new_plan(
    model, data, ee_id, qpos_adr,
    target_pos,
    target_phi_rad,     # phi_world radians
    current_q,          # (3,) last IK solution, or None on first call
    current_frame,      # frame index in the OLD plan (for angular velocity)
    old_plan,           # TrajectoryPlan or None
):
    """
    Snapshot the current sim state, build a fresh TrajectoryPlan, launch IKWorker.

    The start angle is read from old_plan.angle_traj[current_frame] when
    available — this is the single source of truth and requires NO back-
    conversion from joint angles.

    The start angular velocity is estimated by finite-differencing the old
    plan's angle_traj at current_frame via _quintic_angle_vel_at().

    Returns
    -------
    plan       : TrajectoryPlan
    ik_queue   : queue.Queue
    ik_status  : dict
    worker     : IKWorker (already started)
    frame      : int  (0)
    wall_start : float
    """
    current_pos = data.xpos[ee_id][[0, 2]].copy()

    # ── Start angle: read directly from the trajectory we were following ──
    if old_plan is not None and current_frame < old_plan.N:
        start_phi_rad = old_plan.angle_traj[current_frame]  # phi_world radians, exact
        start_phi_vel = _quintic_angle_vel_at(old_plan, current_frame)
    elif current_q is not None:
        # Fallback: recover from joint angles (first call where no old_plan)
        start_phi_rad = phi_world_from_q(current_q)
        start_phi_vel = 0.0
    else:
        # Very first call: read the ACTUAL current angle from joint state.
        # Using target_phi here would snap the angle 225 deg in frame 0 → collision.
        actual_q      = read_actual_q(data, qpos_adr)
        start_phi_rad = phi_world_from_q(actual_q)
        start_phi_vel = 0.0

    # ── EE velocity: world-frame via Jacobian (cvel is LOCAL frame — wrong) ──
    start_vel = ee_world_velocity(model, data, ee_id, qpos_adr)

    plan = plan_trajectory(
        current_pos   = current_pos,
        target_pos    = target_pos,
        start_phi_rad = start_phi_rad,
        target_phi_rad= target_phi_rad,
        start_vel     = start_vel,
        start_phi_vel = start_phi_vel,
        frequency     = FREQUENCY,
    )

    # Hindarm-local frame shift for IK
    hindarm_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hindarm_Hindarm")
    shifted_frame = data.xpos[hindarm_id][[0, 2]].copy()
    local_traj    = plan.cartesian_traj - shifted_frame

    print(f"[Replan] EE={current_pos.tolist()}  vel={start_vel.tolist()}")
    print(f"[Replan] phi_world: {np.degrees(start_phi_rad):.1f}deg -> "
          f"{np.degrees(target_phi_rad):.1f}deg  "
          f"(vel {np.degrees(start_phi_vel):.1f}deg/s)")

    ik_queue  = queue.Queue(maxsize=IK_QUEUE_SIZE)
    ik_status = {"done": False, "cancel": False}
    worker    = IKWorker(local_traj, plan.angle_traj, ik_queue, ik_status)
    worker.start()

    return plan, ik_queue, ik_status, worker, 0, time.perf_counter()


# ============================================================
# TRAJECTORY VISUALISER
# ============================================================

# Maximum line segments we ever draw.  viewer.user_scn.maxgeom is typically
# 1000; we stay well below that.  The trajectory is sub-sampled if needed.
_VIS_MAX_SEGS = 200

def draw_trajectory_line(viewer, plan: TrajectoryPlan, current_frame: int):
    """
    Draw the planned Cartesian trajectory as a coloured poly-line in the
    MuJoCo viewer each frame.  Uses the viewer.user_scn custom-geom API
    (mujoco >= 2.3.3) — no XML changes required.

    Visual encoding
    ---------------
    GREY  — already-executed portion  (frames 0 … current_frame)
    GREEN — remaining portion          (current_frame … N-1)
    WHITE sphere — current EE commanded pos
    RED   sphere — final target pos

    Correct API (verified):
        mjv_initGeom  sets rgba + type defaults
        mjv_connector sets pos/mat/size for a line between two 3-D points
        LINE width is in PIXELS (not metres)
    """
    scn = viewer.user_scn
    scn.ngeom = 0          # wipe all custom geoms from last frame

    traj = plan.cartesian_traj   # (N, 2) world-frame [X, Z]
    N    = plan.N

    # ── Sub-sample so we never exceed _VIS_MAX_SEGS line segments ────
    n_exec     = max(current_frame - 1, 0)
    n_rem      = max(N - 1 - current_frame, 0)
    total_segs = n_exec + n_rem
    if total_segs == 0:
        return

    ratio   = min(1.0, _VIS_MAX_SEGS / total_segs)
    step_ex = max(1, int(1.0 / ratio))
    step_rm = max(1, int(1.0 / ratio))

    # Colour definitions  [R, G, B, A]  float32
    GREY  = np.array([0.55, 0.55, 0.55, 0.5 ], dtype=np.float32)
    GREEN = np.array([0.20, 0.90, 0.30, 0.9 ], dtype=np.float32)
    WHITE = np.array([1.00, 1.00, 1.00, 1.0 ], dtype=np.float32)
    RED   = np.array([0.95, 0.20, 0.20, 1.0 ], dtype=np.float32)

    _EYE = np.eye(3, dtype=np.float64).flatten()   # identity rotation (reused)

    def _add_segment(p1_xz, p2_xz, rgba, width_px=2.0):
        """
        Add one LINE segment between two world [X, Z] points (Y = 0 plane).

        Correct call order (mjv_connector docs):
          1. mjv_initGeom  — sets rgba and type; zeroes pos/mat/size
          2. mjv_connector — sets pos, mat, size from the two endpoints
        """
        if scn.ngeom >= scn.maxgeom - 4:
            return
        g = scn.geoms[scn.ngeom]
        # Step 1: initialise with colour; size/pos/mat will be overwritten
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_LINE,
            np.zeros(3, dtype=np.float64),   # size — overwritten by mjv_connector
            np.zeros(3, dtype=np.float64),   # pos  — overwritten by mjv_connector
            _EYE,                            # mat  — overwritten by mjv_connector
            rgba,
        )
        # Step 2: set geometry from the two endpoints
        # Y = 0 for both points (planar arm lies in the X-Z plane)
        mujoco.mjv_connector(
            g,
            mujoco.mjtGeom.mjGEOM_LINE,
            width_px,
            np.array([p1_xz[0], 0.0, p1_xz[1]], dtype=np.float64),
            np.array([p2_xz[0], 0.0, p2_xz[1]], dtype=np.float64),
        )
        scn.ngeom += 1

    def _add_sphere(pos_xz, rgba, size=0.008):
        """Add a small sphere marker at world [X, Z] (Y = 0)."""
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([size, size, size], dtype=np.float64),
            np.array([pos_xz[0], 0.0, pos_xz[1]], dtype=np.float64),
            _EYE,
            rgba,
        )
        scn.ngeom += 1

    # ── Executed portion (grey, thin) ────────────────────────────────
    for i in range(0, min(current_frame, N - 1), step_ex):
        _add_segment(traj[i], traj[i + 1], GREY, width_px=1.5)

    # ── Remaining portion (green, thick) ─────────────────────────────
    for i in range(max(current_frame, 0), N - 1, step_rm):
        _add_segment(traj[i], traj[i + 1], GREEN, width_px=3.0)

    # ── Current commanded EE (white sphere) ──────────────────────────
    _add_sphere(traj[min(current_frame, N - 1)], WHITE, size=0.010)

    # ── Final target (red sphere) ─────────────────────────────────────
    _add_sphere(traj[-1], RED, size=0.012)

# ============================================================
# PUBLIC INTERFACE: execute_motion
def execute_motion(
    model, data, qpos_adr, act_id, ee_id,
    target_thread    : MovingTargetThread,
    target_angle_deg,
    replan_interval
):
    """
    Run the MuJoCo viewer loop.  Every replan_interval seconds it reads the
    latest target, cancels the current IK worker, and starts a fresh plan
    with full C2 continuity (position, velocity, acceleration).
    """
    target_phi_rad = np.deg2rad(target_angle_deg)

    # Initial plan
    target_pos = target_thread.get_target()
    current_q = None

    plan, ik_queue, ik_status, worker, frame, wall_start = _start_new_plan(
        model, data, ee_id, qpos_adr, target_pos = target_pos,
        target_phi_rad = target_phi_rad,
        current_q      = current_q,
        current_frame  = 0,
        old_plan       = None,
    )
    # Seed current_q from actual joint state so the first ctrl command is smooth
    current_q = read_actual_q(data, qpos_adr)
    last_replan_wall = time.perf_counter()

    # ── Viewer loop ──────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():

            # ── Re-plan check ────────────────────────────────────────
            now = time.perf_counter()
            if now - last_replan_wall >= replan_interval:
                last_replan_wall = now
                new_target = target_thread.get_target()
                print(f"\n[execute_motion] Re-planning -> target {new_target.tolist()}")

                ik_status["cancel"] = True
                while not ik_queue.empty():
                    try:
                        ik_queue.get_nowait()
                    except queue.Empty:
                        break

                plan, ik_queue, ik_status, worker, frame, wall_start = _start_new_plan(
                    model, data, ee_id, qpos_adr,
                    target_pos     = new_target,
                    target_phi_rad = target_phi_rad,
                    current_q      = current_q,
                    current_frame  = frame,   # position in old traj for angle read
                    old_plan       = plan,    # old plan for direct angle read
                )

            # ── Drain IK queue ───────────────────────────────────────
            if frame < plan.N:
                try:
                    idx, q = ik_queue.get_nowait()
                    if idx == frame:
                        current_q = q
                        frame    += 1
                    elif idx > frame:
                        ik_queue.put((idx, q))  # arrived early, put back
                    # idx < frame → stale, discard silently
                except queue.Empty:
                    pass   # IK not ready yet — hold pose

            if frame >= plan.N:
                frame = plan.N - 1   # hold final pose until next re-plan

            # ── Apply joint targets ───────────────────────────────────
            ctrl_target = [0.0, current_q[0], current_q[1], 0.0, current_q[2]]
            for name, val in zip(ACTUATOR_NAMES, ctrl_target):
                data.ctrl[act_id[name]] = val

            mujoco.mj_step(model, data)

            # ── Draw planned trajectory ───────────────────────────────
            # Visualises the full trajectory each frame:
            #   GREY  = already executed  |  GREEN = still to go
            #   WHITE sphere = current commanded pos
            #   RED   sphere = final target
            draw_trajectory_line(viewer, plan, frame)

            viewer.sync()

            # Status readout
            ee         = data.xpos[ee_id]
            tgt        = target_thread.get_target()
            q_deg      = np.degrees(current_q)
            phi_now    = plan.angle_traj[min(frame, plan.N - 1)]
            print(
                f"\r[Sim] t={data.time:.3f}s  "
                f"frame={frame:05d}/{plan.N-1}  "
                f"EE=({ee[0]:.4f},{ee[2]:.4f})  "
                f"tgt=({tgt[0]:.4f},{tgt[1]:.4f})  "
                f"phi={np.degrees(phi_now):.1f}deg  "
                f"q=[{q_deg[0]:6.2f} {q_deg[1]:6.2f} {q_deg[2]:6.2f}]deg  "
                f"replan_in={replan_interval-(now-last_replan_wall):.1f}s",
                end="", flush=True,
            )

            # Real-time throttle
            elapsed_wall = time.perf_counter() - wall_start
            sleep_t = (frame * plan.dt) - elapsed_wall
            if sleep_t > 0:
                time.sleep(sleep_t)
    print("\n[execute_motion] Viewer closed.")

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    qpos_adr, act_id = build_maps(model)
    ee_id            = find_ee(model)

    mujoco.mj_forward(model, data)
    current_pos = data.xpos[ee_id][[0, 2]].copy()
    print(f"[Main] Initial EE position: {current_pos}")

    initial_target = np.array([0.42482, 0.42089])
    target_thread  = MovingTargetThread(initial_target, update_interval=0.5)
    target_thread.start()
    print(f"[Main] Target thread started. Initial target: {initial_target}")
    try:
        execute_motion(model = model, data = data, qpos_adr = qpos_adr,
            act_id = act_id, ee_id = ee_id, target_thread = target_thread,
            target_angle_deg = 135.0, replan_interval = REPLAN_INTERVAL
        )
    finally:
        target_thread.stop()
        print("[Main] Target thread stopped.")

if __name__ == "__main__":
    main()