import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
from Plotting import *

XML_PATH = "Custom_arm.xml"
ARM_IDS  = ["a", "b"]

L1 = 0.2875
L2 = 0.2040
L3 = 0.1860
omega_given = 1

# Starting phi
PHI_A_START_DEG  = -180
PHI_B_START_DEG  = 0
# Target phi
PHI_A_TARGET_DEG = 0
PHI_B_TARGET_DEG = -180

MIM = 0.10                          # apparent mass
DIM = 1.50                          # apparent damping
KIM = 10.0                          # apparent stiffness
IMPEDANCE_MAX_VEL        = 0.25     # m/s, safety clamp on impedance velocity command
IMPEDANCE_MAX_DEFLECTION = 0.03

EQ_TRACK_VEL      = 0.20            # m/s
CONTACT_FORCE_EPS = 0.05

FIRST_CONTACT_FORCE_EPS   = 0.15
DETUMBLE_OMEGA_EPS        = 0.05
MIN_CONTACTS_BEFORE_CAGE  = 4

KIM_CAPTURE               = 60.0
KIM_RAMP_RATE             = 40.0
CAPTURE_HOLD_TIME         = 1.0
CAGE_TRACK_VEL             = 0.35   # m/s, faster equilibrium-closing speed once caging
CAGE_CONTACT_FORCE_EPS     = 1.0    # N, keep closing until a firm grip force is felt
TARGET_LINVEL_EPS   = 0.03          # m/s, target COM speed below this = "translationally settled"
TRACK_VEL_MARGIN     = 0.15         # m/s, tracking speed = target speed + this margin
MAX_TRACK_VEL         = 0.70        # m/s, safety cap on the dynamic tracking speed
TRACK_LEAD_TIME       = 0.05        # s, how far ahead to aim the pursuit point
CAGE_ABORT_LINVEL     = 0.15        # m/s

CAGE_ENTER_HOLD        = 0.10       # s, settle condition must hold this long before entering caging
CAGE_ABORT_HOLD        = 0.15       # s, escape condition must hold this long before aborting caging
MAX_JOINT_VEL          = 2.0        # rad/s, symmetric per-arm joint speed cap (impedance mode only)

NEAR_TARGET_DIST      = 0.05        # m, proximity threshold that triggers the slowdown
NEAR_TARGET_VEL       = 0.15        # m/s, reduced closing speed once inside NEAR_TARGET_DIST
FREEZE_AFTER_CONTACT_COUNT = 6

# Trajectory parameters
FREQUENCY       = 500
V_CRUISE        = 0.15
ALPHA_CURVE     = 0.10
DAMP_RADIUS     = 0.05
T_MIN           = 0.1
T_MAX           = 4.0
BLEND_FRAMES    = int(0.05 * FREQUENCY)

HORIZONTAL_TOL_RAD = 0.05
_VIS_MAX_SEGS = 200
_EYE = np.eye(3, dtype=np.float64).flatten()

_COLOURS = {
    "a": {
        "done"   : np.array([0.55, 0.55, 0.55, 0.50], dtype=np.float32),
        "pending": np.array([0.20, 0.90, 0.30, 0.90], dtype=np.float32),
        "current": np.array([1.00, 1.00, 1.00, 1.00], dtype=np.float32),
        "target" : np.array([0.95, 0.20, 0.20, 1.00], dtype=np.float32),
    },
    "b": {
        "done"   : np.array([0.40, 0.40, 0.60, 0.50], dtype=np.float32),
        "pending": np.array([0.20, 0.85, 0.95, 0.90], dtype=np.float32),
        "current": np.array([1.00, 1.00, 0.00, 1.00], dtype=np.float32),
        "target" : np.array([1.00, 0.50, 0.00, 1.00], dtype=np.float32),
    },
}

@dataclass
class TrajectoryPlan:
    """Holds a fully pre-computed quintic trajectory in world XY."""
    cartesian_traj   : np.ndarray
    angle_traj       : np.ndarray
    dt               : float
    N                : int
    current_pos      : np.ndarray
    target_pos       : np.ndarray
    target_angle_rad : float
    total_time       : float

@dataclass
class ImpedanceState:
    """Per-arm impedance state used once the arm is in the target region."""
    delta_pos     : np.ndarray = None
    vel           : np.ndarray = None
    eq_pos        : np.ndarray = None
    t_contact     : float = 0.0
    was_in_contact: bool  = False
    kim_current   : float = KIM

    def __post_init__(self):
        if self.delta_pos is None:
            self.delta_pos = np.zeros(2)
        if self.vel is None:
            self.vel = np.zeros(2)
        if self.eq_pos is None:
            self.eq_pos = np.zeros(2)

def impedance_update(imp: ImpedanceState, force_xy: np.ndarray, dt: float,
                      kim: float = KIM) -> np.ndarray:
    """Mass-spring-damper impedance law (Eq. 3): given a measured contact
    force, advances the internal impedance state (imp.vel, imp.delta_pos)
    in place and returns the (clamped) compliant velocity.
    """
    accel = (force_xy - DIM * imp.vel - kim * imp.delta_pos) / MIM
    imp.vel += accel * dt
    vmag = np.linalg.norm(imp.vel)
    if vmag > IMPEDANCE_MAX_VEL:
        imp.vel *= IMPEDANCE_MAX_VEL / vmag

    imp.delta_pos += imp.vel * dt
    dmag = np.linalg.norm(imp.delta_pos)
    if dmag > IMPEDANCE_MAX_DEFLECTION:
        imp.delta_pos *= IMPEDANCE_MAX_DEFLECTION / dmag
        imp.vel[:] = 0.0

    return imp.vel.copy()

# QUINTIC TRAJECTORY
def _quintic_coeffs(p0, pf, v0, T):
    T3, T4, T5 = T**3, T**4, T**5
    dp = pf - p0
    a0 = p0
    a1 = v0
    a2 = np.zeros_like(np.atleast_1d(np.asarray(dp, dtype=float)))
    if np.ndim(dp) == 0:
        a2 = 0.0
    a3 = (10 * dp - T * (6 * v0)) / T3
    a4 = (-15 * dp + T * (8 * v0)) / T4
    a5 = (6  * dp - T * (3 * v0)) / T5
    return a0, a1, a2, a3, a4, a5

def _quintic_eval(coeffs, t):
    a0, a1, a2, a3, a4, a5 = coeffs
    return a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5

def _quintic_blend(start_pos, target_pos, start_phi, target_phi,
                   start_vel, start_phi_vel, total_time, frequency):
    if start_vel is None:
        start_vel = np.zeros(2)
    start_vel = np.asarray(start_vel, dtype=float)
    T         = float(total_time)
    ts        = np.linspace(0, T, int(T * frequency))
    pos_coeffs = _quintic_coeffs(start_pos, target_pos, start_vel, T)
    traj       = np.array([_quintic_eval(pos_coeffs, ti) for ti in ts])
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

def plan_trajectory(current_pos, target_pos,
                    start_phi_rad, target_phi_rad,
                    start_vel=None, start_phi_vel=0.0,
                    frequency=FREQUENCY) -> TrajectoryPlan:
    current_pos = np.asarray(current_pos, dtype=float)
    target_pos  = np.asarray(target_pos,  dtype=float)
    if start_vel is None:
        start_vel = np.zeros(2)
    start_vel  = np.asarray(start_vel, dtype=float)
    start_vel  = damp_start_velocity(current_pos, target_pos, start_vel)
    total_time = compute_total_time(current_pos, target_pos, start_vel)

    cartesian_traj, angle_traj = _quintic_blend(
        current_pos, target_pos,
        start_phi_rad, target_phi_rad,
        start_vel, start_phi_vel,
        total_time, frequency,
    )
    dt   = 1.0 / frequency
    N    = len(cartesian_traj)
    return TrajectoryPlan(
        cartesian_traj=cartesian_traj,
        angle_traj=angle_traj,
        dt=dt,
        N=N,
        current_pos=current_pos,
        target_pos=target_pos,
        target_angle_rad=target_phi_rad,
        total_time=total_time,
    )

def build_maps(model, arm_id):
    joint_names    = [f"joint_hindarm_{arm_id}",
                      f"joint_forearm_{arm_id}",
                      f"joint_hand_{arm_id}"]
    actuator_names = [f"hindarm_ctrl_{arm_id}",
                      f"forearm_ctrl_{arm_id}",
                      f"hand_ctrl_{arm_id}"]
    qpos_adr = {
        j: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
        for j in joint_names
    }
    act_id = {
        a: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        for a in actuator_names
    }
    return qpos_adr, act_id

# IK SOLVERS
def solve_ik_a(target_xz, phi_world, fk_origin, elbow_up):
    new_target         = target_xz - fk_origin
    target_x, target_z = new_target

    x_wrist = target_x - L3 * np.cos(phi_world)
    z_wrist = target_z - L3 * np.sin(phi_world)

    cos_q2 = (x_wrist**2 + z_wrist**2 - L1**2 - L2**2) / (2.0 * L1 * L2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    q2     = -np.arccos(cos_q2) if elbow_up else np.arccos(cos_q2)

    alpha    = np.arctan2(z_wrist, x_wrist)
    beta     = np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    q1_world = alpha - beta
    q1       = q1_world - np.pi

    q3 = phi_world - q1 - q2 - np.pi
    q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
    q3 = (q3 + np.pi) % (2 * np.pi) - np.pi
    return np.array([q1, q2, q3])

def solve_ik_b(target_xz, phi_world, fk_origin, elbow_up):
    new_target         = target_xz - fk_origin
    target_x, target_z = new_target

    x_wrist = target_x - L3 * np.cos(phi_world)
    z_wrist = target_z - L3 * np.sin(phi_world)

    cos_q2 = (x_wrist**2 + z_wrist**2 - L1**2 - L2**2) / (2.0 * L1 * L2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    q2     = -np.arccos(cos_q2) if elbow_up else np.arccos(cos_q2)

    alpha = np.arctan2(z_wrist, x_wrist)
    beta  = np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    q1    = alpha - beta
    q3    = phi_world - q1 - q2
    return np.array([q1, q2, q3])

# TARGET HELPER
def is_target_horizontal(model: mujoco.MjModel, data: mujoco.MjData,
                          tol: float = HORIZONTAL_TOL_RAD):
    jid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_rz")
    angle = float(data.qpos[model.jnt_qposadr[jid]])
    a     = angle % np.pi
    dist = min(a, np.pi - a)
    return dist < tol

def get_target(model, data, arm_index):
    if not is_target_horizontal(model, data):
        return None
    bid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Target")
    gid  = next(i for i in range(model.ngeom) if model.geom_bodyid[i] == bid)
    sign = 1 if arm_index == 0 else -1
    sx, sy, sz = model.geom_size[gid][:3]
    local_off  = np.array([sign * sx, sign * sy, sign * sz])
    pos        = data.geom_xpos[gid].copy()
    rot        = data.geom_xmat[gid].reshape(3, 3)
    corner     = pos + rot @ local_off
    return corner[:2]

def get_target_corner_live(model, data, arm_index, gid=None):
    if gid is None:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Target")
        gid = next(i for i in range(model.ngeom) if model.geom_bodyid[i] == bid)
    sign = 1 if arm_index == 0 else -1
    sx, sy, sz = model.geom_size[gid][:3]
    local_off  = np.array([sign * sx, sign * sy, sign * sz])
    pos        = data.geom_xpos[gid].copy()
    rot        = data.geom_xmat[gid].reshape(3, 3)
    corner     = pos + rot @ local_off
    return corner[:2]

def set_target_angular_velocity(model: mujoco.MjModel, data: mujoco.MjData, omega: float):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_rz")
    dof = model.jnt_dofadr[jid]
    data.qvel[dof] = omega

def get_target_angular_velocity(model: mujoco.MjModel, data: mujoco.MjData, dof=None):
    if dof is None:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_rz")
        dof = model.jnt_dofadr[jid]
    return float(data.qvel[dof])

def get_target_planar_velocity(model: mujoco.MjModel, data: mujoco.MjData, bid: int):
    res = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, res, 0)
    omega_z    = float(res[2])
    lin_vel_xy = np.array([res[3], res[4]])
    return omega_z, lin_vel_xy

# TRAJECTORY VISUALISATION
def _add_segment(scn, p1_xy, p2_xy, rgba, z_height=0.0, width_px=2.0):
    if scn.ngeom >= scn.maxgeom - 4:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3, np.float64), np.zeros(3, np.float64), _EYE, rgba,
    )
    mujoco.mjv_connector(
        g, mujoco.mjtGeom.mjGEOM_LINE, width_px,
        np.array([p1_xy[0], p1_xy[1], z_height], np.float64),
        np.array([p2_xy[0], p2_xy[1], z_height], np.float64),
    )
    scn.ngeom += 1

def _add_sphere(scn, pos_xy, rgba, z_height=0.0, size=0.008):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([size] * 3, np.float64),
        np.array([pos_xy[0], pos_xy[1], z_height], np.float64),
        _EYE, rgba,
    )
    scn.ngeom += 1

def draw_trajectories(viewer,
                      plan_a: Optional[TrajectoryPlan], frame_a: int,
                      plan_b: Optional[TrajectoryPlan], frame_b: int,
                      z_height_a: float = 0.0,
                      z_height_b: float = 0.0):
    if not hasattr(viewer, "user_scn"):
        return
    scn       = viewer.user_scn
    scn.ngeom = 0

    for plan, frame, arm_id, z_h in (
        (plan_a, frame_a, "a", z_height_a),
        (plan_b, frame_b, "b", z_height_b),
    ):
        if plan is None:
            continue

        col  = _COLOURS[arm_id]
        traj = plan.cartesian_traj
        N    = plan.N

        n_exec     = max(frame - 1, 0)
        n_rem      = max(N - 1 - frame, 0)
        total_segs = n_exec + n_rem

        if total_segs == 0:
            _add_sphere(scn, traj[0], col["current"], z_height=z_h, size=0.010)
            continue

        ratio   = min(1.0, _VIS_MAX_SEGS / total_segs)
        step_ex = max(1, int(1.0 / ratio))
        step_rm = max(1, int(1.0 / ratio))

        for i in range(0, min(frame, N - 1), step_ex):
            _add_segment(scn, traj[i], traj[i + 1],
                         col["done"], z_height=z_h, width_px=1.5)
        for i in range(max(frame, 0), N - 1, step_rm):
            _add_segment(scn, traj[i], traj[i + 1],
                         col["pending"], z_height=z_h, width_px=3.0)

        _add_sphere(scn, traj[min(frame, N - 1)], col["current"],
                    z_height=z_h, size=0.010)
        _add_sphere(scn, traj[-1], col["target"],
                    z_height=z_h, size=0.012)

def compute_reaction_forces(model: mujoco.MjModel,
                             data:  mujoco.MjData,
                             ee_site_ids: dict,
                             target_bid: int = None,
                             ee_body: dict = None) -> dict:
    if target_bid is None:
        target_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Target")
    if ee_body is None:
        ee_body = {arm: int(model.site_bodyid[sid])
                   for arm, sid in ee_site_ids.items()}
 
    result = {arm: {"force": np.zeros(3), "torque": np.zeros(3)}
              for arm in ee_site_ids}
    wrench = np.zeros(6)   # [Fx, Fy, Fz, 0, 0, 0] in contact frame
 
    for i in range(data.ncon):
        c  = data.contact[i]
        b1 = int(model.geom_bodyid[c.geom1])
        b2 = int(model.geom_bodyid[c.geom2])
 
        mujoco.mj_contactForce(model, data, i, wrench)
        rot     = c.frame.reshape(3, 3)
        f_world = rot.T @ wrench[:3]
 
        for arm, ee_bid in ee_body.items():
            if b1 == ee_bid and b2 == target_bid:
                sign = +1.0
            elif b1 == target_bid and b2 == ee_bid:
                sign = -1.0
            else:
                continue
 
            f_ee = sign * f_world
            r        = c.pos - data.xipos[ee_bid]
            t_ee     = np.cross(r, f_ee)
 
            result[arm]["force"]  += f_ee
            result[arm]["torque"] += t_ee
 
    return result

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    set_target_angular_velocity(model, data, omega_given)
    maps  = {arm_id: build_maps(model, arm_id) for arm_id in ARM_IDS}
    impedance_dt = float(model.opt.timestep)

    ee_site  = {arm_id: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                           f"EE_site_{arm_id}")
                for arm_id in ARM_IDS}
    sh_joint = {arm_id: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                           f"joint_hindarm_{arm_id}")
                for arm_id in ARM_IDS}
    _target_bid    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Target")
    _target_gid    = next(i for i in range(model.ngeom) if model.geom_bodyid[i] == _target_bid)
    _target_rz_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_rz")
    _target_rz_dof = model.jnt_dofadr[_target_rz_jid]
    _ee_body       = {arm: int(model.site_bodyid[sid]) for arm, sid in ee_site.items()}

    mujoco.mj_forward(model, data)
    phi_a_target_rad = np.deg2rad(PHI_A_TARGET_DEG)
    phi_b_target_rad = np.deg2rad(PHI_B_TARGET_DEG)
    current_phi_a = np.deg2rad(PHI_A_START_DEG)
    current_phi_b = np.deg2rad(PHI_B_START_DEG)
    ee_a_world = data.site_xpos[ee_site["a"]][:2].copy()
    ee_b_world = data.site_xpos[ee_site["b"]][:2].copy()
    
    target_a   = get_target(model, data, arm_index=0) - [0.2,0]
    target_b   = get_target(model, data, arm_index=1) + [0.2,0]

    plan_a = plan_trajectory(ee_a_world, target_a, current_phi_a, phi_a_target_rad)
    dist = float(np.linalg.norm(target_a - ee_a_world))
    print(f"[Plan-A] {plan_a.N} pts | dist={dist:.3f} m | T={plan_a.total_time:.2f} s | "
          f"{np.round(ee_a_world, 4).tolist()} → {np.round(target_a, 4).tolist()} | "
          f"phi {np.degrees(current_phi_a):.1f}° → {np.degrees(phi_a_target_rad):.1f}°")
    plan_b = plan_trajectory(ee_b_world, target_b, current_phi_b, phi_b_target_rad)
    dist = float(np.linalg.norm(target_b - ee_b_world))
    print(f"[Plan-B] {plan_a.N} pts | dist={dist:.3f} m | T={plan_a.total_time:.2f} s | "
          f"{np.round(ee_b_world, 4).tolist()} → {np.round(target_b, 4).tolist()} | "
          f"phi {np.degrees(current_phi_b):.1f}° → {np.degrees(phi_b_target_rad):.1f}°")
    frame_a         = 0
    frame_b         = 0
    sim_time_plan_a = data.time
    sim_time_plan_b = data.time

    t0     = time.perf_counter()
    sim_t0 = data.time

    impedance_mode = False
    imp_st = {"a": ImpedanceState(), "b": ImpedanceState()}
    active_arm    = "a"
    caging_mode   = False
    contact_count = 0
    caging_start_time = 0.0
    captured          = False
    frozen_wp_a       = None
    frozen_wp_b       = None
    frozen_phi_a      = None
    frozen_phi_b      = None
    settle_since      = None
    escape_since      = None
    near_target_flag  = {"a": False, "b": False}
    frozen_after_contacts = False
    frozen_q_a            = None
    frozen_q_b            = None
    q_a_prev = None
    q_b_prev = None
    all_angle_armA = []
    all_angle_armB = []
    rxn_forces_a   = []
    rxn_forces_b   = []
    rxn_torques_a  = []
    rxn_torques_b  = []
    with mujoco.viewer.launch_passive(model, data) as viewer:
        try:
            while viewer.is_running():
                sim_now = data.time
                qpos_adr_a, act_id_a = maps["a"]
                qpos_adr_b, act_id_b = maps["b"]
                origin_a = data.xanchor[sh_joint["a"]][:2].copy()
                origin_b = data.xanchor[sh_joint["b"]][:2].copy()
                traj_done_a = frame_a >= plan_a.N - 1
                traj_done_b = frame_b >= plan_b.N - 1
                if not impedance_mode and traj_done_a and traj_done_b:
                    impedance_mode = True
                    imp_st["a"].eq_pos    = data.site_xpos[ee_site["a"]][:2].copy()
                    imp_st["b"].eq_pos    = data.site_xpos[ee_site["b"]][:2].copy()
                    imp_st["a"].delta_pos = np.zeros(2)
                    imp_st["b"].delta_pos = np.zeros(2)
                    imp_st["a"].vel       = np.zeros(2)
                    imp_st["b"].vel       = np.zeros(2)
                    imp_st["a"].t_contact = sim_now
                    imp_st["b"].t_contact = sim_now
                    imp_st["a"].was_in_contact = False
                    imp_st["b"].was_in_contact = False
                    imp_st["a"].kim_current    = KIM
                    imp_st["b"].kim_current    = KIM
                    active_arm    = "a"
                    caging_mode   = False
                    contact_count = 0
                    caging_start_time = 0.0
                    captured          = False
                    frozen_wp_a       = None
                    frozen_wp_b       = None
                    frozen_phi_a      = None
                    frozen_phi_b      = None
                    settle_since      = None
                    escape_since      = None
                    near_target_flag["a"] = False
                    near_target_flag["b"] = False
                    frozen_after_contacts = False
                    frozen_q_a            = None
                    frozen_q_b            = None
                    print(f"[Impedance] activated at t={sim_now:.2f}s | "
                          f"both arms reached target region → compliant contact mode")

                if not impedance_mode:
                    frame_a = min(max(int((sim_now - sim_time_plan_a) / plan_a.dt), 0), plan_a.N - 1)
                    frame_b = min(max(int((sim_now - sim_time_plan_b) / plan_b.dt), 0), plan_b.N - 1)
                    current_phi_a = float(plan_a.angle_traj[frame_a])
                    current_phi_b = float(plan_b.angle_traj[frame_b])

                # IK and control
                if impedance_mode:
                    omega_target, target_lin_vel = get_target_planar_velocity(model, data, _target_bid)
                    lin_speed    = float(np.linalg.norm(target_lin_vel))
                    settled_now = (abs(omega_target) < DETUMBLE_OMEGA_EPS) and (lin_speed < TARGET_LINVEL_EPS)
                    f_a = rxn_forces_a[-1][:2] if rxn_forces_a else np.zeros(2)
                    f_b = rxn_forces_b[-1][:2] if rxn_forces_b else np.zeros(2)
                    a_now_contact = np.linalg.norm(f_a) > FIRST_CONTACT_FORCE_EPS
                    b_now_contact = np.linalg.norm(f_b) > FIRST_CONTACT_FORCE_EPS

                    if not caging_mode:
                        if active_arm == "a" and a_now_contact and not imp_st["a"].was_in_contact:
                            contact_count += 1
                            active_arm = "b"
                            near_target_flag["a"] = False
                            print(f"[Impedance] contact #{contact_count} (arm a) at t={sim_now:.2f}s "
                                  f"→ hand-off to arm b")
                            print("Current Omega:", omega_target, "| Current linear speed:", lin_speed)
                        elif active_arm == "b" and b_now_contact and not imp_st["b"].was_in_contact:
                            contact_count += 1
                            active_arm = "a"
                            near_target_flag["b"] = False
                            print(f"[Impedance] contact #{contact_count} (arm b) at t={sim_now:.2f}s "
                                  f"→ hand-off to arm a")
                            print("Current Omega:", omega_target, "| Current linear speed:", lin_speed)
                    imp_st["a"].was_in_contact = a_now_contact
                    imp_st["b"].was_in_contact = b_now_contact

                    if not frozen_after_contacts and contact_count > FREEZE_AFTER_CONTACT_COUNT:
                        frozen_after_contacts = True
                        frozen_q_a = q_a_prev.copy() if q_a_prev is not None else None
                        frozen_q_b = q_b_prev.copy() if q_b_prev is not None else None
                        print(f"[Impedance] contact_count={contact_count} > {FREEZE_AFTER_CONTACT_COUNT} "
                              f"at t={sim_now:.2f}s → freezing both arms at current joint angles")

                    if not frozen_after_contacts and contact_count >= MIN_CONTACTS_BEFORE_CAGE:
                        if settled_now:
                            if settle_since is None:
                                settle_since = sim_now
                            elif (sim_now - settle_since) >= CAGE_ENTER_HOLD:
                                caging_mode   = True
                                caging_start_time = sim_now
                                settle_since  = None
                                print(f"[Impedance] target settled (|omega|={omega_target:.3f} rad/s, "
                                      f"|v|={lin_speed:.3f} m/s) after {contact_count} contacts "
                                      f"→ caging mode, both arms close in")
                        else:
                            settle_since = None
                    else:
                        settle_since = None

                    escaped_now = lin_speed > CAGE_ABORT_LINVEL
                    if caging_mode and not captured and not frozen_after_contacts:
                        if escaped_now:
                            if escape_since is None:
                                escape_since = sim_now
                            elif (sim_now - escape_since) >= CAGE_ABORT_HOLD:
                                caging_mode   = False
                                contact_count = 0
                                escape_since  = None
                                print(f"[Impedance] target escaped caging (|v|={lin_speed:.3f} m/s) "
                                      f"at t={sim_now:.2f}s → reverting to sequential detumbling "
                                      f"(contact count reset)")
                        else:
                            escape_since = None
                    else:
                        escape_since = None

                    if caging_mode and not captured and not frozen_after_contacts and \
                            (sim_now - caging_start_time) >= CAPTURE_HOLD_TIME:
                        captured     = True
                        frozen_wp_a  = (imp_st["a"].eq_pos + imp_st["a"].delta_pos).copy()
                        frozen_wp_b  = (imp_st["b"].eq_pos + imp_st["b"].delta_pos).copy()
                        frozen_phi_a = current_phi_a
                        frozen_phi_b = current_phi_b
                        print(f"[Impedance] capture confirmed at t={sim_now:.2f}s "
                              f"→ arms locked, holding position")

                    if frozen_after_contacts:
                        wp_a      = imp_st["a"].eq_pos + imp_st["a"].delta_pos
                        phi_a_now = current_phi_a
                        wp_b      = imp_st["b"].eq_pos + imp_st["b"].delta_pos
                        phi_b_now = current_phi_b
                    elif captured:
                        wp_a      = frozen_wp_a
                        phi_a_now = frozen_phi_a
                        wp_b      = frozen_wp_b
                        phi_b_now = frozen_phi_b
                    else:
                        track_a = caging_mode or (active_arm == "a")
                        track_b = caging_mode or (active_arm == "b")
                        base_track_vel_a = CAGE_TRACK_VEL if caging_mode else EQ_TRACK_VEL
                        base_track_vel_b = CAGE_TRACK_VEL if caging_mode else EQ_TRACK_VEL
                        force_eps_a      = CAGE_CONTACT_FORCE_EPS if caging_mode else CONTACT_FORCE_EPS
                        force_eps_b      = CAGE_CONTACT_FORCE_EPS if caging_mode else CONTACT_FORCE_EPS

                        track_vel_a = float(np.clip(lin_speed + TRACK_VEL_MARGIN,
                                                     base_track_vel_a, MAX_TRACK_VEL))
                        track_vel_b = float(np.clip(lin_speed + TRACK_VEL_MARGIN,
                                                     base_track_vel_b, MAX_TRACK_VEL))
                        lead_offset = target_lin_vel * TRACK_LEAD_TIME

                        if track_a:
                            corner_a = get_target_corner_live(model, data, arm_index=0, gid=_target_gid)
                            corner_a = corner_a + lead_offset
                            ee_a_now = data.site_xpos[ee_site["a"]][:2]
                            dist_to_target_a = float(np.linalg.norm(corner_a - ee_a_now))
                            if dist_to_target_a < NEAR_TARGET_DIST:
                                track_vel_a = NEAR_TARGET_VEL
                                if not near_target_flag["a"]:
                                    near_target_flag["a"] = True
                                    print(f"[Impedance] arm a within {NEAR_TARGET_DIST:.3f} m of target "
                                          f"(d={dist_to_target_a:.3f} m) at t={sim_now:.2f}s "
                                          f"→ closing speed reduced to {NEAR_TARGET_VEL:.3f} m/s")
                            else:
                                near_target_flag["a"] = False

                        if track_b:
                            corner_b = get_target_corner_live(model, data, arm_index=1, gid=_target_gid)
                            corner_b = corner_b + lead_offset
                            ee_b_now = data.site_xpos[ee_site["b"]][:2]
                            dist_to_target_b = float(np.linalg.norm(corner_b - ee_b_now))
                            if dist_to_target_b < NEAR_TARGET_DIST:
                                track_vel_b = NEAR_TARGET_VEL
                                if not near_target_flag["b"]:
                                    near_target_flag["b"] = True
                                    print(f"[Impedance] arm b within {NEAR_TARGET_DIST:.3f} m of target "
                                          f"(d={dist_to_target_b:.3f} m) at t={sim_now:.2f}s "
                                          f"→ closing speed reduced to {NEAR_TARGET_VEL:.3f} m/s")
                            else:
                                near_target_flag["b"] = False

                        if track_a and np.linalg.norm(f_a) < force_eps_a:
                            dir_a    = corner_a - imp_st["a"].eq_pos
                            dist_a   = np.linalg.norm(dir_a)
                            if dist_a > 1e-4:
                                imp_st["a"].eq_pos = imp_st["a"].eq_pos + \
                                    dir_a / dist_a * min(track_vel_a * impedance_dt, dist_a)

                        if track_b and np.linalg.norm(f_b) < force_eps_b:
                            dir_b    = corner_b - imp_st["b"].eq_pos
                            dist_b   = np.linalg.norm(dir_b)
                            if dist_b > 1e-4:
                                imp_st["b"].eq_pos = imp_st["b"].eq_pos + \
                                    dir_b / dist_b * min(track_vel_b * impedance_dt, dist_b)

                        target_kim = KIM_CAPTURE if caging_mode else KIM
                        step       = KIM_RAMP_RATE * impedance_dt
                        for arm in ("a", "b"):
                            if imp_st[arm].kim_current < target_kim:
                                imp_st[arm].kim_current = min(target_kim, imp_st[arm].kim_current + step)
                            elif imp_st[arm].kim_current > target_kim:
                                imp_st[arm].kim_current = max(target_kim, imp_st[arm].kim_current - step)

                        impedance_update(imp_st["a"], f_a, impedance_dt, kim=imp_st["a"].kim_current)
                        impedance_update(imp_st["b"], f_b, impedance_dt, kim=imp_st["b"].kim_current)
                        wp_a      = imp_st["a"].eq_pos + imp_st["a"].delta_pos
                        phi_a_now = current_phi_a
                        wp_b      = imp_st["b"].eq_pos + imp_st["b"].delta_pos
                        phi_b_now = current_phi_b
                else:
                    wp_a      = plan_a.cartesian_traj[frame_a]
                    phi_a_now = plan_a.angle_traj[frame_a]
                    wp_b      = plan_b.cartesian_traj[frame_b]
                    phi_b_now = plan_b.angle_traj[frame_b]

                q_a       = solve_ik_a(wp_a, phi_a_now, origin_a, elbow_up=False)
                q_b       = solve_ik_b(wp_b, phi_b_now, origin_b, elbow_up=True)

                if frozen_after_contacts:
                    if frozen_q_a is not None:
                        q_a = frozen_q_a.copy()
                    if frozen_q_b is not None:
                        q_b = frozen_q_b.copy()

                if impedance_mode:
                    max_step = MAX_JOINT_VEL * impedance_dt
                    if q_a_prev is not None:
                        step_a = q_a - q_a_prev
                        smag_a = np.linalg.norm(step_a)
                        if smag_a > max_step:
                            q_a = q_a_prev + step_a / smag_a * max_step
                    if q_b_prev is not None:
                        step_b = q_b - q_b_prev
                        smag_b = np.linalg.norm(step_b)
                        if smag_b > max_step:
                            q_b = q_b_prev + step_b / smag_b * max_step
                q_a_prev = q_a.copy()
                q_b_prev = q_b.copy()

                all_angle_armA.append(q_a)
                all_angle_armB.append(q_b)

                for name, val in zip(["hindarm_ctrl_a", "forearm_ctrl_a", "hand_ctrl_a"], q_a):
                    data.ctrl[act_id_a[name]] = val
                for name, val in zip(["hindarm_ctrl_b", "forearm_ctrl_b", "hand_ctrl_b"], q_b):
                    data.ctrl[act_id_b[name]] = val

                mujoco.mj_step(model, data)

                # Record reaction forces
                rxn = compute_reaction_forces(model, data, ee_site,
                                               target_bid=_target_bid, ee_body=_ee_body)
                rxn_forces_a.append(rxn["a"]["force"].copy())
                rxn_forces_b.append(rxn["b"]["force"].copy())
                rxn_torques_a.append(rxn["a"]["torque"].copy())
                rxn_torques_b.append(rxn["b"]["torque"].copy())
                z_a = float(data.xanchor[sh_joint["a"]][2])
                z_b = float(data.xanchor[sh_joint["b"]][2])
                draw_trajectories(viewer, plan_a, frame_a, plan_b, frame_b,
                                z_height_a=z_a, z_height_b=z_b)
                viewer.sync()
                elapsed_sim  = data.time - sim_t0
                elapsed_wall = time.perf_counter() - t0
                sleep_t = elapsed_sim - elapsed_wall
                if sleep_t > 0:
                    time.sleep(sleep_t)
        finally:
            print("[Info] Simulation ended")
            viewer.close()
            plot_angles(all_angle_armA, all_angle_armB)
            plot_reaction_forces(rxn_forces_a, rxn_forces_b)
            plot_reaction_torques(rxn_torques_a, rxn_torques_b)

if __name__ == "__main__":
    main()