import mujoco
import mujoco.viewer
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional

XML_PATH = "Custom_arm.xml"
ARM_IDS  = ["a", "b"]

L1 = 0.2875
L2 = 0.2040
L3 = 0.1860

# Starting phi
PHI_A_START_DEG  = -180
PHI_B_START_DEG  = 0
# Target phi
PHI_A_TARGET_DEG = 0
PHI_B_TARGET_DEG = -180

REPLAN_INTERVAL = 10

# Trajectory parameters
FREQUENCY       = 500
V_CRUISE        = 0.15
ALPHA_CURVE     = 0.10
DAMP_RADIUS     = 0.05
T_MIN           = 0.1
T_MAX           = 4.0
BLEND_FRAMES    = int(0.05 * FREQUENCY)

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
    dist = float(np.linalg.norm(target_pos - current_pos))
    print(f"[Plan] {N} pts | dist={dist:.3f} m | T={total_time:.2f} s | "
          f"{np.round(current_pos, 4).tolist()} → {np.round(target_pos, 4).tolist()} | "
          f"phi {np.degrees(start_phi_rad):.1f}° → {np.degrees(target_phi_rad):.1f}°")
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
def get_target(model, data, arm_index):
    """Return the world-XY corner of the target box for the given arm."""
    bid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Target")
    gid  = next(i for i in range(model.ngeom) if model.geom_bodyid[i] == bid)
    sign = 1 if arm_index == 0 else -1
    sx, sy, sz = model.geom_size[gid][:3]
    local_off  = np.array([sign * sx, sign * sy, sign * sz])
    pos        = data.geom_xpos[gid].copy()
    rot        = data.geom_xmat[gid].reshape(3, 3)
    corner     = pos + rot @ local_off
    return corner[:2]

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

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    maps  = {arm_id: build_maps(model, arm_id) for arm_id in ARM_IDS}

    ee_site  = {arm_id: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                           f"EE_site_{arm_id}")
                for arm_id in ARM_IDS}
    sh_joint = {arm_id: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                           f"joint_hindarm_{arm_id}")
                for arm_id in ARM_IDS}

    mujoco.mj_forward(model, data)

    phi_a_target_rad = np.deg2rad(PHI_A_TARGET_DEG)
    phi_b_target_rad = np.deg2rad(PHI_B_TARGET_DEG)
    current_phi_a = np.deg2rad(PHI_A_START_DEG)
    current_phi_b = np.deg2rad(PHI_B_START_DEG)
    ee_a_world = data.site_xpos[ee_site["a"]][:2].copy()
    ee_b_world = data.site_xpos[ee_site["b"]][:2].copy()
    target_a   = get_target(model, data, arm_index=0)
    target_b   = get_target(model, data, arm_index=1)

    plan_a = plan_trajectory(ee_a_world, target_a, current_phi_a, phi_a_target_rad)
    plan_b = plan_trajectory(ee_b_world, target_b, current_phi_b, phi_b_target_rad)
    frame_a         = 0
    frame_b         = 0
    sim_time_plan_a = data.time
    sim_time_plan_b = data.time
    last_replan_a   = data.time
    last_replan_b   = data.time

    t0     = time.perf_counter()
    sim_t0 = data.time

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            sim_now = data.time
            qpos_adr_a, act_id_a = maps["a"]
            qpos_adr_b, act_id_b = maps["b"]
            origin_a = data.xanchor[sh_joint["a"]][:2].copy()
            origin_b = data.xanchor[sh_joint["b"]][:2].copy()
            traj_done_a = frame_a >= plan_a.N - 1
            traj_done_b = frame_b >= plan_b.N - 1

            if traj_done_a or sim_now - last_replan_a >= REPLAN_INTERVAL:
                new_target_a = get_target(model, data, arm_index=0)
                ee_now_a     = data.site_xpos[ee_site["a"]][:2].copy()
                plan_a = plan_trajectory(ee_now_a, new_target_a,
                                         current_phi_a, phi_a_target_rad)
                frame_a         = 0
                sim_time_plan_a = sim_now
                last_replan_a   = sim_now
                print(f"[Replan-A] → {np.round(new_target_a, 4).tolist()} | "
                      f"phi {np.degrees(current_phi_a):.1f}° → "
                      f"{np.degrees(phi_a_target_rad):.1f}°")

            if traj_done_b or sim_now - last_replan_b >= REPLAN_INTERVAL:
                new_target_b = get_target(model, data, arm_index=1)
                ee_now_b     = data.site_xpos[ee_site["b"]][:2].copy()
                plan_b = plan_trajectory(ee_now_b, new_target_b,
                                         current_phi_b, phi_b_target_rad)
                frame_b         = 0
                sim_time_plan_b = sim_now
                last_replan_b   = sim_now
                print(f"[Replan-B] → {np.round(new_target_b, 4).tolist()} | "
                      f"phi {np.degrees(current_phi_b):.1f}° → "
                      f"{np.degrees(phi_b_target_rad):.1f}°")

            frame_a = min(int((sim_now - sim_time_plan_a) / plan_a.dt), plan_a.N - 1)
            frame_b = min(int((sim_now - sim_time_plan_b) / plan_b.dt), plan_b.N - 1)

            current_phi_a = float(plan_a.angle_traj[frame_a])
            current_phi_b = float(plan_b.angle_traj[frame_b])

            # IK and control
            wp_a      = plan_a.cartesian_traj[frame_a]
            phi_a_now = plan_a.angle_traj[frame_a]
            q_a       = solve_ik_a(wp_a, phi_a_now, origin_a, elbow_up=False)

            wp_b      = plan_b.cartesian_traj[frame_b]
            phi_b_now = plan_b.angle_traj[frame_b]
            q_b       = solve_ik_b(wp_b, phi_b_now, origin_b, elbow_up=True)

            for name, val in zip(["hindarm_ctrl_a", "forearm_ctrl_a", "hand_ctrl_a"], q_a):
                data.ctrl[act_id_a[name]] = val
            for name, val in zip(["hindarm_ctrl_b", "forearm_ctrl_b", "hand_ctrl_b"], q_b):
                data.ctrl[act_id_b[name]] = val

            mujoco.mj_step(model, data)

            z_a = float(data.xanchor[sh_joint["a"]][2])
            z_b = float(data.xanchor[sh_joint["b"]][2])
            draw_trajectories(viewer,
                              plan_a, frame_a,
                              plan_b, frame_b,
                              z_height_a=z_a,
                              z_height_b=z_b)
            viewer.sync()
            elapsed_sim  = data.time - sim_t0
            elapsed_wall = time.perf_counter() - t0
            sleep_t = elapsed_sim - elapsed_wall
            if sleep_t > 0:
                time.sleep(sleep_t)

if __name__ == "__main__":
    main()