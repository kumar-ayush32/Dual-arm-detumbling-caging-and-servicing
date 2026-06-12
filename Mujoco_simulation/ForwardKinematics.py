import os
from pyexpat import model
import mujoco
import mujoco.viewer
import numpy as np
import time
from typing import Tuple, List
os.environ["MUJOCO_GL"] = "glfw"
XML_PATH = "main.xml"

"""
tmp.qpos        # Joint angles
tmp.qvel        # Joint velocities
tmp.ctrl        # Actuator commands
tmp.xpos        # Body positions
tmp.xmat        # Body orientations
tmp.xquat       # Quaternion orientation
tmp.qfrc_actuator   # Joint torques
tmp.xfrc_applied    # External forces
tmp.subtree_com     # Center of mass
tmp.time
"""
# Pose sequence: (hip °, hindarm °, forearm °, wrist °, hand °)
# Using only Hindarm, Forearm, Hand for single plane 3DOF motion
POSES = [(0, 60, 45, 0, 30)]
N_RAMP = 800
N_SETTLE = 200

JOINT_NAMES = [
    "joint_hip",
    "joint_hindarm",
    "joint_forearm",
    "joint_wrist",
    "joint_hand"
]
ACTUATOR_NAMES = [
    "hip_ctrl",
    "hindarm_ctrl",
    "forearm_ctrl",
    "wrist_ctrl",
    "hand_ctrl"
]
DOF3_LABELS = [
    "Hindarm    (θ₁) :",
    "Forearm (θ₂)    :",
    "Hand (θ₃)       :"
]
EE_CANDIDATES = [
    "gripper_Gripper",
    "hand_Hand",
    "wrist_Wrist",
    "forearm_Forearm"
]

def build_maps(model):
    qpos_adr = {
        n: model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)]
        for n in JOINT_NAMES
    }
    act_id = {n: mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,n)
        for n in ACTUATOR_NAMES
    }
    return qpos_adr, act_id

def find_end_effector(model):
    for name in EE_CANDIDATES:
        bid = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
        if bid >= 0:
            return bid, name
    bid = model.nbody - 1
    name = (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,bid) or f"body[{bid}]"
    )
    return bid, name

def _set_ctrl(data, act_id, values):
    for name, val in zip(ACTUATOR_NAMES, values):
        data.ctrl[act_id[name]] = val

def _smooth_step(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)

def forward_kinematics(model, hip_deg, hindarm_deg, forearm_deg, wrist_deg,hand_deg):
    """
    Compute FK on a temporary MjData.
    Does not modify the live simulation.
    """
    tmp = mujoco.MjData(model)
    qpos_adr, _ = build_maps(model)
    angles_rad = np.radians([hip_deg,hindarm_deg,forearm_deg,wrist_deg,hand_deg])

    for jname, angle in zip(JOINT_NAMES, angles_rad):
        tmp.qpos[qpos_adr[jname]] = angle
    mujoco.mj_kinematics(model, tmp)
    ee_id, ee_name = find_end_effector(model)

    return (tmp.xpos[ee_id].copy(),tmp.xmat[ee_id].reshape(3, 3).copy(),ee_name)

def print_fk(pos, rot, ee_name, angles):
    print(f"FK  →  end-effector: {ee_name}")
    for label, angle in zip(DOF3_LABELS, (angles[1], angles[2], angles[4])):
        print(f"    {label:18s}"f"  {angle:+9.3f}°"f"   ({np.radians(angle):+.5f} rad)")
    print(
        f"  Position (m):"
        f"  x={pos[0]:+.5f}"
        f"  y={pos[1]:+.5f}"
        f"  z={pos[2]:+.5f}"
    )

    print(
        f"  Reach from origin:"
        f" {np.linalg.norm(pos):.5f} m"
    )

    print("  Rotation matrix:")

    for i, row in enumerate(rot):
        print(
            f"    {'xyz'[i]}:"
            f" [{row[0]:+8.5f}"
            f"  {row[1]:+8.5f}"
            f"  {row[2]:+8.5f}]"
        )

def run_native(model, data, act_id):
    with mujoco.viewer.launch_passive(model,data) as viewer:
        viewer.sync()
        time.sleep(0.4)

        for idx, pose in enumerate(POSES):
            if not viewer.is_running():
                break
            pos, rot, ee = forward_kinematics(model,*pose)

            print(f"\n[Pose {idx + 1}/{len(POSES)}]"f" θ = {pose}°")
            print_fk(pos,rot,ee,pose)

            target = (list(np.radians(pose)) + [0.0, 0.0])
            start = [float(data.ctrl[act_id[a]]) for a in ACTUATOR_NAMES]

            for i in range(N_RAMP):
                if not viewer.is_running():
                    return
                alpha = _smooth_step(i / max(N_RAMP - 1, 1))
                ctrl_values = [s + alpha * (t - s) for s, t in zip(start,target)]
                _set_ctrl(data,act_id,ctrl_values)
                mujoco.mj_step(model,data)
                viewer.sync()

            for _ in range(N_SETTLE):
                if not viewer.is_running():
                    return

                _set_ctrl(data,act_id,target)
                mujoco.mj_step(model,data)
                viewer.sync()

        print("\nAll poses completed.","\nClose viewer window to exit.")
        ee_id, _ = find_end_effector(model)
        hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hip_Hip")
        hindarm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hindarm_Hindarm")
        forearm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "forearm_Forearm")
        wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_Wrist")
        hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_Hand")
        counter = 0

        while viewer.is_running():
            mujoco.mj_step(model, data)
            counter += 1
            if counter % 100 == 0:
                print()
                print("Hip          :", data.xpos[hip_id])
                print("Hindarm      :", data.xpos[hindarm_id])
                print("Forearm      :", data.xpos[forearm_id])
                print("Wrist        :", data.xpos[wrist_id])
                print("Hand         :", data.xpos[hand_id])
                print("End-Effector :", data.xpos[ee_id])
                print("Link Lengths   : "f"{np.linalg.norm(data.xpos[hip_id]-data.xpos[hindarm_id]):.5f} m, "
                      f"{np.linalg.norm(data.xpos[hindarm_id]-data.xpos[forearm_id]):.5f} m, "
                      f"{np.linalg.norm(data.xpos[forearm_id]-data.xpos[wrist_id]):.5f} m, "
                      f"{np.linalg.norm(data.xpos[wrist_id]-data.xpos[hand_id]):.5f} m",
                        f"{np.linalg.norm(data.xpos[hand_id]-data.xpos[ee_id]):.5f} m")

            viewer.sync()

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    model.opt.magnetic[:] = [0.0, 0.0, 0.0]
    model.opt.gravity[:] = [0.0, 0.0, 0.0]
    data = mujoco.MjData(model)
    _, act_id = build_maps(model)
    print(f"Actuator IDs: {act_id}")
    _, ee_name = find_end_effector(model)

    print(f"MuJoCo version : "f"{mujoco.__version__}")
    print(f"End-effector   : "f"'{ee_name}'")
    print("Viewer backend : native")
    run_native(model,data,act_id)

if __name__ == "__main__":
    main()