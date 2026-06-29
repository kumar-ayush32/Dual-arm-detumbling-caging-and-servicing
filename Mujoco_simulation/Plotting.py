import matplotlib.pyplot as plt
import numpy as np
def plot_reaction_forces(rxn_a: list, rxn_b: list):
    print("\n[Plot] Plotting reaction forces at end-effectors")
 
    ra = np.array(rxn_a)   # (T, 3)
    rb = np.array(rxn_b)
 
    samples_a = np.arange(len(ra))
    samples_b = np.arange(len(rb))
 
    labels  = ["Fx (N)", "Fy (N)", "Fz (N)", "|F| (N)"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 6), layout="constrained")
    fig.suptitle("End-Effector Reaction Forces at Target Contact", fontsize=13)
 
    for col in range(3):
        axes[0, col].plot(samples_a, ra[:, col], linewidth=0.8)
        axes[0, col].set_title(f"Arm A: {labels[col]}")
        axes[0, col].set_xlabel("Sample")
        axes[0, col].set_ylabel(labels[col])
        axes[0, col].grid(True, linewidth=0.4)
 
        axes[1, col].plot(samples_b, rb[:, col], linewidth=0.8, color="tab:orange")
        axes[1, col].set_title(f"Arm B: {labels[col]}")
        axes[1, col].set_xlabel("Sample")
        axes[1, col].set_ylabel(labels[col])
        axes[1, col].grid(True, linewidth=0.4)
 
    # Resultant magnitude
    mag_a = np.linalg.norm(ra, axis=1)
    mag_b = np.linalg.norm(rb, axis=1)
 
    axes[0, 3].plot(samples_a, mag_a, color="tab:red", linewidth=0.8)
    axes[0, 3].set_title("Arm A: |F| (N)")
    axes[0, 3].set_xlabel("Sample")
    axes[0, 3].set_ylabel("|F| (N)")
    axes[0, 3].grid(True, linewidth=0.4)
 
    axes[1, 3].plot(samples_b, mag_b, color="tab:purple", linewidth=0.8)
    axes[1, 3].set_title("Arm B: |F| (N)")
    axes[1, 3].set_xlabel("Sample")
    axes[1, 3].set_ylabel("|F| (N)")
    axes[1, 3].grid(True, linewidth=0.4)
 
    plt.show()

def plot_reaction_torques(rxn_a: list, rxn_b: list):
    print("\n[Plot] Plotting reaction torques at end-effectors")
 
    ta = np.array(rxn_a)   # (T, 3)
    tb = np.array(rxn_b)
 
    samples_a = np.arange(len(ta))
    samples_b = np.arange(len(tb))
 
    labels = ["Tx (N·m)", "Ty (N·m)", "Tz (N·m)", "|T| (N·m)"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 6), layout="constrained")
    fig.suptitle("End-Effector Reaction Torques at Target Contact", fontsize=13)
 
    for col in range(3):
        axes[0, col].plot(samples_a, ta[:, col], linewidth=0.8, color="tab:blue")
        axes[0, col].set_title(f"Arm A: {labels[col]}")
        axes[0, col].set_xlabel("Sample")
        axes[0, col].set_ylabel(labels[col])
        axes[0, col].grid(True, linewidth=0.4)
 
        axes[1, col].plot(samples_b, tb[:, col], linewidth=0.8, color="tab:orange")
        axes[1, col].set_title(f"Arm B: {labels[col]}")
        axes[1, col].set_xlabel("Sample")
        axes[1, col].set_ylabel(labels[col])
        axes[1, col].grid(True, linewidth=0.4)
 
    # Resultant magnitude
    mag_a = np.linalg.norm(ta, axis=1)
    mag_b = np.linalg.norm(tb, axis=1)
 
    axes[0, 3].plot(samples_a, mag_a, color="tab:red", linewidth=0.8)
    axes[0, 3].set_title("Arm A: |T| (N·m)")
    axes[0, 3].set_xlabel("Sample")
    axes[0, 3].set_ylabel("|T| (N·m)")
    axes[0, 3].grid(True, linewidth=0.4)
 
    axes[1, 3].plot(samples_b, mag_b, color="tab:purple", linewidth=0.8)
    axes[1, 3].set_title("Arm B: |T| (N·m)")
    axes[1, 3].set_xlabel("Sample")
    axes[1, 3].set_ylabel("|T| (N·m)")
    axes[1, 3].grid(True, linewidth=0.4)
 
    plt.show()

def plot_angles(all_angle_armA, all_angle_armB):
    print("\n[Plot] Plotting angles of all actuators")
    all_angle_armA = np.rad2deg(np.array(all_angle_armA))
    all_angle_armB = np.rad2deg(np.array(all_angle_armB))
    fig,ax=plt.subplots(2,1,figsize=(10,5),layout="constrained")
    ax[0].plot(all_angle_armA)
    ax[0].set_xlabel("Number of samples")
    ax[0].set_ylabel("Angles (in degrees)")
    ax[0].set_title("Angles of ARM-A")
    ax[0].grid()
    ax[1].plot(all_angle_armB)
    ax[1].set_xlabel("Number of samples")
    ax[1].set_ylabel("Angles (in degrees)")
    ax[1].set_title("Angles of ARM-B")
    ax[1].grid()
    plt.show()