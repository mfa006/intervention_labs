import numpy as np
import matplotlib.pyplot as plt

# Load saved .npy files containing [ [ee_x, ee_y], [base_x, base_y] ]
nav1 = np.load('base_trajectory_nav1.npy', allow_pickle=True)
nav2 = np.load('base_trajectory_nav2.npy', allow_pickle=True)
nav3 = np.load('base_trajectory_nav3.npy', allow_pickle=True)

# Unpack data
ee1_x, ee1_y = nav1[0]
base1_x, base1_y = nav1[1]

ee2_x, ee2_y = nav2[0]
base2_x, base2_y = nav2[1]

ee3_x, ee3_y = nav3[0]
base3_x, base3_y = nav3[1]


# Set up plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title("Trajectory of Base and End-Effector under Different Integration Modes", fontsize=16, pad=15)
ax.set_xlabel("X Position [m]", fontsize=14)
ax.set_ylabel("Y Position [m]", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_aspect("equal")

# Plot Move then Rotate
ax.plot(ee1_x, ee1_y, 'r--', linewidth=2, label='EE: Move then Rotate')
ax.plot(base1_x, base1_y, 'r-', linewidth=2, label='Base: Move then Rotate')
ax.scatter(ee1_x[0], ee1_y[0], color='r', marker='o', s=60, edgecolors='black')
ax.scatter(ee1_x[-1], ee1_y[-1], color='r', marker='x', s=60)

# Plot Rotate then Move
ax.plot(ee2_x, ee2_y, 'g--', linewidth=2, label='EE: Rotate then Move')
ax.plot(base2_x, base2_y, 'g-', linewidth=2, label='Base: Rotate then Move')
ax.scatter(ee2_x[0], ee2_y[0], color='g', marker='o', s=60, edgecolors='black')
ax.scatter(ee2_x[-1], ee2_y[-1], color='g', marker='x', s=60)

# Plot Simultaneous Move and Rotate
ax.plot(ee3_x, ee3_y, 'b--', linewidth=2, label='EE: Move and Rotate Together')
ax.plot(base3_x, base3_y, 'b-', linewidth=2, label='Base: Move and Rotate Together')
ax.scatter(ee3_x[0], ee3_y[0], color='b', marker='o', s=60, edgecolors='black')
ax.scatter(ee3_x[-1], ee3_y[-1], color='b', marker='x', s=60)

# Legend
ax.legend(fontsize=12, loc='upper left', frameon=True)

# Final layout and show
plt.tight_layout()
plt.show()