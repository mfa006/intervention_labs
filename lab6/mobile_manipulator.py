from lab6_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.animation as anim
import matplotlib.transforms as trans

# Robot model
d =  np.zeros(3)                           # displacement along Z-axis
theta = np.array([0.2, 0.5, 0.2])
# theta =  np.array([0, np.pi/4, np.pi/4])                       # rotation around Z-axis
alpha =  np.zeros(3)                       # rotation around X-axis
a =  np.array([0.5, 0.75, 0.5])                           # displacement along X-axis
revolute = [True, True, True]                     # flags specifying the type of joints
robot = MobileManipulator(d, theta, a, alpha, revolute)

# Task definition
# === Task Mode Configuration ===
# Use this tuple to define control setup:
# task_mode:        "exercise 1", "exercise 2", or "exercise 3"
# weights:          "none", "joints", "base", "moderate"
# navigate:         "none", "rotate_move", "move_rotate", "move_and_rotate"



config = "ex3_rotmove" # choose one of: "ex1", "ex2_base", "ex2_joints", "ex2_moderate", "ex3_rotmove", "ex3_moverot", "ex3_both"

# === Define presets for all configurations ===
preset_configs = {
    "ex1": ("exercise 1", "none", "none"),

    "ex2_base": ("exercise 2", "base", "none"),
    "ex2_joints": ("exercise 2", "joints", "none"),
    "ex2_moderate": ("exercise 2", "moderate", "none"),

    "ex3_rotmove": ("exercise 3", "none", "rotate_move"),
    "ex3_moverot": ("exercise 3", "none", "move_rotate"),
    "ex3_both": ("exercise 3", "none", "move_and_rotate")
}

# === Unpack config from dictionary ===
task_mode, weights, navigate = preset_configs[config]

# === Configuration checks ===
assert task_mode in ["exercise 1", "exercise 2", "exercise 3"]
assert weights in ["none", "joints", "base", "moderate"]
assert navigate in ["none", "rotate_move", "move_rotate", "move_and_rotate"]

# set task-specific flags and task list
if task_mode == "exercise 1":
    DLS_Weighted = False
    tasks = [
        JointLimits("Joint limits", np.array([-0.5, 0.2]), np.array([0.03, 0.05]), 2, robot),
        Position2D("End-effector position", np.array([1.0, 0.5]).reshape(2, 1), robot)
    ]
elif task_mode in ["exercise 2", "exercise 3"]:
    DLS_Weighted = True
    tasks = [
        Configuration2D("End-effector configuration", np.array([1.0, 0.5, 0]).reshape(3, 1), robot)
    ]

# deterministic targets for reproducible evaluation
config_targets = [
    np.array([1.5, 0.0, 0.0]).reshape(3, 1),          # straight forward
    np.array([0.0, 1.5, np.pi/2]).reshape(3, 1),      # rotate left, move up
    np.array([1.5, 1.5, np.pi/4]).reshape(3, 1),      # arc motion
    np.array([-1.5, -1.5, np.pi]).reshape(3, 1),      # rotate 180, move back
    np.array([0.0, -1.5, -np.pi/2]).reshape(3, 1),    # rotate right, move down
    np.array([0.0, -1.0, -np.pi/2]).reshape(3, 1)     # straight downward
]
# config_targets = [
#     np.array([1.5, 0.0, 0.0]).reshape(3, 1),          # straight forward
#     np.array([0.0, 1.5, np.pi/2]).reshape(3, 1),      # rotate left, move up
#     np.array([1.5, 1.5, np.pi/4]).reshape(3, 1),      # arc motion
#     np.array([-1.0, -1.5, np.pi]).reshape(3, 1),      # turn around and move
#     np.array([0.5, -1.5, -np.pi/2]).reshape(3, 1),    # sharp right and move
#     np.array([-1.2, 1.0, np.pi/3]).reshape(3, 1),     # reverse arc left
#     np.array([1.2, -1.2, -np.pi/3]).reshape(3, 1),    # reverse arc right
#     np.array([0.0, 0.0, 0.0]).reshape(3, 1),          # return to origin
#     np.array([1.8, 0.0, 0.0]).reshape(3, 1),          # long forward
#     np.array([0.0, -1.8, -np.pi/2]).reshape(3, 1),    # rotate right, move down
#     np.array([-1.5, 1.5, 3*np.pi/4]).reshape(3, 1),   # diagonal up left
#     np.array([1.5, -1.5, -3*np.pi/4]).reshape(3, 1),  # diagonal down right
#     np.array([1.0, 1.0, 0.0]).reshape(3, 1),          # small arc
#     np.array([-1.0, 0.0, np.pi]).reshape(3, 1),       # rotate 180 and back
#     np.array([0.0, -1.0, -np.pi/2]).reshape(3, 1)     # straight downward
# ]



# Simulation params
if task_mode=="exercise 3":
    dt = 1.0/10.0
else:
    dt = 1.0/60.0

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
rectangle = patch.Rectangle((-0.25, -0.15), 0.5, 0.3, color='blue', alpha=0.3)
veh = ax.add_patch(rectangle)
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target
PPx = []
PPy = []
time_vector = []
velocity_log = [] # for exercsie 2
target_idx = 0 #id for exercise 3

#store the end-effector pos and mobile pos
base_x, base_y = [], []
ee_x, ee_y = [], []

# Simulation initialization
def init():
    global tasks, time_vector, last_log, target_idx
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    #Choosing the desired pos of end effector (which is index of the last task)
    if tasks[-1].name == "End-effector configuration":
        if task_mode == "exercise 3":
            tasks[-1].setDesired(config_targets[target_idx])
            if target_idx < len(config_targets) - 1:
                target_idx += 1
        else:
            tasks[-1].setDesired(np.array([
                np.random.uniform(-1.5, 1.5),
                np.random.uniform(-1.5, 1.5),
                0.2
            ]).reshape(3, 1))
    else:
        tasks[-1].setDesired(np.array([
            np.random.uniform(-1.5, 1.5),
            np.random.uniform(-1.5, 1.5)
        ]).reshape(2, 1))
    
    if time_vector:
        last_log = time_vector[-1]
    else:
        last_log = 0
    return line, path, point



# Simulation loop
def simulate(t):
    global tasks
    global robot
    global PPx, PPy
    
    ### Recursive Task-Priority algorithm
    P = np.eye(robot.getDOF())  # Null space projector
    # Initialize output vector (joint velocity)
    dq = np.zeros((robot.getDOF(),1))
    # Loop over tasks
    for task in tasks:
        # Update task state
        task.update(robot)

        # 4)Augmentation of the base Task class to include checking if task is active (method: bool isActive()):
        if task.isActive():

            # Compute augmented Jacobian
            J_bar = task.getJacobian() @ P

            
            # Compute task velocityrobot.getDOF()
            # Accumulate velocity
            # select weight
            if DLS_Weighted:
                
                if task_mode == "exercise 3":
                    W = 0.1*np.eye(robot.getDOF())
                elif task_mode == "exercise 2":
                    if weights == "joints":
                        W = 5.0 * np.diag([1, 1, 10, 10, 10])   # More penalty on the joints
                    elif weights == "base":
                        W = 5.0 * np.diag([10, 10, 1, 1, 1])   # More penalty on the base
                    elif weights == "moderate":
                        W = 0.2 * np.diag([2, 2, 3, 1, 2])  # moderate but more penalty on the base 

                dq += DLS(J_bar, 0.1, W) @ (task.getError() - task.getJacobian() @ dq)
                # velocity_log.append(dq.copy())
            else:
                dq += DLS(J_bar, 0.1) @ (task.getError() - task.getJacobian() @ dq)
            

            velocity_log.append(dq.copy())

            # dq = dq + DLS(J_bar, 0.1)@(task.getError() - task.getJacobian()@dq)

            # Update null-space projector
            P = P - np.linalg.pinv(J_bar) @ J_bar
    ###
    # Update robot
    if task_mode == "exercise 3":
        robot.update(dq, dt, navigate)
    else:
        robot.update(dq, dt)
    
    # Update drawing
    # -- Manipulator links
    PP = robot.drawing()
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(tasks[-1].getDesired()[0], tasks[-1].getDesired()[1])    # set to the last task (end effector pos)
    # -- Mobile base
    eta = robot.getBasePose()
    veh.set_transform(trans.Affine2D().rotate(eta[2,0]) + trans.Affine2D().translate(eta[0,0], eta[1,0]) + ax.transData)

    # log base pose
    base_pose = robot.getBasePose()
    base_x.append(base_pose[0, 0])
    base_y.append(base_pose[1, 0])

    # log end-effector pose
    ee_transform = robot.getEETransform()
    ee_x.append(ee_transform[0, 3])
    ee_y.append(ee_transform[1, 3])
    

    # Log time
    time_vector.append(t + last_log)

    return line, veh, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()

# Plot error norm after simulation
# plt.figure(figsize=(8, 6))
if task_mode == "exercise 2":
    # Plot results after simulation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ---- subplot 1: configuration task error (pos + orientation)
    ax1.plot(time_vector, tasks[0].pos_error_norm, label='e1 (position)', linewidth=2)
    ax1.plot(time_vector, tasks[0].ori_error_norm, label='e2 (orientation)', linewidth=2)
    ax1.set_ylabel('Error Norm')
    ax1.set_title('End-Effector Configuration Task Error')
    ax1.grid()
    ax1.legend()

    # ---- subplot 2: joint velocities
    for i in range(robot.getDOF()):
        ax2.plot(time_vector, [v[i, 0] for v in velocity_log], label=f'DOF {i+1}', linewidth=1.5)

    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Velocity [rad/s]')
    ax2.set_title('Velocity Output from Task-Priority Algorithm')
    ax2.grid()
    ax2.legend()

    plt.tight_layout()
    plt.show()
elif task_mode == "exercise 1":
    plt.axhline(y = -0.5, color = 'r', linestyle = '--')
    plt.axhline(y = 0.2, color = 'r', linestyle = '--')
    for i, task in enumerate(tasks):
        plt.plot(time_vector, task.error_norm, label="e{} ({})".format(i+1, task.name))

    plt.xlabel('Time [s]')
    plt.ylabel('Error Norm')
    plt.title('Task-Priority Control Error Norm')
    plt.legend()
    plt.grid(True)

    plt.show()

elif task_mode == "exercise 3": 

    fig_xy, ax_xy = plt.subplots()
    ax_xy.set_title("Mobile Base and End-Effector Trajectories")
    ax_xy.set_xlabel("X [m]")
    ax_xy.set_ylabel("Y [m]")
    ax_xy.grid(True)

    ax_xy.plot(base_x, base_y, label="Base Trajectory", linewidth=2)
    ax_xy.plot(ee_x, ee_y, label="End-Effector Trajectory", linewidth=2)
    ax_xy.legend()
    plt.tight_layout()

    # Save the base and end-effector trajectories in the correct format
    ee_array = np.array([ee_x, ee_y])        # shape (2, N)
    base_array = np.array([base_x, base_y])  # shape (2, N)

    if navigate == "rotate_move":
        np.save('base_trajectory_nav1.npy', [ee_array, base_array])
    elif navigate == "move_rotate":
        np.save('base_trajectory_nav2.npy', [ee_array, base_array])
    elif navigate == "move_and_rotate":
        np.save('base_trajectory_nav3.npy', [ee_array, base_array])

    plt.show()

    

