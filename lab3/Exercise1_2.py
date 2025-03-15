from lab4_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
# Robot model - 3-link manipulator
d = np.zeros(3)                 # displacement along Z-axis
theta = np.zeros(3) # np.array([0, np.pi/4, np.pi/4]).reshape(3, 1) # rotation around Z-axis
alpha = np.zeros(3)             # rotation around X-axis
a = np.array([0.75, 0.5, 0.5])  # displacement along X-axis
revolute = np.array([True,True,True])             # flags specifying the type of joints
robot = Manipulator(d, theta, a, alpha, revolute) # Manipulator object
# sigma_d = T[-1][0:2,3].reshape(2,1)
# Task hierarchy definition
tasks = [ 
    # Position2D("End-effector position", np.array([1.0, 0.5]).reshape(2,1)),  # Task for end-effector position
    # Orientation2D("End-effector orientation", np.array(np.pi).reshape(1,1))  # Task for end-effector orientation
    # Configuration2D("End-effector configuration", np.array([1.0, 0.5, np.pi]).reshape(3,1)),
    # JointPosition("Joint 1 position", np.array([0]).reshape(1,1), joint=0)  # Ensure shape (1,1)

    # Exercise 2
    Position2D("End-effector position", np.array([1.0, 0.5]).reshape(2, 1), link=3), 
    Orientation2D("End-effector orientation", np.array([[0]]), link=2),

    # Checking if the Configuration2D works
    # Configuration2D("End-effector configuration", np.array([1.0, 0.5, np.pi]).reshape(3,1), link=3),
]
 
# Set K matrix for task 1
K = 2
tasks[0].setK(K)
# Set up FFV
tasks[0].setFeedForwardVelocity(0)


# Simulation params
dt = 1.0/60.0

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target
PPx = []
PPy = []
err_joint_pos1, err_ee_pose = [], []
time_vector = []
last_log = 0

# Simulation initialization
def init():
    global tasks, last_log, time_vector
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])

    #Choosing the desired pos of end effector
    if tasks[0].name == "End-effector configuration":
        tasks[0].setDesired(np.array([np.random.uniform(-1.5, 1.5), np.random.uniform(-1, 1), 0.2]).reshape(3,1))
    else:
        tasks[0].setDesired(np.array([np.random.uniform(-1.5, 1.5), np.random.uniform(-1, 1)]).reshape(2,1))
    
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
    global last_log

    ### Recursive Task-Priority algorithm
    P = np.eye(3)  # Null space projector
    # Initialize output vector (joint velocity)
    dq = np.zeros((3,1)) #.reshape(3, 1)
    # Loop over tasks
    for task in tasks:
        # Update task state
        task.update(robot)
        # Compute augmented Jacobian
        J_bar = task.getJacobian() @ P
        # Compute task velocityrobot.getDOF()
        # dq = dq + DLS(J_bar, 0.1)@(task.getError() - task.getJacobian()@dq)

        # Accumulate velocity
        # dq = dq + DLS(J_bar, 0.1)@(task.getError() - task.getJacobian()@dq)

        # exercise 2
        print("Task:", task.name)
        print("FFVelocity shape:", task.getFeedForwardVelocity().shape)
        print("K shape:", task.getK().shape)
        print("Error shape:", task.getError().shape)
        print("Jacobian shape:", task.getJacobian().shape)
        print("dq shape:", dq.shape)

        dq = dq + DLS(J_bar,0.1) @ (task.getFeedForwardVelocity() + task.getK() @ task.getError() - task.getJacobian() @ dq) 

        # Update null-space projector
        P = P - np.linalg.pinv(J_bar) @ J_bar
    ###

    # Update robot
    robot.update(dq, dt)
    
    # Update drawing
    PP = robot.drawing()
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(tasks[0].getDesired()[0], tasks[0].getDesired()[1])

    # Log time
    time_vector.append(t + last_log)

    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
interval=10, blit=True, init_func=init, repeat=True)
plt.show()

# Plotting error
plt.figure(figsize=(8, 6))
plt.plot(time_vector, tasks[0].error_norm, label="e1 ({})".format(tasks[0].name))
# plt.plot(time_vector, tasks[1].error_norm, label="e1 ({})".format(tasks[1].name))
plt.xlabel('Time [s]')
plt.ylabel('Error [1]')
plt.title('Priority Task: Evolution of the TP control errors.')
plt.legend()
plt.grid(True)
# plt.savefig(f"e2_{task_flag}.png")
plt.show()