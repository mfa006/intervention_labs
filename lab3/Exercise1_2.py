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
    # Exercise 1
    Position2D("End-effector position", np.array([1.0, 0.5]).reshape(2,1), robot),  # Task for end-effector position
    # Orientation2D("End-effector orientation", np.array([0]).reshape(1,1), robot)  # Task for end-effector orientation
    # Configuration2D("End-effector configuration", np.array([1.0, 0.5, np.pi]).reshape(3,1), robot),
    # JointPosition("Joint 1 position", np.array([0]).reshape(1,1), robot, joint=0)  # Ensure shape (1,1)

    # Exercise 2
    # Position2D("End-effector position", np.array([1.0, 0.5]).reshape(2, 1), robot, link=3), 
    # Orientation2D("End-effector orientation", np.array([0]), robot, link=2),

    # Checking if the Configuration2D works
    # Configuration2D("End-effector configuration", np.array([1.0, 0.5, np.pi]).reshape(3,1), robot, link=3),
]
 
# K gain and feed-forward velocity setup for exercise 2
K = 2   # We tried these Gains [1, 1.5, and 2]
tasks[0].setK(K) 
tasks[0].setFeedForwardVelocity(0)


# Simulation params
dt = 1.0/100.0

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation of {}'.format(' and '.join([task.name for task in tasks]))) # depending on the tasks we are running
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
    line.set_data([], [])  # reset line data
    path.set_data([], [])  # reset path data
    point.set_data([], [])  # reset target point

    # choosing desired position
    if tasks[0].name == "End-effector configuration":  # check task type
        tasks[0].setDesired(np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0.2]).reshape(3,1))  # set 3D target
    else:
        tasks[0].setDesired(np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)]).reshape(2,1))  # set 2D target

    if time_vector:  # check if time exists
        last_log = time_vector[-1]  # update last log
    else:
        last_log = 0  # reset log

    return line, path, point

# Simulation loop
def simulate(t):
    global tasks
    global robot
    global PPx, PPy
    global last_log

    ### Recursive Task-Priority algorithm
    P = np.eye(robot.getDOF())  # Null space projector
    # Initialize output vector (joint velocity)
    dq = np.zeros((robot.getDOF(),1))
    # Loop over tasks
    for task in tasks:
        # Update task state
        task.update(robot)
        # Compute augmented Jacobian
        J_bar = task.getJacobian() @ P
        #exercise 1
        # Compute task velocityrobot.getDOF()
        # Accumulate velocity
        dq = dq + DLS(J_bar, 0.1)@(task.getError() - task.getJacobian()@dq)

        # exercise 2
        # Compute task velocityrobot.getDOF()
        # Accumulate velocity
        # dq = dq + DLS(J_bar,0.1) @ (task.getFeedForwardVelocity() + task.getK() @ task.getError() - task.getJacobian() @ dq) 

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
filename = "simulation_{}.png".format('_'.join([task.name.replace(' ', '_') for task in tasks]))
fig.savefig(filename)

# Plotting error
plt.figure(figsize=(8, 6))
for i, task in enumerate(tasks):
    plt.plot(time_vector, task.error_norm, label="e{} ({})".format(i+1, task.name))
plt.xlabel('Time [s]')
plt.ylabel('Error [1]')
plt.title('Priority Task: Control Error Norm for {}'.format(' and '.join([task.name for task in tasks])))
plt.legend()
plt.grid(True)
filename = "error_{}.png".format('_'.join([task.name.replace(' ', '_') for task in tasks]))
plt.savefig(filename)
# plt.savefig(f"e2_{task_flag}.png")
plt.show()