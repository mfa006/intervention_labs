from lab4_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.patches as patch

# Robot model
d = np.zeros(3)                            # displacement along Z-axis
theta = np.array([0, np.pi/4, np.pi/4]).reshape(3, 1)                        # rotation around Z-axis
alpha = np.zeros(3)                       # rotation around X-axis
a = np.array([0.75, 0.5, 0.5])                            # displacement along X-axis
revolute = [True, True, True]                     # flags specifying the type of joints
robot = Manipulator(d, theta, a, alpha, revolute) # Manipulator object

exercise_1 = True  # True runs exercise 1 and False to switch to exercise 2

# Task hierarchy definition
obstacle_pos = np.array([0.0, 1.0]).reshape(2,1)
obstacle_r = 0.5
obstacle_pos_1 = np.array([-0.5, -0.75]).reshape(2,1)
obstacle_r_1 = 0.3
obstacle_pos_2 = np.array([0.75, -0.5]).reshape(2,1)
obstacle_r_2 = 0.2

if exercise_1:
    tasks = [ 
          Obstacle2D("Obstacle avoidance", np.array([obstacle_r, obstacle_r+0.05]), robot, obstacle_pos),
          Obstacle2D("Obstacle avoidance", np.array([obstacle_r_1, obstacle_r_1+0.05]), robot, obstacle_pos_1),
          Obstacle2D("Obstacle avoidance", np.array([obstacle_r_2, obstacle_r_2+0.05]), robot, obstacle_pos_2),
          Position2D("End-effector position", np.array([1.0, 0.5]).reshape(2,1), robot),
    ]
else: # exercise 2
    tasks = [
          JointLimits("Joint limits", np.array([-0.5, 0.5]), np.array([0.01, 0.04]), robot),
          Position2D("End-effector position", np.array([1.0, 0.5]).reshape(2,1), robot),
        ] 

# Simulation params
dt = 1.0/60.0

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
if exercise_1:
    ax.set_title('Simulation of {}'.format(' and '.join([task.name for task in tasks]))) # depending on the tasks we are running
ax.set_title('Simulation of Obstacle Avoidance and End-effector Position')
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
if exercise_1:
    ax.add_patch(patch.Circle(obstacle_pos.flatten(), obstacle_r, color='red', alpha=0.3))
    ax.add_patch(patch.Circle(obstacle_pos_1.flatten(), obstacle_r_1, color='blue', alpha=0.3))  # Second obstacle
    ax.add_patch(patch.Circle(obstacle_pos_2.flatten(), obstacle_r_2, color='green', alpha=0.3))  # Third obstacle
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target

PPx = []
PPy = []
time_vector = []


# Simulation initialization
def init():
    global tasks, time_vector, last_log
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])

    #Choosing the desired pos of end effector (which is index of the last task)
    if tasks[-1].name == "End-effector configuration":
        tasks[-1].setDesired(np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0.2]).reshape(3,1))
    else:
        tasks[-1].setDesired(np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)]).reshape(2,1))
    
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
    
    ### Recursive Task-Priority algorithm (w/set-based tasks)
    # The algorithm works in the same way as in Lab4. 
    # The only difference is that it checks if a task is active.
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
            dq = dq + DLS(J_bar, 0.1)@(task.getError() - task.getJacobian()@dq)

            # exercise 2
            print("Task:", task.name)
            print("FFVelocity shape:", task.getFeedForwardVelocity().shape)
            # print("K shape:", task.getK().shape)
            print("Error shape:", task.getError().shape)
            print("Jacobian shape:", task.getJacobian().shape)
            print("dq shape:", dq.shape) 

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
    point.set_data(tasks[-1].getDesired()[0], tasks[-1].getDesired()[1]) #the last task is always the end-effector pos

    # Log time
    time_vector.append(t + last_log)
    
    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()

# Plot error norm after simulation
plt.figure(figsize=(8, 6))
if exercise_1:
    for i, task in enumerate(tasks):
        plt.plot(time_vector, task.error_norm, label="e{} ({})".format(i+1, task.name))

    plt.xlabel('Time [s]')
    plt.ylabel('Error Norm')
    plt.title('Task-Priority Control Error Norm')
    plt.legend()
    plt.grid(True)
    
else:
    plt.axhline(y = -0.5, color = 'r', linestyle = '--')
    plt.axhline(y = 0.5, color = 'r', linestyle = '--')
    for i, task in enumerate(tasks):
        plt.plot(time_vector, task.error_norm, label="e{} ({})".format(i+1, task.name))

    plt.xlabel('Time [s]')
    plt.ylabel('Error Norm')
    plt.title('Task-Priority Control Error Norm')
    plt.legend()
    plt.grid(True)

# Save error plot
filename = "error_{}.png".format('_'.join([task.name.replace(' ', '_') for task in tasks]))
plt.savefig(filename)
plt.show()