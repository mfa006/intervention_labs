from lab6_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.animation as anim
import matplotlib.transforms as trans

# Robot model
d =  np.zeros(3)                           # displacement along Z-axis
theta =  np.array([0, np.pi/4, np.pi/4])                       # rotation around Z-axis
alpha =  np.zeros(3)                       # rotation around X-axis
a =  np.array([0.75, 0.5, 0.25])                           # displacement along X-axis
revolute = [True, True, True]                     # flags specifying the type of joints
robot = MobileManipulator(d, theta, a, alpha, revolute)

# Task definition

tasks = [
        JointLimits("Joint limits", np.array([-0.5, 0.5]), np.array([0.03, 0.05]), 2, robot), ###
        Position2D("End-effector position", np.array([1.0, 0.5]).reshape(2,1), robot)
        ] 

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
rectangle = patch.Rectangle((-0.25, -0.15), 0.5, 0.3, color='blue', alpha=0.3)
veh = ax.add_patch(rectangle)
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

            # Update null-space projector
            P = P - np.linalg.pinv(J_bar) @ J_bar
    ###
    # Update robot
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

    return line, veh, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()