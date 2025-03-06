from lab4_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot model - 3-link manipulator
d =                             # displacement along Z-axis
theta =                         # rotation around Z-axis
alpha =                         # rotation around X-axis
a =                             # displacement along X-axis
revolute =                      # flags specifying the type of joints
robot = Manipulator(d, theta, a, alpha, revolute) # Manipulator object

# Task hierarchy definition
tasks = [ 
            Position2D("End-effector position", np.array([1.0, 0.5]).reshape(2,1))
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
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target
PPx = []
PPy = []

# Simulation initialization
def init():
    global tasks
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    return line, path, point

# Simulation loop
def simulate(t):
    global tasks
    global robot
    global PPx, PPy

    ### Recursive Task-Priority algorithm
    # Initialize null-space projector
    # Initialize output vector (joint velocity)
    # Loop over tasks
        # Update task state
        # Compute augmented Jacobian
        # Compute task velocity
        # Accumulate velocity
        # Update null-space projector
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

    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()