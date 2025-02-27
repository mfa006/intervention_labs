# Import necessary libraries
from lab2_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot definition (3 revolute joint planar manipulator)
d =                             # displacement along Z-axis
q =                             # rotation around Z-axis (theta)
alpha =                         # displacement along X-axis
a =                             # rotation around X-axis 
revolute =                      # flags specifying the type of joints

# Desired values of task variables
sigma1_d = np.array([0.0, 1.0]).reshape(2,1) # Position of the end-effector
sigma2_d = np.array([[0.0]]) # Position of joint 1

# Simulation params
dt = 1.0/60.0
Tt = 10 # Total simulation time
tt = np.arange(0, Tt, dt) # Simulation time vector

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.grid()
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target
PPx = []
PPy = []

# Simulation initialization
def init():
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    return line, path, point

# Simulation loop
def simulate(t):
    global q, a, d, alpha, revolute, sigma1_d, sigma2_d
    global PPx, PPy
    
    # Update robot
    T = kinematics(d, q.flatten(), a, alpha)
    J = jacobian(T, revolute)

    # Update control
    # TASK 1
    sigma1 =                 # Current position of the end-effector
    err1 =                   # Error in Cartesian position
    J1 =                     # Jacobian of the first task
    P1 =                     # Null space projector
    
    # TASK 2
    sigma2 =                 # Current position of joint 1
    err2 =                   # Error in joint position
    J2 =                     # Jacobian of the second task
    J2bar =                  # Augmented Jacobian
    
    # Combining tasks
    dq1 =                    # Velocity for the first task
    dq12 =                   # Velocity for both tasks

    q = q + dq12 * dt # Simulation update

    # Update drawing
    PP = robotPoints2D(T)
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(sigma1_d[0], sigma1_d[1])

    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=False)
plt.show()