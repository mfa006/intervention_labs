# Import necessary libraries
from lab2_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# d = np.zeros(2)           # displacement along Z-axis
# q = np.array([0.2, 0.5])  # rotation around Z-axis (theta)
# a = np.array([0.75, 0.5]) # displacement along X-axis
# alpha = np.zeros(2)       # rotation around X-axis 


# Robot definition (3 revolute joint planar manipulator)
d = np.zeros(3)                         # displacement along Z-axis
q = np.array([0, 0.785, 0.785])               # rotation around Z-axis (theta)
alpha = np.zeros(3)                     # displacement along X-axis
a = np.array([0.75, 0.5,0.25])               # displacement around X-axis 
revolute = np.array([True,True,True])                             # flags specifying the type of joints
K = np.diag([1, 1])

# Setting desired position of end-effector to the current one
T = kinematics(d, q.flatten(), a, alpha) # flatten() needed if q defined as column vector !
sigma_d = T[-1][0:2,3].reshape(2,1)

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
# Memory
joint_pos1, joint_pos2, joint_pos3 = [], [], []
time_vector = []

# Simulation initialization
def init():
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    return line, path, point

# Simulation loop
def simulate(t):
    global q, a, d, alpha, revolute, sigma_d
    global PPx, PPy
    
    # Update robot
    T = kinematics(d, q.flatten(), a, alpha)
    J = jacobian(T, revolute)
    
    # Update control
    PP = robotPoints2D(T)
    sigma = T[-1][0:2,3].reshape(2,1)# Current position of the end-effector
    err = sigma_d - sigma    # Error in position
    Jbar = J[0:2, :]         # Task Jacobian
    P = np.eye(3) - np.linalg.pinv(Jbar)@Jbar   # Null space projector
    y = np.array([np.sin(t),np.cos(t),np.sin(t)]).reshape(3,1) # Arbitrary joint velocity
    dq = np.linalg.pinv(Jbar)@K@err + P@y                    # Control signal
    q = q + dt * dq.reshape(3) # Simulation update

    # Update drawing
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(sigma_d[0], sigma_d[1])

    #for plotting the angles
    joint_pos1.append(q[0])
    joint_pos2.append(q[1])
    joint_pos3.append(q[2])
    time_vector.append(t)

    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 60, dt), 
                                interval=10, blit=True, init_func=init, repeat=False)
plt.show()

# Plot the joint positions over time
plt.figure(figsize=(8, 6))
plt.plot(time_vector, joint_pos1, label='Joint 1')
plt.plot(time_vector, joint_pos2, label='Joint 2')
plt.plot(time_vector, joint_pos3, label='Joint 3')
plt.xlabel('Time [s]')
plt.ylabel('Joint Angle [rad]')
plt.title('Joint Positions over Time')
plt.legend()
plt.grid(True)
plt.show()