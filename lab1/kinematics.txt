# Import necessary libraries
from lab2_robotics import * # Import our library (includes Numpy)
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot definition (planar 2 link manipulator)
d = np.zeros(2)           # displacement along Z-axis
q = np.array([0.2, 0.5])  # rotation around Z-axis (theta)
a = np.array([0.75, 0.5]) # displacement along X-axis
alpha = np.zeros(2)       # rotation around X-axis 

# Simulation params
dt = 0.01 # Sampling time
Tt = 10 # Total simulation time
tt = np.arange(0, Tt, dt) # Simulation time vector

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Kinematics')
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.set_aspect('equal')
ax.grid()
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'r-', lw=1) # End-effector path
joint_pos = []
# Memory
PPx = []
PPy = []

# Simulation initialization
def init():
    line.set_data([], [])
    path.set_data([], [])
    return line, path

# Simulation loop
def simulate(t):
    global d, q, a, alpha
    global PPx, PPy, joint_pos
    
    # Update robot
    T = kinematics(d, q, a, alpha)
    dq = np.array([0.5,2.0])#np.ones(2) # Define how joint velocity changes with time!
    q = q + dt * dq
    joint_pos.append(q)
    # Update drawing
    PP = robotPoints2D(T)
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    
    return line, path

# Run simulation
animation = anim.FuncAnimation(fig, simulate, tt, 
                                interval=1, blit=True, init_func=init, repeat=False)
plt.show()
joint_pos_file = "plot_data/joint_pose.npy"
np.save(joint_pos_file, joint_pos)
joint_pos = np.load(joint_pos_file) 

# Load the joint positions from the saved file
joint_pos_data = np.load(joint_pos_file)
# Convert to a numpy array  
joint_pos_data = np.array(joint_pos_data.tolist())

# Create a time vector corresponding to the simulation time
time_vector = np.linspace(0, Tt, len(joint_pos_data))

# Plot the joint positions over time
plt.figure(figsize=(8, 6))
plt.plot(time_vector, joint_pos_data[:, 0], label='Joint 1')
plt.plot(time_vector, joint_pos_data[:, 1], label='Joint 2')
plt.xlabel('Time [s]')
plt.ylabel('Joint Angle [rad]')
plt.title('Joint Positions over Time')
plt.legend()
plt.grid(True)
plt.show()