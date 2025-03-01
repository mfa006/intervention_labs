# Import necessary libraries
from lab2_robotics_c import * # Import our library (includes Numpy)
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from plot import *
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
# Memory
PPx = []
PPy = []

#robot motion memory with time
q1, q2, time_vector = [], [], []

# Simulation initialization
def init():
    line.set_data([], [])
    path.set_data([], [])
    return line, path

#Simulation loop
def simulate(t):
	global d, q, a, alpha
	global PPx, PPy
    
   # Update robot
	T = kinematics(d, q, a, alpha)
	# dq = np.zeros(2) # Define how joint velocity changes with time!
	dq = np.array([0.3, 0.6])   #how the joint velocity changes with time
    
	q = q + dt * dq
   	 
	#updating q for making plots
	q1.append(q[0]) #robot first joint
	q2.append(q[1]) #robot second joint
	time_vector.append(t)   #simulation time

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


#Simulation of joint positions over time
plt.title('Joint Positions') # Set the title of the plot
plt.xlabel('Time [s]') # Label for the x-axis representing simulation time
plt.ylabel('Angle [rad]') # Label for the y-axis representing joint angles in radians
plt.grid(True) # Enable grid for better readability of the plot
plt.plot(time_vector, q1, label='q1') # Plot the first joint's angle over time
plt.plot(time_vector, q2, label='q2') # Plot the second joint's angle over time
plt.legend() # Display the legend to differentiate between q1 and q2
plt.show() # Render the plot
