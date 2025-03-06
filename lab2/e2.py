# Import necessary libraries
from lab2_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot definition (3 revolute joint planar manipulator)
d = np.zeros(3)                            # displacement along Z-axis
q = np.array([0, np.pi/4, np.pi/4]).reshape(3, 1)               # rotation around Z-axis (theta) q1 = 0 [rad], q2 = pi/4 [rad], q3 = pi/6 [rad]
alpha =  np.zeros(3)                       # displacement along X-axis
a =   np.array([0.75, 0.5, 0.5])           # rotation around X-axis 
revolute = np.array([True,True,True])      # flags specifying the type of joints

K1 = np.diag([1, 1])                   # Gain for the first task
K2 = np.diag([1])                      # Gain for the second task

task_flag = 1 # Flag for choosing the case (1 for case 1: end-effector position control, and 2 for case 2: joint position control)
#goals for EE priority
goals = [
    np.array([0.4, -1.2]).reshape(2, 1),
    np.array([-1.2, 0.2]).reshape(2, 1),
    np.array([-0.1, -0.8]).reshape(2, 1),
    np.array([-0.6, -0.6]).reshape(2, 1),
    np.array([0.2, -0.6]).reshape(2, 1),
    np.array([-0.6,1.2]).reshape(2, 1),
    np.array([0.5,0.4]).reshape(2, 1),
    np.array([-0.2,0.2]).reshape(2, 1)
]
#goals for Joint priority
# goals =[
#     np.array([0.4, -0.3]).reshape(2, 1),
#     np.array([-0.2, 0.2]).reshape(2, 1),
#     np.array([-0.1, -0.3]).reshape(2, 1),
#     np.array([-0.2, -0.3]).reshape(2, 1),
#     np.array([0.2, -0.3]).reshape(2, 1),
#     np.array([-0.2,0.3]).reshape(2, 1),
#     np.array([0.5,0.4]).reshape(2, 1),
#     np.array([-0.2,0.2]).reshape(2, 1)
# ]
# Desired values of task variables 
current_goal_idx = 0
sigma1_d = goals[current_goal_idx]
# sigma1_d = np.array([-0.6,0.0]).reshape(2,1) # Position of the end-effector
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
joint_pos1, ee_pose = [], []
time_vector = []

# Simulation initialization
def init(): 
    global current_goal_idx,sigma1_d,goals
    sigma1_d = goals[current_goal_idx]
    line.set_data([], [])
    path.set_data([], [])
    point.set_data(sigma1_d[0], sigma1_d[1])
    current_goal_idx=current_goal_idx+1
    print(current_goal_idx,sigma1_d)
    return line, path, point

# Simulation loop
def simulate(t):
    global q, a, d, alpha, revolute, sigma1_d, sigma2_d
    global PPx, PPy, current_goal_idx
    
    # Update robot
    T = kinematics(d, q.flatten(), a, alpha)
    J = jacobian(T, revolute)
    # Update control
    # Choose the case based on the flag
    if task_flag == 1:
        # CASE 1: End-effector position control as the top hierarchy task
        # TASK 1: End-effector position control
        sigma1 = T[-1][0:2, -1].reshape(2, 1)                # Current position of the end-effector
        err1 = sigma1_d - sigma1                # Error in Cartesian position
        x1_dot = K1 @ err1                      # Velocity for the first task
        J1 = J[0:2, :]                   # Jacobian of the first task
        P1 = np.eye(3) - np.linalg.pinv(J1) @ J1                   # Null space projector
        
        # TASK 2: Joint position control
        sigma2 = q[0]               # Current position of joint 1
        err2 = sigma2_d - sigma2                 # Error in joint position
        x2_dot = K2 @ err2                      # Velocity for the second task
        J2 = np.array([1, 0, 0]).reshape(1, 3)                   # Jacobian of the second task
        J2bar = J2 @ P1                 # Augmented Jacobian
        
        # Combining tasks
        J_DLS1 = DLS(J1, 0.1) # DLS for the first task
        J_DLS2 = DLS(J2bar, 0.1) # DLS for the second task
        dq1 = (J_DLS1 @ x1_dot).reshape(3, 1)                  # Velocity for the first task
        dq12 = dq1 + J_DLS2 @ (x2_dot - J2 @ dq1)                 # Velocity for both tasks
        
        # if np.linalg.norm(err1) < 0.05:
        # # Move to the next goal (cycling through the list)
        #     current_goal_idx = (current_goal_idx + 1) % len(goals)
        #     sigma1_d = goals[current_goal_idx]

    elif task_flag == 2:
        # CASE 2: Joint position control as the top hierarchy task
        # TASK 1: Joint position control
        sigma2 = q[0]               # Current position of joint 1
        err2 = sigma2_d - sigma2                 # Error in joint position
        x2_dot = K2 @ err2                      # Velocity for the second task
        J1 = np.array([1, 0, 0]).reshape(1, 3)                   # Jacobian of the second task
        P1 = np.eye(3) - np.linalg.pinv(J1) @ J1                   # Null space projector

            
        # TASK 2: End-effector position control
        sigma1 = T[-1][0:2, -1].reshape(2, 1)                # Current position of the end-effector
        err1 = sigma1_d - sigma1                # Error in Cartesian position
        x1_dot = K1 @ err1                      # Velocity for the first task
        J2 = J[0:2, :]                   # Jacobian of the first task
        J2bar = J2 @ P1                 # Augmented Jacobian

        # Combining tasks
        J_DLS1 = DLS(J1, 0.1) # DLS for the first task
        J_DLS2 = DLS(J2bar, 0.1) # DLS for the second task
        dq2 = (J_DLS1 @ x2_dot).reshape(3, 1)                  # Velocity for the first task
        dq12 = dq2 + J_DLS2 @ (x1_dot - J1 @ dq2)                 # Velocity for both tasks
        # if np.linalg.norm(err1) < 0.05:
        # # Move to the next goal (cycling through the list)
        #     current_goal_idx = (current_goal_idx + 1) % len(goals)
        #     sigma1_d = goals[current_goal_idx]

    q = q + dq12 * dt # Simulation update

    # Update drawing
    PP = robotPoints2D(T)
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    
    # point.set_data(sigma1_d[0], sigma1_d[1]) #goal
    
    #for plotting
    ee_pose.append(q[2])
    joint_pos1.append(q[0])
    # time_vector.append(t)
    
    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=False, init_func=init, repeat=True)
plt.show()

plt.figure(figsize=(8, 6))
time_vector = [i * dt for i in range(len(joint_pos1))]
plt.plot(time_vector, joint_pos1, label='e_2 (Joint 1 Position)')
plt.plot(time_vector, ee_pose, label='e_1 (End-Effector Position)')
plt.xlabel('Time [s]')
plt.ylabel('Error [1]')
plt.title('Joint Positions over Time')
plt.legend()
plt.grid(True)
plt.show()