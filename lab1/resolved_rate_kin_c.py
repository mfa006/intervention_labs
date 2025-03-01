# Import necessary libraries
from lab2_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from plot import *

# Robot definition
d = np.zeros(2)           # displacement along Z-axis
q = np.array([0.2, 0.5])  # rotation around Z-axis (theta)
a = np.array([0.75, 0.5]) # displacement along X-axis
alpha = np.zeros(2)       # rotation around X-axis 
revolute = [True, True]
sigma_d = np.array([0,1]) # goal position
K = np.diag([1, 1])

# Simulation params
dt = 1.0/60.0

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid()
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target
PPx = []
PPy = []

# Lists to store control error norms
transpose_errors, peusdoinverse_errors, DLS_errors = [], [], []

# Simulation initialization
def init():
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    return line, path, point

##making the controls for the three control solutions
def control(type: str, J: np.ndarray, lambda_: float = 0.1) -> np.ndarray:
# Dictionary storing the three control methods as lambda functions
    methods = {
        "transpose": lambda J: J.T,  # Jacobian Transpose Method: uses the transpose of the Jacobian
        "peusdoinverse": np.linalg.pinv,  # Pseudoinverse Method: uses NumPy's built-in pseudoinverse function
        "DLS": lambda J: DLS(J, lambda_)  # Damped Least Squares (DLS) Method: applies damping to avoid singularities
    }
    
    # Check if the specified control type exists in the dictionary
    if type in methods:
        return methods[type](J)  # Apply the selected control method to the Jacobian
    else:
        
        raise ValueError(f"Invalid controller type '{type}'. Choose from {list(methods.keys())}.") # Raise an error if an invalid control type is provided


# Control Error Norm Function
def update_error_norm(err: np.ndarray, controller_solution: str): 
    error_norm = np.linalg.norm(err)  # Compute the Euclidean norm of the error vector

    # Dictionary mapping each control method to its corresponding error list
    error_dict = {
        "transpose": transpose_errors,  # Store error norms for Jacobian Transpose control
        "peusdoinverse": peusdoinverse_errors,    # Store error norms for Pseudoinverse control
        "DLS": DLS_errors               # Store error norms for Damped Least Squares (DLS) control
    }

    # Append the computed error norm to the corresponding list if the control method is valid
    if controller_solution in error_dict:
        error_dict[controller_solution].append(error_norm)
    else:
        # Raise an error if an invalid control type is provided
        raise ValueError(f"Invalid controller type '{controller_solution}'. Choose from {list(error_dict.keys())}.")

# Simulation loop
def simulate(t):
    global d, q, a, alpha, revolute, sigma_d
    global PPx, PPy

    # Update robot
    T = kinematics(d, q, a, alpha)
    J = jacobian(T, revolute) # Implement!

    # Update control
    P = robotPoints2D(T)  # The positions of the robot's joints and end-effector in 2D space
    P_sigma = [P[0, -1], P[1, -1]]  # Extracting the x and y coordinates of the end-effector
    sigma = np.array(P_sigma)  # Converting extracted position to a NumPy array

    # Compute control error
    err = sigma_d - sigma  # the error difference between the desired and actual end-effector position

    velocity_correction = K @ err  # Compute the velocity correction using the control gain matrix K

    # Compute the joint velocity update using the chosen control method
    dq = control(controller_solution, J[0:2, :]) @ velocity_correction    # The function extracts the first two rows of J (representing linear motion in x and y)
    # and applies the selected control method (Transpose, Pseudoinverse, or DLS) to compute dq.

    # Update joint positions using the computed velocity update
    q += dt * dq  


    # Update drawing
    # P = robotPoints2D(T)
    line.set_data(P[0,:], P[1,:])
    PPx.append(P[0,-1])
    PPy.append(P[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(sigma_d[0], sigma_d[1])

    # Update error norm
    update_error_norm(err, controller_solution)

    return line, path, point


#this is where we select the control solution
controller_solution = "DLS"#["transpose","peusdoinverse","DLS"] #transpose solution, peusdoinverse solution and DLS
error_files = {
    "transpose": "transpose_errors.npy",
    "peusdoinverse": "peusdoinverse_errors.npy",
    "DLS": "DLS_errors.npy",
}

# Run simulation with the current controller
ax.set_title(f'{controller_solution} control')
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)

# This call blocks until the window is closed (or the simulation ends)
plt.show()

# Save errors to file after simulation is finished
if controller_solution in error_files:
    np.save(error_files[controller_solution], eval(f"{controller_solution}_errors"))
else:
    raise ValueError(f"Invalid controller type '{controller_solution}' for saving errors.")
