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
transpose_errors, pinverse_errors, DLS_errors = [], [], []
joint_pos = []

# Simulation initialization
def init():
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    return line, path, point

##making the controls for the three control solutions
def control(type: str, J: np.ndarray, lambda_: float = 0.1) -> np.ndarray:
    #dictionary of the three controls
    methods = {
        "transpose": lambda J: J.T,
        "pinverse": np.linalg.pinv,
        "DLS": lambda J: DLS(J, lambda_)
    }
    
    if type in methods:
        return methods[type](J)
    else:
        raise ValueError(f"Invalid controller type '{type}'. Choose from {list(methods.keys())}.")

# --- Helper Function for Error Norm ---
def update_error_norm(pose,err,controller_solution):
    """Computes the error norm and appends it to the corresponding list based on the controller type."""
    error_norm = np.linalg.norm(err)
    
    error_dict = {
        "transpose": transpose_errors,
        "pinverse": pinverse_errors,
        "DLS": DLS_errors, 
        "joint_pose": joint_pos,
    }

    if controller_solution in error_dict:
        error_dict[controller_solution].append(error_norm)
        error_dict["joint_pose"].append(pose)
    else:
        raise ValueError(f"Invalid controller type '{controller_solution}'. Choose from {list(error_dict.keys())}.")

def show_n_save_plots():
    """saves the joint position""" 
    plot_control_error('plot_data/transpose_errors.npy', 'plot_data/pinverse_errors.npy', 'plot_data/DLS_errors.npy')


# Simulation loop
def simulate(t,controller_solution):
    global d, q, a, alpha, revolute, sigma_d
    global PPx, PPy

    # Update robot
    T = kinematics(d, q, a, alpha)
    J = jacobian(T, revolute) # Implement!
 
    # Update control
    #extracting the robot pos in 2D plane for sigma
    P = robotPoints2D(T) 
    P_sigma = [P[0, -1], P[1, -1]]
    sigma = np.array(P_sigma)      # Position of the end-effector
    err =  sigma_d - sigma        # Control error  ~sigma_E = sigma_E,d - sigma_E
    print(err)
    delta_f = K @ err #feedback action   df = K x ~sigma_E
    # X_dot_E = sigma_dot_E + delta_f  # No compensation sigma_dot_E ->0
    dq =control(controller_solution, J[0:2, :]) @ delta_f  # vel_vect = control_sol x delta_f
    q += dt * dq # velocity to position
    update_error_norm(q,err, controller_solution)
    
    # Update drawing
    # P = robotPoints2D(T)
    line.set_data(P[0,:], P[1,:])
    PPx.append(P[0,-1])
    PPy.append(P[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(sigma_d[0], sigma_d[1])

    return line, path, point


#this is where we select the control solution
controller_solution = "DLS"#["transpose","pinverse","DLS"] #transpose solution, pinverse solution and DLS
error_files = {
    "transpose": "plot_data/transpose_errors.npy",
    "pinverse": "plot_data/pinverse_errors.npy",
    "DLS": "plot_data/DLS_errors.npy",
    "joint_pose": "plot_data/joint_pose.npy"
}

# Run simulation with the current controller
ax.set_title(f'{controller_solution} control')
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt),
                                interval=10, blit=True, init_func=init,
                                repeat=False, fargs=(controller_solution,))

# This call blocks until the window is closed (or the simulation ends)
plt.show()

# Save errors to file after simulation is finished
if controller_solution in error_files:
    np.save(error_files[controller_solution], eval(f"{controller_solution}_errors"))
    np.save(error_files["joint_pose"], "joint_pose")
else:
    raise ValueError(f"Invalid controller type '{controller_solution}' for saving errors.")

show_n_save_plots()
