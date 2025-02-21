# Import necessary libraries
from lab2_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot definition
d = np.zeros(2)           # displacement along Z-axis
q = np.array([0.2, 0.5])  # rotation around Z-axis (theta)
a = np.array([0.75, 0.5]) # displacement along X-axis
alpha = np.zeros(2)       # rotation around X-axis 
revolute = [True, True]
sigma_d = np.array([1.0, 1.0])
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

# Simulation initialization
def init():
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    return line, path, point

# #this is where we select the control solution
# controller_solution = "transpose" #transpose solution, pinverse solution and dls

# # Lists to store control error norms
# transpose_errors, pinverse_errors, DLS_errors = [], [], []

# ##making the controls for the three control solutions
# def control(type: str, J: np.ndarray, lambda_: float = 0.1) -> np.ndarray:
#     #dictionary of the three controls
#     methods = {
#         "transpose": lambda J: J.T,
#         "pinverse": np.linalg.pinv,
#         "DLS": lambda J: DLS(J, lambda_)
#     }
    
#     if type in methods:
#         return methods[type](J)
#     else:
#         raise ValueError(f"Invalid controller type '{type}'. Choose from {list(methods.keys())}.")

# # --- Helper Function for Error Norm ---
# def update_error_norm(err: np.ndarray, controller_solution: str):
#     """Computes the error norm and appends it to the corresponding list based on the controller type."""
#     error_norm = np.linalg.norm(err)
    
#     error_dict = {
#         "transpose": transpose_errors,
#         "pinverse": pinverse_errors,
#         "DLS": DLS_errors
#     }

#     if controller_solution in error_dict:
#         error_dict[controller_solution].append(error_norm)
#     else:
#         raise ValueError(f"Invalid controller type '{controller_solution}'. Choose from {list(error_dict.keys())}.")


# Simulation loop
def simulate(t):
    global d, q, a, alpha, revolute, sigma_d
    global PPx, PPy

    # Update robot
    T = kinematics(d, q, a, alpha)
    J = jacobian(T, revolute) # Implement!

    # Update control
    sigma_d = np.array([1.0, 1.0])     # Position of the end-effector
    err =  0.1       # Control error
    dq = np.array([0.6,0.6])#np.ones(2)# Control solution
    q += dt * dq

    # # Update control
    # #extracting the robot pos in 2D plane for sigma
    # P = robotPoints2D(T)
    # # P_sigma = [P[0,-1], P[1,-1]]
    # P_sigma = [P[0, -1], P[1, -1]]
    # sigma = np.array(P_sigma)      # Position of the end-effector
    # err =  sigma_d - sigma        # Control error
    # delta_f = K @ err #feedback action 
    # dq =control(controller_solution, J[0:2, :]) @ delta_f 
    
    # Update drawing
    P = robotPoints2D(T)
    line.set_data(P[0,:], P[1,:])
    PPx.append(P[0,-1])
    PPy.append(P[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(sigma_d[0], sigma_d[1])

    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()

# # --- Save Errors to File ---
# error_files = {
#     "transpose": "transpose_errors.npy",
#     "pinverse": "pinverse_errors.npy",
#     "DLS": "DLS_errors.npy"
# }

# if controller_solution in error_files:
#     np.save(error_files[controller_solution], eval(f"{controller_solution}_errors"))
# else:
#     raise ValueError(f"Invalid controller type '{controller_solution}' for saving errors.")