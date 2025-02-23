import numpy as np # Import Numpy

def DH(d, theta, a, alpha):
    '''
        Function builds elementary Denavit-Hartenberg transformation matrices 
        and returns the transformation matrix resulting from their multiplication.

        Arguments:
        d (double): displacement along Z-axis
        theta (double): rotation around Z-axis
        a (double): displacement along X-axis
        alpha (double): rotation around X-axis

        Returns:
        (Numpy array): composition of elementary DH transformations
    '''
    # 1. Build matrices representing elementary transformations (based on input parameters).
    T_matrix = np.identity(4)
    #First Transformation matrix, T1(Translational, dist. btwn)
    T1 = T_matrix.copy()
    T1[2:3, 3:] = d

    #Second Transformation matrix, T2 (Rotational)
    T2 = T_matrix.copy()
    T2[0:2, 0:2] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    #Third Transformation matrix, T3 (Translational, approach)
    T3 = T_matrix.copy()
    T3[0:1, 3:] = a

    #the fourth transformation
    T4 = T_matrix.copy()
    T4[1:3, 1:3] = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    # 2. Multiply matrices in the correct order (result in T).

    T = T1@T2@T3@T4
    return T

def kinematics(d, theta, a, alpha):
    '''
        Functions builds a list of transformation matrices, for a kinematic chain,
        descried by a given set of Denavit-Hartenberg parameters. 
        All transformations are computed from the base frame.

        Arguments:
        d (list of double): list of displacements along Z-axis
        theta (list of double): list of rotations around Z-axis
        a (list of double): list of displacements along X-axis
        alpha (list of double): list of rotations around X-axis

        Returns:
        (list of Numpy array): list of transformations along the kinematic chain (from the base frame)
    '''
    T = [np.eye(4)] # Base transformation
    # For each set of DH parameters:
    for i in range(len(d)):
        # 1. Compute the DH transformation matrix.
        T_DH = DH(d[i], theta[i], a[i], alpha[i])
        # 2. Compute the resulting accumulated transformation from the base frame.
        cum_T = T[-1]@ T_DH #T[-1] is the final transformation from the base to the end-effector
        # 3. Append the computed transformation to T.
        T.append(cum_T)
    return T


# Inverse kinematics
def jacobian(T, revolute):
    '''
        Function builds a Jacobian for the end-effector of a robot,
        described by a list of kinematic transformations and a list of joint types.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint

        Returns:
        (Numpy array): end-effector Jacobian
    '''
    # 1. Initialize J and O.
    #initializing the jacobian
    J = np.zeros((6, len(T)-1)) #len(T) =number of joints

    #the z-axis of the base frame
    z0 = np.array([0, 0, 1]) #not used

    #the origin of the base frame o_0
    O_0 = np.zeros(3) #not used

    #storing the final transformation from the base to the end-effector T[-1]
    T_n = T[-1]

    #
    O_n = T_n[:3, 3]
    
    # 2. For each joint of the robot
    for i in range(len(T)-1):
        
    #   a. Extract z and o.
        #z_i
        z_i = T[i][:3, 2] #
        #o_i
        O_i = T[i][:3, 3]
    #   b. Check joint type.
        # making a flag for the type of join
        rol=1 if revolute[i] else 0
    #   c. Modify corresponding column of J.
        #computing the linear jacobian 
        J[:3, i] = rol*np.cross(z_i, (O_n - O_i)) + (1 - rol) * z_i
        #computin the angulr jacobian
        J[3:, i] = rol*z_i
    return J

# Damped Least-Squares
def DLS(A, damping):
    '''
        Function computes the damped least-squares (DLS) solution to the matrix inverse problem.

        Arguments:
        A (Numpy array): matrix to be inverted
        damping (double): damping factor

        Returns:
        (Numpy array): inversion of the input matrix
    '''
    A_mult = A@A.T
    # DLS = A.T @ np.linalg.inv(A_mult + (damping**2 * np.eye((A @ A.T).shape[0])))
    DLS = A.T @ np.linalg.inv(A_mult + (damping**2 * np.eye(A.shape[0])))

    return DLS# Implement the formula to compute the DLS of matrix A.

# Extract characteristic points of a robot projected on X-Y plane
def robotPoints2D(T):
    '''
        Function extracts the characteristic points of a kinematic chain on a 2D plane,
        based on the list of transformations that describe it.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
    
        Returns:
        (Numpy array): an array of 2D points
    '''
    P = np.zeros((2,len(T)))
    for i in range(len(T)):
        P[:,i] = T[i][0:2,3]
    return P