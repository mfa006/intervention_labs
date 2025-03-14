import numpy as np 
from scipy.linalg import block_diag
import math


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
    # 2. Multiply matrices in the correct order (result in T).
    
    # Translation along z-axis
    # M1 = block_diag(np.identity(2), np.identity(2))
    # M1[2, 3] = d

    # # Rotation about z-axis
    # M2 = block_diag(np.array([[np.cos(theta), -np.sin(theta)],
    #                           [np.sin(theta),  np.cos(theta)]]), 1, 1)

    # # Translation along x-axis
    # M3 = block_diag(np.identity(2), np.identity(2))
    # M3[0, -1] = a

    # # Rotation about x-axis
    # M4 = block_diag(1, np.array([[np.cos(alpha), -np.sin(alpha)],
    #                              [np.sin(alpha),  np.cos(alpha)]]), 1)

    # # Combine all transformations
    # T = M1 @ M2 @ M3 @ M4
    # return T
    T = np.array([[math.cos(theta), -math.sin(theta)*math.cos(alpha), math.sin(theta)*math.sin(alpha), a*math.cos(theta)],
                   [math.sin(theta), math.cos(theta)*math.cos(alpha), -math.cos(theta)*math.sin(alpha), a*math.sin(theta)],
                   [0, math.sin(alpha), math.cos(alpha), d],
                   [0, 0, 0, 1]])
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
    # 1. Compute the DH transformation matrix.
    # 2. Compute the resulting accumulated transformation from the base frame.
    # 3. Append the computed transformation to T.
    for d_i,theta_i,a_i,alpha_i in zip(d,theta,a,alpha):
        T.append(T[-1]@DH(d_i,theta_i,a_i,alpha_i))
        #Appending a new transformation after multiplying it with the last transformation. 
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
    # 2. For each joint of the robot
    #   a. Extract z and o.
    #   b. Check joint type.
    #   c. Modify corresponding column of J.
    J = []
    O_n = T[-1][:3,-1]  # points of last transform
    for Ti,rev_flag in zip(T,revolute):
        Ri = Ti[:3,:3]  #Rotation matrix 3x3 matrix 
        Oi = Ti[:3,-1] #Origin vector
        zi = Ri[:3,-1]  #z-vector
        J.append(np.vstack([(np.cross(rev_flag*zi,(O_n-Oi)) +
                                           (1-rev_flag)*zi).reshape(3,1),(rev_flag*zi).reshape(3,1)]))
        # For revolute joints (rev_flag=1): 
        #   J_i =[ [ z_i x (O_n - O_i) ],
        #         [       z_i         ]]
        # For prismatic joints (rev_flag=0):
        #   J_i =[ [ z_i ],
        #         [ 0 ]]
    J = np.hstack(J) #stacking jacobians

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
    # A_dls = A^T * (A*A^T + λ^2*I)^(-1)
    # DLS = A.T@np.linalg.inv(A@A.T+(damping**2)*np.identity(2)) 
    A_mult = A@A.T
    DLS = A.T @ np.linalg.inv(A_mult + (damping**2 * np.eye((A @ A.T).shape[0])))

    return DLS # Implement the formula to compute the DLS of matrix A.

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