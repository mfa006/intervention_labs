from lab2_robotics import * 
import numpy as np # Includes numpy import

def jacobianLink(T, revolute, link): # Needed in Exercise 2
    '''
        Function builds a Jacobian for the end-effector of a robot,
        described by a list of kinematic transformations and a list of joint types.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
        link(integer): index of the link for which the Jacobian is computed

        Returns:
        (Numpy array): end-effector Jacobian
    '''

    # Code almost identical to the one from lab2_robotics ...

    # Initializing the Jacobian with the same dimensions as the original code
    J = np.zeros((6, len(T)-1))  # len(T)-1 = number of joints

    # Storing the transformation of the TARGET LINK (instead of end-effector)
    T_n = T[link]  # Use the specified link's transformation
    O_n = T_n[:3, 3]  # Position of the target link

    # 2. For each joint UP TO THE TARGET LINK
    for i in range(link):  # Only consider joints affecting the specified link
        # a. Extract z and o (identical to your original code)
        z_i = T[i][:3, 2]  # Z-axis of joint i
        O_i = T[i][:3, 3]  # Position of joint i

        # b. Check joint type (your original logic)
        rol = 1 if revolute[i] else 0  # Flag for revolute joint

        # c. Modify Jacobian column (same formula as your code)
        # Linear velocity component
        J[:3, i] = rol * np.cross(z_i, (O_n - O_i)) + (1 - rol) * z_i
        # Angular velocity component
        J[3:, i] = rol * z_i

    return J

'''
    Class representing a robotic manipulator.
'''
class Manipulator:
    '''
        Constructor.

        Arguments:
        d (Numpy array): list of displacements along Z-axis
        theta (Numpy array): list of rotations around Z-axis
        a (Numpy array): list of displacements along X-axis
        alpha (Numpy array): list of rotations around X-axis
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
    '''
    def __init__(self, d, theta, a, alpha, revolute):
        self.d = d
        self.theta = theta
        self.a = a
        self.alpha = alpha
        self.revolute = revolute
        self.dof = len(self.revolute)
        self.q = np.zeros(self.dof).reshape(-1, 1)
        self.update(0.0, 0.0)

    '''
        Method that updates the state of the robot.

        Arguments:
        dq (Numpy array): a column vector of joint velocities
        dt (double): sampling time
    '''
    def update(self, dq, dt):
        self.q += dq * dt
        for i in range(len(self.revolute)):
            if self.revolute[i]:
                self.theta[i] = self.q[i]
            else:
                self.d[i] = self.q[i]
        self.T = kinematics(self.d, self.theta, self.a, self.alpha)

    ''' 
        Method that returns the characteristic points of the robot.
    '''
    def drawing(self):
        return robotPoints2D(self.T)

    '''
        Method that returns the end-effector Jacobian.
    '''
    def getEEJacobian(self):
        return jacobian(self.T, self.revolute)

    '''
        Method that returns the end-effector transformation.
    '''
    def getEETransform(self):
        return self.T[-1]

    '''
        Method that returns the position of a selected joint.

        Argument:
        joint (integer): index of the joint

        Returns:
        (double): position of the joint
    '''
    def getJointPos(self, joint):
        return self.q[joint]

    '''
        Method that returns number of DOF of the manipulator.
    '''
    def getDOF(self):
        return self.dof
    
    # 2a get the transformation for a selected link
    def getLinkTransformation(self, link):
        return self.T[link]
    
    # 2b get the Jacobian for a selected link
    def getLinkJacobian(self, link):
        return jacobianLink(self.T, self.revolute, link)

'''
    Base class representing an abstract Task.
'''
class Task:
    '''
        Constructor.

        Arguments:
        name (string): title of the task
        desired (Numpy array): desired sigma (goal)
    '''
    def __init__(self, name, desired):
        self.name = name # task title
        self.sigma_d = desired # desired sigma
        self.error_norm = [] # list to store the error norm
        self.FeedForwardVelocity = np.zeros_like(desired) # 3a.field holding the feedforward velocity vector.
        self.K = np.eye(len(desired)) # 3b.field holding the gain matrix K.
        
    '''
        Method updating the task variables (abstract).

        Arguments:
        robot (object of class Manipulator): reference to the manipulator
    '''
    def update(self, robot):
        pass

    ''' 
        Method setting the desired sigma.

        Arguments:
        value(Numpy array): value of the desired sigma (goal)
    '''
    def setDesired(self, value):
        self.sigma_d = value

    '''
        Method returning the desired sigma.
    '''
    def getDesired(self):
        return self.sigma_d

    '''
        Method returning the task Jacobian.
    '''
    def getJacobian(self):
        return self.J

    '''
        Method returning the task error (tilde sigma).
    '''    
    def getError(self):
        return self.err
    
    # 3c method to set the feedforward velocity vector
    def setFeedForwardVelocity(self, value):
        self.FeedForwardVelocity = np.full_like(self.sigma_d, value, dtype=np.float64) 
    
    # 3d method to get the feedforward velocity vector.
    def getFeedForwardVelocity(self):
        return self.FeedForwardVelocity
    
    # 3e method to set the gain matrix K
    def setK(self, value):
        self.K = np.eye(len(self.sigma_d)) * value
    
    # 3f method to get the gain matrix K
    def getK(self):
        return self.K


'''
    Subclass of Task, representing the 2D position task.
'''
# class Position2D(Task):
#     ### for exercise 1 ###
#     def __init__(self, name, desired, robot): # constructor with task name, desired state, and robot
#         super().__init__(name, desired)
#         num_joints = robot.getDOF() # number of joints
#         self.J = np.zeros((len(desired), num_joints)) # Initialize Jacobian
#         self.err = np.zeros((len(desired), 1)) # Initialize error
    
#     ### for exercise 1 ###
#     def update(self, robot):
#         self.J = robot.getEEJacobian()[:len(self.sigma_d), :]  # Update Jacobian
#         sigma = robot.getEETransform()[:len(self.sigma_d), -1].reshape(2, 1) # end-effector position 
#         self.err = self.getDesired() - sigma  # task error
#         self.error_norm.append(np.linalg.norm(self.err))  # store error norm


class Position2D(Task):
    #for exercise 1
    def __init__(self, name, desired, robot):
        super().__init__(name, desired)
        num_joints = robot.getDOF()
        self.J = np.zeros((len(desired), num_joints)) # Initialize with proper dimensions
        self.err = np.zeros((len(desired), 1))# Initialize with proper dimensions
    
    # #for exercise 2
    # def __init__(self, name, desired, robot, link): # with link index.
    #     super().__init__(name, desired)
    #     num_joints = robot.getDOF()
    #     self.J = np.zeros((len(desired), num_joints))     # Initialize with proper dimensions
    #     self.err = np.zeros_like(desired)   # Initialize with proper dimensions
    #     self.link = link            # Store selected link index
    #     self.FFVelocity = np.zeros_like(desired) # Initialize feedforward velocity vector
    #     self.K = np.eye(len(desired)) # Initialize gain matrix K
        
    def update(self, robot):
        # exercise 1
        self.J = robot.getEEJacobian()[:len(self.sigma_d), :]         # Update task Jacobian
        sigma = robot.getEETransform()[:len(self.sigma_d), -1].reshape(2, 1) #get the last column (end-effector position) 
        self.err = self.getDesired() - sigma # Update task error
        self.error_norm.append(np.linalg.norm(self.err)) # Update error norm

        # # exercise 2
        # self.J = robot.getLinkJacobian(self.link)[:len(self.sigma_d), :]
        # sigma = robot.getLinkTransformation(self.link)[:len(self.sigma_d), 3].reshape(len(self.sigma_d),1)
        # self.err = self.getDesired() - sigma
        # self.error_norm.append(np.linalg.norm(self.err)) # Update error norm


         
'''
    Subclass of Task, representing the 2D orientation task.
'''
# class Orientation2D(Task):
#     ### for exercise 1 ###
#     def __init__(self, name, desired, robot): # passing the robot to the constructor
#         super().__init__(name, desired)
#         self.num_joints = robot.getDOF() # number of joints
#         self.J = np.zeros((len(desired), self.num_joints)) # initialize jacobian
#         self.err = np.array([0]) # initialize error

#     ### for exercise 1 ###
#     def update(self, robot):
#         self.J = robot.getEEJacobian()[-1, :].reshape(len(self.sigma_d), self.num_joints) # update jacobian
#         angle = np.arctan2(robot.getEETransform()[1, 0], robot.getEETransform()[0, 0]) # compute orientation
#         self.err = self.getDesired() - angle # task error
#         self.error_norm.append(np.linalg.norm(self.err)) # store error norm 

class Orientation2D(Task):
    # exercise 1
    def __init__(self, name, desired, robot):
        super().__init__(name, desired)
        self.num_joints = robot.getDOF()
        self.J = np.zeros((len(desired),self.num_joints)) # Initialize with proper dimensions
        self.err = np.array([0]) # Initialize with proper dimensions

    # exercise 2
    # def __init__(self, name, desired, robot, link): # with link index.
    #     super().__init__(name, desired)
    #     self.num_joints = robot.getDOF()
    #     self.J = np.zeros((len(desired), self.num_joints))  
    #     self.FFVelocity = np.zeros_like(desired)  
    #     self.K = np.eye(len(desired))  
    #     self.link = link
        
    def update(self, robot):       
        # exercise 1
        self.J = robot.getEEJacobian()[-1,:].reshape(len(self.sigma_d), self.num_joints)
        angle = np.arctan2(robot.getEETransform()[1,0], robot.getEETransform()[0,0])
        self.err = (self.getDesired() - angle)
        self.error_norm.append(np.linalg.norm(self.err)) # Update error norm

        # exercise 2
        # self.J = robot.getLinkJacobian(self.link)[-1, :].reshape((len(self.sigma_d), self.num_joints))
        # angle = np.arctan2(robot.getLinkTransformation(self.link)[1, 0], robot.getLinkTransformation(self.link)[0, 0])
        # self.err = self.getDesired() - angle
        # self.error_norm.append(np.linalg.norm(self.err)) # Update error norm


'''
    Subclass of Task, representing the 2D configuration task.
'''
# class Configuration2D(Task):
#     # exercise 1
#     def __init__(self, name, desired, robot):  # initialize task
#         super().__init__(name, desired)
#         self.num_joints = robot.getDOF()  # number of joints
#         self.J = np.zeros((len(desired), self.num_joints))  # initialize jacobian
#         self.err = np.zeros((len(desired), 1))  # initialize error

#     def update(self, robot):  
#         # exercise 1
#         self.J[:len(self.sigma_d)-1, :] = robot.getEEJacobian()[:len(self.sigma_d)-1, :]  # position jacobian
#         self.J[2, :] = robot.getEEJacobian()[-1, :]  # orientation jacobian
#         position = robot.getEETransform()[:2, -1].reshape(2,1)  # end-effector position
#         orientation = np.array([[np.arctan2(robot.getEETransform()[1, 0], robot.getEETransform()[0, 0])]])  # end-effector angle 
#         sigma = np.vstack([position, orientation])  # combine position and orientation
#         self.err = self.getDesired() - sigma  # compute error
#         self.error_norm.append(np.linalg.norm(self.err))  # store error norm


class Configuration2D(Task):
    # exercise 1
    def __init__(self, name, desired, robot):
        super().__init__(name, desired)
        self.num_joints = robot.getDOF()
        self.J = np.zeros((len(desired),self.num_joints)) # Initialize with proper dimensions
        self.err = np.zeros((len(desired),1)) # Initialize with proper dimensions

    # exercise 2
    # def __init__(self, name, desired, robot, link):  # with link index.
    #     super().__init__(name, desired)
    #     self.num_joints = robot.getDOF()
    #     self.J = np.zeros((len(desired), self.num_joints))  
    #     self.FFVelocity = np.zeros_like(desired)  
    #     self.K = np.eye(len(desired))  
    #     self.link = link
    #     self.err = np.zeros((len(desired),1)) # Initialize with proper dimensions

        
    def update(self, robot):
        # exercise 1
        self.J[:len(self.sigma_d)-1, :] = robot.getEEJacobian()[:len(self.sigma_d)-1, :]  # First two rows for position
        self.J[-1, :] = robot.getEEJacobian()[-1, :]  # Third row for orientation (rotation about Z)
        # Compute end-effector position and orientation
        position = robot.getEETransform()[:2, -1].reshape(2,1)  # Extract (x, y)
        orientation = np.array([[np.arctan2(robot.getEETransform()[1, 0], robot.getEETransform()[0, 0])]])  # Extract θ (orientation)
        # Compute error
        sigma = np.vstack([position, orientation])  # Stack position and orientation
        self.err = self.getDesired() - sigma  # Compute error for full configuration (x, y, θ)
        self.error_norm.append(np.linalg.norm(self.err))

        # exercise 2
        # self.J[:len(self.sigma_d)-1, :] = robot.getLinkJacobian(self.link)[:len(self.sigma_d)-1, :]
        # self.J[-1, :] = robot.getLinkJacobian(self.link)[-1, :]
        # orientation = np.arctan2(robot.getLinkTransformation(self.link)[1, 0], robot.getLinkTransformation(self.link)[0, 0])
        # position = robot.getLinkTransformation(self.link)[:2, -1].reshape(2,1)
        # sigma = np.vstack([position, orientation])
        # self.err = self.getDesired() - sigma
        # self.error_norm.append(np.linalg.norm(self.err))

''' 
    Subclass of Task, representing the joint position task.
'''
# class JointPosition(Task):
#     # for exercise 1
#     def __init__(self, name, desired, robot, joint=0):  # default joint is 0
#         super().__init__(name, desired)
#         self.joint = joint  # selected joint index
#         self.num_joints = robot.getDOF()  # number of joints
#         self.J = np.zeros((1, self.num_joints))  # initialize jacobian
#         self.err = np.zeros((1, 1))  # initialize error

#     # for exercise 1
#     def update(self, robot):
#         self.J = np.zeros((1, robot.getDOF()))  # reset jacobian
#         self.J[0, self.joint] = 1  # activate jacobian for selected joint
        
#         sigma = robot.getJointPos(self.joint).reshape(1, 1)  # get joint position
#         self.err = self.getDesired() - sigma  # compute joint error

class JointPosition(Task):
    def __init__(self, name, desired, robot, joint=0):  # Default to first joint
        super().__init__(name, desired)
        self.joint = joint  # Store selected joint index
        self.num_joints = robot.getDOF()
        self.J = np.zeros((1, self.num_joints))  # Initialize Jacobian (default 3 DOF)
        self.err = np.zeros((1, 1))  # Initialize error

    def update(self, robot):
        self.J = np.zeros((1, robot.getDOF()))  # Reset Jacobian with correct size
        self.J[0, self.joint] = 1  # Activate the Jacobian for the selected joint

        sigma = robot.getJointPos(self.joint).reshape(1, 1)  # Get selected joint position
        self.err = self.getDesired() - sigma  # Compute joint position error
        self.error_norm.append(np.linalg.norm(self.err))

