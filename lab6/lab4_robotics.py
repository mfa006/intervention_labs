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

    # initialize jacobian  
    J = np.zeros((6, len(T)-1))  # number of joints  

    # store target link transform  
    T_n = T[link]  # transformation of selected link  
    O_n = T_n[:3, 3]  # position of selected link  

    # iterate joints up to target link  
    for i in range(link):  
        # extract z-axis and position  
        z_i = T[i][:3, 2]  # joint z-axis  
        O_i = T[i][:3, 3]  # joint position  

        # check joint type  
        rol = 1 if revolute[i] else 0  # revolute flag  

        # compute jacobian  
        J[:3, i] = rol * np.cross(z_i, (O_n - O_i)) + (1 - rol) * z_i  # linear velocity  
        J[3:, i] = rol * z_i  # angular velocity  

    return J

'''
    Class representing a robotic manipulator.
'''
class Manipulator:
    def __init__(self, d, theta, a, alpha, revolute):
        self.d = d
        self.theta = theta
        self.a = a
        self.alpha = alpha
        self.revolute = revolute
        self.dof = len(self.revolute)
        self.q = np.zeros(self.dof).reshape(-1, 1)
        self.update(0.0, 0.0)

    def update(self, dq, dt):
        self.q += dq * dt
        for i in range(len(self.revolute)):
            if self.revolute[i]:
                self.theta[i] = self.q[i]
            else:
                self.d[i] = self.q[i]
        self.T = kinematics(self.d, self.theta, self.a, self.alpha)

    def drawing(self):
        return robotPoints2D(self.T)

    def getEEJacobian(self):
        return jacobian(self.T, self.revolute)

    def getEETransform(self):
        return self.T[-1]

    def getJointPos(self, joint):
        return self.q[joint, 2]

    def getDOF(self):
        return self.dof
    
    def getLinkTransformation(self, link):
        return self.T[link]
    
    def getLinkJacobian(self, link):
        return jacobianLink(self.T, self.revolute, link)

'''
    Base class representing an abstract Task.
'''
class Task:
    def __init__(self, name, desired, K=None):
        self.name = name 
        self.sigma_d = desired 
        self.error_norm = [] 
        self.FeedForwardVelocity = np.zeros_like(desired)  
        self.K = K #np.eye(len(desired)) 
        self.active = True 
        #
        self.pos_error_norm = []
        self.ori_error_norm = []

    def update(self, robot):
        pass

    def setDesired(self, value):
        self.sigma_d = value

    def getDesired(self):
        return self.sigma_d

    def getJacobian(self):
        return self.J

    def getError(self):
        return self.err
    
    def setFeedForwardVelocity(self, value):
        self.FeedForwardVelocity = np.full_like(self.sigma_d, value, dtype=np.float64) 
    
    def getFeedForwardVelocity(self):
        return self.FeedForwardVelocity
    
    def setK(self, value):
        self.K = np.eye(len(self.sigma_d)) * value
    
    def getK(self):
        return self.K
    
    ####
    def isActive(self):
        return self.active

'''
    Subclass of Task, representing the 2D position task.
'''
class Position2D(Task):
    #for exercise 1
    def __init__(self, name, desired, robot): # constructor with task name, desired state, and robot
        super().__init__(name, desired)
        num_joints = robot.getDOF() # number of joints
        self.J = np.zeros((len(desired), num_joints)) # Initialize Jacobian
        self.err = np.zeros((len(desired), 1)) # Initialize error
        
    def update(self, robot):
        # exercise 1
        self.J = robot.getEEJacobian()[:len(self.sigma_d), :]  # Update Jacobian
        sigma = robot.getEETransform()[:len(self.sigma_d), -1].reshape(2, 1) # end-effector position 
        self.err = self.getDesired() - sigma  # task error
        self.error_norm.append(np.linalg.norm(self.err))  # store error norm
    
    # 4)Augmentation of the base Task class to include checking if task is active
    def isActive(self):
        return True         
'''
    Subclass of Task, representing the 2D orientation task.
'''

class Orientation2D(Task):
    # exercise 1
    def __init__(self, name, desired, robot): # passing the robot to the constructor
        super().__init__(name, desired)
        self.num_joints = robot.getDOF() # number of joints
        self.J = np.zeros((len(desired), self.num_joints)) # initialize jacobian
        self.err = np.array([0]) # initialize error
        
    def update(self, robot):       
        # exercise 1
        self.J = robot.getEEJacobian()[-1, :].reshape(len(self.sigma_d), self.num_joints) # update jacobian
        angle = np.arctan2(robot.getEETransform()[1, 0], robot.getEETransform()[0, 0]) # compute orientation
        self.err = self.getDesired() - angle # task error
        self.error_norm.append(np.linalg.norm(self.err)) # store error norm


'''
    Subclass of Task, representing the 2D configuration task.
'''

class Configuration2D(Task):
    # exercise 1
    def __init__(self, name, desired, robot):  # initialize task
        super().__init__(name, desired)
        self.num_joints = robot.getDOF()  # number of joints
        self.J = np.zeros((len(desired), self.num_joints))  # initialize jacobian
        self.err = np.zeros((len(desired), 1))  # initialize error
        self.K = np.eye(len(desired))  # initialize gain matrix



    def update(self, robot):
        # exercise 1
        self.J[:len(self.sigma_d)-1, :] = robot.getEEJacobian()[:len(self.sigma_d)-1, :]  # position jacobian
        self.J[2, :] = robot.getEEJacobian()[-1, :]  # orientation jacobian
        position = robot.getEETransform()[:2, -1].reshape(2,1)  # end-effector position
        orientation = np.array([[np.arctan2(robot.getEETransform()[1, 0], robot.getEETransform()[0, 0])]])  # end-effector angle 
        sigma = np.vstack([position, orientation])  # combine position and orientation
        self.err = self.getDesired() - sigma  # compute error

        # store norms separately
        pos_err = self.err[0:2]  # x, y error
        ori_err = self.err[2]    # θ error

        self.pos_error_norm.append(np.linalg.norm(pos_err))
        self.ori_error_norm.append(np.abs(ori_err))
        # self.error_norm.append(np.linalg.norm(self.err))  # store error norm
    
    def isActive(self):
        return True


''' 
    Subclass of Task, representing the joint position task.
'''

class JointPosition(Task):
    def __init__(self, name, desired, robot, joint=0):  # default joint is 0
        super().__init__(name, desired)
        self.joint = joint  # selected joint index
        self.num_joints = robot.getDOF()  # number of joints
        self.J = np.zeros((1, self.num_joints))  # initialize jacobian
        self.err = np.zeros((1, 1))  # initialize error

    def update(self, robot):
        self.J = np.zeros((1, robot.getDOF()))  # reset jacobian
        self.J[0, self.joint] = 1  # activate jacobian for selected joint
        
        sigma = robot.getJointPos(self.joint).reshape(1, 1)  # get joint position
        self.err = self.getDesired() - sigma  # compute joint error

# 5)Implementation of the Obstacle2D class
'''
    Implementation of the Obstacle2D class
'''
class Obstacle2D(Task):
    def __init__(self, name, r, robot, obst_pos):  # r = radius range of the obstacle
        super().__init__(name, r)
        self.r = r  # store radius range
        self.obst_pos = obst_pos  # store obstacle position
        self.num_joints = robot.getDOF()  # number of joints
        self.J = np.zeros((len(r), self.num_joints))  # initialize jacobian
        self.err = np.zeros((len(r), 1))  # initialize error
        self.active = False  # obstacle initially inactive
    
    def isActive(self):
        return self.active  # return actual activation state

    def update(self, robot):       
        # get the jacobian for the end-effector
        self.J = robot.getEEJacobian()[:len(self.r), :]  
        
        # get the end-effector position
        pos = robot.getEETransform()[:len(self.r), -1].reshape(len(self.r), 1)  

        # compute error as normalized direction from the obstacle
        err_diff = pos - self.obst_pos
        err_norm_diff = np.linalg.norm(err_diff)
        self.err = err_diff / err_norm_diff    

        # check activation condition based on distance
        if self.active is False and min(err_norm_diff, self.r[0]) == err_norm_diff: # activate if within inner radius
            self.active = True
        elif self.active is True and max(err_norm_diff, self.r[1]) == err_norm_diff: # deactivate if beyond outer radius
            self.active = False
        
        # Store distance norm in error_norm
        self.error_norm.append(err_norm_diff)

class JointLimits(Task):
    def __init__(self, name, joint_limits, activation_margin, joint_num, robot):
        super().__init__(name, 0)
        self.joint_limits = joint_limits  # [lower_limit, upper_limit]
        self.activation_margin = activation_margin  # [activation, deactivation]
        self.num_joints = robot.getDOF()
        self.J = np.zeros((1, self.num_joints))  # jacobian matrix for a single joint task
        self.joint_num = joint_num
        self.err = np.zeros((1, 1))  # error vector
        self.active = 0  # State: -1 (upper limit), 0 (inactive), 1 (lower limit)
    
    def isActive(self):
        return self.active

    def update(self, robot):
    
        self.J.fill(0)
        self.J[0, self.joint_num] = 1 #for joint 1
        
        self.err[0, 0] = self.active #erro update based on the activation state
        
        # Get current position and thresholds
        joint_pos = robot.getJointPos(self.joint_num) #current joint position
        lower, upper = self.joint_limits # joint limits
        act_margin, deact_margin = self.activation_margin # activation thresholds

        # store distance norm in error_norm
        self.error_norm.append(joint_pos)
        
        # updating activation state based on thresholds and joint position
        if self.active == 0:
            # Check upper activation threshold
            if joint_pos >= upper - act_margin:
                self.active = -1  # activate upper limit avoidance
            # Check lower activation threshold
            elif joint_pos <= lower + act_margin:
                self.active = 1   # activate lower limit avoidance
        
        elif self.active == -1:
            # Check upper deactivation threshold
            if joint_pos <= upper - deact_margin:
                self.active = 0   # deactivate upper limit
        elif self.active == 1:
            # Check lower deactivation threshold
            if joint_pos >= lower + deact_margin:
                self.active = 0   # deactivate lower limit 