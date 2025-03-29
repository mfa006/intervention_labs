from lab4_robotics import *

class MobileManipulator:
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

        # 4c) List representing all degrees of freedom of the robot
        self.revoluteExt = [True, False]  # List of joint types extended with base joints
        self.revoluteExt.extend(self.revolute)
        self.dof = len(self.revoluteExt) # Number of DOF of the system 

        # 4b) Field to store distance between robot centre and manipulator base position
        self.r = 0           # Distance from robot centre to manipulator base

        self.q = np.zeros((len(self.revolute),1)) # Vector of joint positions (manipulator)
        # 4a) Field to store the pose of the mobile base (position and orientation)
        self.eta = np.zeros((3,1)) # Vector of base pose (position & orientation)
        self.update(np.zeros((self.dof,1)), 0.0) # Initialise robot state

    '''
        Method that updates the state of the robot.

        Arguments:
        dQ (Numpy array): a column vector of quasi velocities
        dt (double): sampling time
    '''
    
    # 4d) Rewritten update() method including:
    def update(self, dQ, dt, navigate="move_rotate"):
        # update joint pos
        self.q += dQ[2:, 0].reshape(-1,1) * dt

        # update joint angles
        for i in range(len(self.revolute)):
            if self.revolute[i]:
                self.theta[i] = self.q[i]
            else:
                self.d[i] = self.q[i]
        

        # # Update mobile base pose
        # self.eta[0, 0] += dQ[1, 0]*np.cos(self.eta[2,0])*dt
        # self.eta[1, 0] += dQ[1, 0]*np.sin(self.eta[2, 0])*dt
        # self.eta[2, 0] += dQ[0,0]*dt

        # update base pose
        w = dQ[0, 0]   # angular velocity
        v = dQ[1, 0]   # linear velocity

        if navigate == "rotate_move": # rotate, then move
            self.eta[2, 0] += w * dt
            self.eta[0, 0] += v * dt * np.cos(self.eta[2, 0])
            self.eta[1, 0] += v * dt * np.sin(self.eta[2, 0])

        elif navigate == "move_rotate": # move, then rotate
            self.eta[0, 0] += v * dt * np.cos(self.eta[2, 0])
            self.eta[1, 0] += v * dt * np.sin(self.eta[2, 0])
            self.eta[2, 0] += w * dt

        elif navigate == "move_and_rotate": # move and rotate at the same time
            if abs(w) < 1e-6:
                # straight motion
                self.eta[0, 0] += v * dt * np.cos(self.eta[2, 0])
                self.eta[1, 0] += v * dt * np.sin(self.eta[2, 0])
            else:
                # exact integration
                th = self.eta[2, 0]
                self.eta[0, 0] += (v / w) * (np.sin(th + w * dt) - np.sin(th))
                self.eta[1, 0] -= (v / w) * (np.cos(th + w * dt) - np.cos(th))
                self.eta[2, 0] += w * dt

        # base transform
        Tb = np.eye(4)
        Tb[0:2, 3] = self.eta[0:2, 0]
        Tb[0:2, 0:2] = np.array([
            [np.cos(self.eta[2, 0]), -np.sin(self.eta[2, 0])],
            [np.sin(self.eta[2, 0]),  np.cos(self.eta[2, 0])]
        ])

        # full kinematics
        self.theta[0] += -np.pi / 2
        dExt = np.concatenate([[0, self.r], self.d])
        thetaExt = np.concatenate([[np.pi/2, 0], self.theta])
        aExt = np.concatenate([[0, 0], self.a])
        alphaExt = np.concatenate([[np.pi/2, -np.pi/2], self.alpha])

        self.T = kinematics(dExt, thetaExt, aExt, alphaExt, Tb)


    ''' 
        Method that returns the characteristic points of the robot.
    '''
    def drawing(self):
        return robotPoints2D(self.T)

    '''
        Method that returns the end-effector Jacobian.
    '''
    def getEEJacobian(self):
        return jacobian(self.T, self.revoluteExt)

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
        return self.q[joint-2, 0] # 4e)


    def getBasePose(self):  # 4f) Accessor returning the pose of the robot mobile base
        return self.eta

    '''
        Method that returns number of DOF of the manipulator.
    '''
    def getDOF(self):
        return self.dof

    ###
    # 4e) Updated implementation of all other methods, where necessary
    def getLinkJacobian(self, link):
        return jacobianLink(self.T, self.revoluteExt, link)

    def getLinkTransform(self, link):
        return self.T[link]

