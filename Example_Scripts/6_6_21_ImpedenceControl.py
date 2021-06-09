
import numpy as np

class SlidingWindow:
    def __init__(self,):
        """
        INPUTS:
            Demonstration Data:
                X = trajectury [theta,theta_dot,theta_ddot,fe]
                x = current state
                theta_k = joint angles (knee)
                f_e = sensed forces
            Sliding Window Parameters
                l = window length
                t_s = time step to move window
        COMP VARS
            x_diff = difference between current and desired state (xtilde)
                    Xtilde = Xhat - X

            Y = "force" exerted by stiffness matrix according to xerr
                {Kp*(xhat-x)}_t = {I_m*theta_ddot + Kv*theta_dot - fe}_t


        Xhat = equalibrium state (desired trajectory)
        fc = control input
        Kp = full stiffness matricies
        Kv = full damping matricies


        k = number of demonstrations
        N = number of different situations
        """
        self.window_length = 100
        self.time_step = 100
        self.ns = 4 # number of states

        # System Defenition
        self.Im = 4
    def run(self,demonstrations):

        # 1. Interaction Model
        for demonstration in demonstrations:
            Kp_profile = self.stiffness_estimation(demonstration,
                                                   l=self.window_length,
                                                   tstep=self.time_step)

        # 2. Stiffness Estimations
        # 3. Model Fitting
        # 4. Reproduction

    def stiffness_from_demonstration(self,demonstration,l,tstep):

        """
        Kp= 4x4
        X,Y = 1x4

        Sliding Window
            For each window, estimate Kp and append each into list with index t
        """

        #Y = np.empty((self.ns,0))X = np.empty((self.ns, 0))
        Kp = []
        i0 = 0
        i1 = tstep

        # Iterate through window
        while i1 <= len(demonstration):

            #################################################
            #Solve for Stiffness Matrix in This Window#######

            X = []
            Y = []
            for i in range(i0,i1):
                x = demonstration[i][0]
                xdot = demonstration[i][1]
                xddot = demonstration[i][2]
                fe = demonstration[i][3]

                x_tilde = x_hat - x
                y_tilde = self.Im*xddot + Kv*xdot - fe

                X.append(x_tilde)
                Y.append(y_tilde)

            # Calculate Stiffness Estimation at timestep t
            this_Kp = self.solve_least_squares(X, Y)
            this_Kp = self.nearest_psd(this_Kp)
            Kp.append(this_Kp)

            # Advance Window
            t0 = t1
            t1 = t1 +tstep


        return Kp

    def solve_least_squares(self,X,Y):
        return -1

    def nearest_psd(self,A):
        """
        From: https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix
        """
        C = (A + A.T) / 2
        eigval, eigvec = np.linalg.eig(C)
        eigval[eigval < 0] = 0

        return eigvec.dot(np.diag(eigval)).dot(eigvec.T)