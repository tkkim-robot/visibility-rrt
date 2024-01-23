
import math
import scipy
import time
import matplotlib.pyplot as plt
import numpy as np

from cbf import CBF
import utils.env as env
from utils.node import Node

"""
Created on Jan 22, 2024
@author: Taekyung Kim

@description: This code implements a discrete-time finite horizon LQR with unicycle model using velocity control. 
This code also provides a path planning function with the computed LQR gain, both using QP and not using QP.
They are similar to "motion_plannig_with/without_QP()" in cbf.py, but they use LQR gains instead.
The main function shows tracking a randomly generated goal points with LQR planning. (can turn on/off QP)

@note: This code is a refactorized version of the LQR-CBF-RRT* paper with non-linear dynamics.
The code baseline is from Guang Yang.
Please see this origina code for detail : https://github.com/mingyucai/LQR_CBF_rrtStar/blob/main/nonlinear_dynamic_model/LQR_nonlinear_planning.py
Currently. only supports unicycle model with velocity control.

@required-scripts: cbf.py, env.py

@being-used-in: LQR_CBF_rrtStar.py
"""

class LQR_CBF_Planner:

    def __init__(self):

        self.N = 3  # number of state variables
        self.M = 2  # number of control variables
        self.DT = 0.2  # discretization step

        self.MAX_TIME = 8.0  # Maximum simulation time
        self.GOAL_DIST = 0.6 # m

        # LQR gain is invariant
        self.Q = np.matrix("0.5 0 0; 0 1 0; 0 0 0.01")
        self.R = np.matrix("0.1 0; 0 0.01")

        # initialize CBF
        self.env = env.Env()
        self.obs_circle = self.env.obs_circle
        # TODO: currently not supporting rectangle and boundary obstacle
        # self.obs_rectangle = self.env.obs_rectangle
        # self.obs_boundary = self.env.obs_boundary
        self.cbf_rrt_simulation = CBF(self.obs_circle)

    def lqr_cbf_planning(self, start_node, goal_node, LQR_gain, solve_QP = False, show_animation = True):

        # FIXME: add yaw angle into the planning algorithm (currently just compute it using arctan2)
        sx = start_node.x
        sy = start_node.y
        gx = goal_node.x
        gy = goal_node.y
        #gtheta = np.arctan2(gy-sy, gx-sx)
        gtheta = np.arctan2(gy-sy, gx-sx)

        # Linearize system model
        xd = np.matrix([[gx], [gy], [gtheta]])
        ud = np.matrix([[0], [0]])
        self.A, self.B, self.C = self.get_linear_model(xd, ud)

        # Check the hash table to store LQR feedback Gain
        waypoint = (gx, gy)
        if waypoint in LQR_gain:
            # print('found one prebious gain')
            self.K = LQR_gain[waypoint]
        else:
            self.K = self.finite_dLQR(self.A, self.B, self.Q, self.R)
            LQR_gain[waypoint] = self.K

        # TODO: this does not necessary when only using QPConstraint
        stheta = 0  # start angle
        self.cbf_rrt_simulation.set_initial_state(np.array([sx, sy, stheta]))

        # initialize robot trajectroy, start from current state
        rx, ry = [sx], [sy]
        error = []

        xk = np.array([sx, sy, stheta]).reshape(3, 1)  # State vector

        found_path = False

        i = 0 # idx
        time = 0.0
        while time <= self.MAX_TIME:
            time += self.DT

            x = xk - xd
            u = self.K[i] @ x
            # print("x ", x)
            # print("u ", u)
            # print("K ", self.K[i])
            i += 1
            
            if solve_QP:
                #solve QP with CBF, update control input u
                try:
                    u = np.array(u).squeeze() # convert matrix to array
                    u = self.cbf_rrt_simulation.QP_controller([x[0, 0] + gx, x[1, 0] + gy, x[2, 0] + gtheta], u, model = "unicycle")
                    u = np.matrix(u).reshape(2, -1) # convert array to matrix
                except:
                    print('The CBF-QP at current steering step is infeasible')
                    break
            else:
                # check if LQR control input is safe with respect to CBF constraint, not solving QP
                if not self.cbf_rrt_simulation.QP_constraint([x[0, 0] + gx, x[1, 0] + gy, x[2, 0] + gtheta], u, model = "unicycle_velocity_control"):
                    break

            # update current state
            xk = self.A @ xk + self.B @ u + self.C

            rx.append(xk[0, 0])
            ry.append(xk[1, 0])

            d = math.sqrt((gx - rx[-1]) ** 2 + (gy - ry[-1]) ** 2)
            error.append(d)

            if d <= self.GOAL_DIST:
                found_path = True
                # print('errors ', d)
                break

            # animation
            if show_animation:  # pragma: no cover
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])
                plt.plot(sx, sy, "or")
                plt.plot(gx, gy, "ob")
                plt.plot(rx, ry, "-r")
                plt.axis("equal")
                plt.title("iteration: {}".format(i))
                plt.pause(0.5)

        if not found_path:
            print("Cannot found !!")
            return rx, ry, error, found_path

        print("Fonud path to goal")
        return rx, ry, error, found_path


    def finite_dLQR(self, A, B, Q, R):
        """
        Finite horizon discrete-time LQR
        """
        N = 50
        # Create a list of N + 1 elements
        P = [None] * (N + 1)
        Qf = Q
        # LQR via Dynamic Programming
        P[N] = Qf
        # For i = N, ..., 1
        for i in range(N, 0, -1):
            # state cost matrix
            P[i-1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
                R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)   

        K = []
        for i in range(0, N, 1):
            K.append(-np.linalg.inv(R + B.T @ P[i] @ B) @ B.T @ P[i] @ A)
        
        return K

    def infinite_dLQR(self, A, B, Q, R):
        """
        * Currently not used.
        Solve the infinite horizon discrete time lqr controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        """
        # first, solve the ricatti equation
        P = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
        # compute the LQR gain
        K = np.matrix(scipy.linalg.inv(B.T*P*B+R)*(B.T*P*A))

        #eigVals, eigVecs = scipy.linalg.eig(A-B*K)        
        return -K
    
    def get_linear_model(self, x_bar, u_bar):
        """
        Computes the Discrete-time LTI approximated state space model x' = Ax + Bu + C
        """

        x = x_bar[0]
        y = x_bar[1]
        theta = x_bar[2]

        v = u_bar[0]
        yaw = u_bar[1]

        A = np.zeros((self.N, self.N))
        A[0, 2] = -v * np.sin(theta)
        A[1, 2] = v * np.sin(theta)
        A_lin = np.eye(self.N) + self.DT * A

        B = np.zeros((self.N, self.M))
        B[0, 0] = np.cos(theta)
        B[1, 0] = np.sin(theta)
        B[2, 1] = 1
        B_lin = self.DT * B

        f_xu = np.array(
            [v * np.cos(theta), v * np.sin(theta), theta]
        ).reshape(self.N, 1)
        C_lin = self.DT * (
            f_xu - np.dot(A, x_bar.reshape(self.N, 1)) - np.dot(B, u_bar.reshape(self.M, 1))
        )

        return np.round(A_lin, 4), np.round(B_lin, 4), np.round(C_lin, 4)


if __name__ == '__main__':

    print(__file__ + " start!!")
    import random

    SHOW_ANIMATION = True
    SOLVE_QP = False

    ntest = 10  # number of goal
    area = 50.0  # sampling area

    lqr_cbf_planner = LQR_CBF_Planner()

    # initialize a has table for storing LQR gain
    # TODO: this should be modified to be optional 
    LQR_gain = dict()

    for i in range(ntest):
        start_time = time.time()

        gx = random.uniform(-area, area)
        gy = random.uniform(-area, area)

        sx = 0.0
        sy = 0.0

        start_node = Node([sx, sy])
        goal_node = Node([gx, gy])

        print("goal", gy, gx)

        rx, ry, error, foundpath = lqr_cbf_planner.lqr_cbf_planning(start_node, goal_node, LQR_gain=LQR_gain, solve_QP = SOLVE_QP, show_animation=SHOW_ANIMATION)

        print("time of running LQR: ", time.time() - start_time)

        if not SHOW_ANIMATION:
            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
            ax1.plot(sx, sy, "or")
            ax1.plot(gx, gy, "ob")
            ax1.plot(rx, ry, "-r")
            ax1.grid()
            
            ax2.plot(error, label="errors")
            ax2.legend(loc='upper right')
            ax2.grid()
            plt.show()

    print("end main")

