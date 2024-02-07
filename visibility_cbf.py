
import matplotlib.pyplot as plt
import math
import numpy as np

from scipy.integrate import solve_ivp
from utils.utils import angular_diff, angle_normalize

"""
Created on Jan 23, 2024
@author: Taekyung Kim

@description: This code implements a visibility-aware CBF with an unicycle model using velocity control.
This code also provides a path generation function with the CBF constraint.
The main function shows some examples.

@note: This function imposes visibility constraint using CBF.
Currently. only supports unicycle model with velocity control.

@being-used-in: LQR_CBF_planning.py
"""


class Visibility_CBF:
    def __init__(self):
        self.DT = 0.1  #Integration Length
        self.N = 50 # Number of Control Updates

        # TODO: should be tuned and add explanation
        self.k1_unicyle_cbf = 3 # CBF coefficient for h(x)

        # robot attributes
        self.fov = 90 * (math.pi/180) # field of view
        self.w_lower_lim = -0.3
        self.w_upper_lim = 0.3
        self.unicycle_constant_v = 1.0

    def set_initial_state(self, initial_state):
        self.init_state = np.array(initial_state)

    def set_critical_point(self, critical_point):
        self.critical_point = np.array(critical_point)

    def QP_constraint(self, x_current, u_ref, model="unicycle_velocity_control"):    
        """
        input:
            x_current: x(k)
            u_ref: u(k) 
            - x(k) and u(k) are updated at outer loop
        return:
            True: if CBF constraint is satisfied
            False: if CBF constraint is violated
        """   
        if model == "unicycle_velocity_control":
            x = x_current[0]
            y = x_current[1]
            theta = x_current[2]

            v = u_ref[0] # Linear velocity
            w = u_ref[1] # Angular velocity

            xc = self.critical_point[0]
            yc = self.critical_point[1]
            #print("yc", yc, "xc", xc, "y", y, "x", x)
            #q = 180/3.14
            #print("thetac before norm: ", math.atan2(yc-y, xc-x)*q)
            thetac = angle_normalize(math.atan2(yc-y, xc-x))

            # temporal variable
            z = math.cos(theta)*math.cos(thetac) + math.sin(theta)*math.sin(thetac)
            dtheta = math.acos(z) - self.fov/2
            dtheta = max(0, dtheta) # dtheta should be greater than 0
            # assume self.w_upper_lim is positive and self.w_lower_lim is -1 * self.w_upper_lim
            tt_rot = dtheta/abs(self.w_upper_lim) # time to roate, always greater than 0

            if tt_rot == 0.0:
                # don't even need to rotate
                # don't have to check the ramainder, because it must satisfy the CBF constraint
                #print("tt_rot is 0")
                return True

            dist = math.hypot(x-xc, y-yc)
            tt_reach = dist/v # time to reach the critical point

            # Unicycle with velocity control
            h = tt_reach - tt_rot
            # print("theta, thetac ", theta*q, thetac*q)
            # print("dtheta ", math.acos(z)*q, dtheta*q)
            # print("tt_reach", tt_reach)
            # print("tt_rot", tt_rot)
            # print("z", z)
            Lfh = (x-xc)/dist*math.cos(theta) + (y-yc)/dist*math.sin(theta)
            Lgh = -1/abs(self.w_upper_lim) * (1/math.sqrt(1 - z**2)) * (math.sin(theta)*math.cos(thetac) - math.cos(theta)*math.sin(thetac))
    
            CBF_Constraint = Lfh + Lgh*w + self.k1_unicyle_cbf*h

            #print(CBF_Constraint)

            if CBF_Constraint < 0:
                return False
        return True


    def motion_planning_without_QP(self, u_ref, model="unicycle_velocity_control"):
        if model == "unicycle_velocity_control":
            x_current = self.init_state
            self.x = np.zeros((0, 3)) # generated state traj
            self.u = np.zeros((0, 2)) # generated control traj

            for i in range(len(u_ref)):
                u_current = np.array([self.unicycle_constant_v, u_ref[i]])
                if not self.QP_constraint(x_current, u_current, model="unicycle_velocity_control"):
                    # stop generating traj if CBF constraint is violated
                    print("CBF constraint is violated")
                    break
                else:
                    def unicycle_model_velocity_control(t, y):
                        return [math.cos(y[2]) * self.unicycle_constant_v, math.sin(y[2]) * self.unicycle_constant_v, u_ref[i]]

                    solution = solve_ivp(fun=unicycle_model_velocity_control, t_span=[0, self.DT], y0=x_current, dense_output=True)
                    x_current = solution.y[:, -1] # y[:, -1] to obtain the state after integration, which is at time = self.DT
                    self.x = np.vstack((self.x, x_current))
                    self.u = np.vstack((self.u, u_current)) # vertically stack u

            return self.x, self.u
        
    def plot_traj(self,x,u):
        fig, ax = plt.subplots()
        ax.plot(self.critical_point[0], self.critical_point[1], 'ro')
        thetas = x[:, 2]
        ax.quiver(x[:, 0], x[:, 1], np.cos(thetas), np.sin(thetas))
        ax.plot(x[:,0], x[:,1])
        ax.set_xlim(-1,10)
        ax.set_ylim(-1,10)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        plt.show()

if __name__ == "__main__":
    CBF_Planning = Visibility_CBF()

    # FIXME: should implement QP
    with_QP = False
    without_QP = True

    # Test four examples
    # Case 1
    initial_state = np.array([1.0, 1.0, np.pi/2])
    critical_point = np.array([4.0, 1.0])
    CBF_Planning.set_initial_state(initial_state)
    CBF_Planning.set_critical_point(critical_point)
    u_ref = [-0.1 for _ in range(50)] # w

    if with_QP:
        x, u= CBF_Planning.motion_planning_with_QP(u_ref, model="unicycle_velocity_control")
        CBF_Planning.plot_traj(x,u)
    if without_QP:
        x, u= CBF_Planning.motion_planning_without_QP(u_ref, model="unicycle_velocity_control")
        CBF_Planning.plot_traj(x,u)

    print("\n Case 1 is done \n")

    # Case 2
    initial_state = np.array([1.0, 1.0, np.pi/2])
    critical_point = np.array([1.0, 4.0])
    CBF_Planning.set_initial_state(initial_state)
    CBF_Planning.set_critical_point(critical_point)
    u_ref = [0.1 for _ in range(50)] 

    if with_QP:
        x, u= CBF_Planning.motion_planning_with_QP(u_ref, model="unicycle_velocity_control")
        CBF_Planning.plot_traj(x,u)
    if without_QP:
        x, u= CBF_Planning.motion_planning_without_QP(u_ref, model="unicycle_velocity_control")
        CBF_Planning.plot_traj(x,u)

    print("\n Case 2 is done \n")

    # Case 3
    initial_state = np.array([1.0, 1.0, np.pi/2])
    critical_point = np.array([5.0, 1.0])
    CBF_Planning.set_initial_state(initial_state)
    CBF_Planning.set_critical_point(critical_point)
    u_ref = [-0.5 for _ in range(50)] 

    if with_QP:
        x, u= CBF_Planning.motion_planning_with_QP(u_ref, model="unicycle_velocity_control")
        CBF_Planning.plot_traj(x,u)
    if without_QP:
        x, u= CBF_Planning.motion_planning_without_QP(u_ref, model="unicycle_velocity_control")
        CBF_Planning.plot_traj(x,u)

    print("\n Case 3 is done \n")

    # Case 4
    initial_state = np.array([1.0, 1.0, np.pi/2])
    critical_point = np.array([5.0, 5.0])
    CBF_Planning.set_initial_state(initial_state)
    CBF_Planning.set_critical_point(critical_point)
    u_ref = [-0.5 for _ in range(50)] 

    if with_QP:
        x, u= CBF_Planning.motion_planning_with_QP(u_ref, model="unicycle_velocity_control")
        CBF_Planning.plot_traj(x,u)
    if without_QP:
        x, u= CBF_Planning.motion_planning_without_QP(u_ref, model="unicycle_velocity_control")
        CBF_Planning.plot_traj(x,u)

    print("\n Case 4 is done \n")