import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cvxpy as cp
import os
import glob
import subprocess

"""
Created on Feb 8, 2024
@author: Taekyung Kim

@description: This code implements a CBF-QP controller that tracks a set of waypoints.
It provides two dynamics models: Unicycle2D and DynamicUnicycle2D.
It has useful tools to analyze the union of sensing footprints and the safety of the robot.

@required-scripts: robot.py
"""

class UnicyclePathFollower:
    def __init__(self, type, X0, waypoints, dt=0.05, tf=100,
                  show_animation=False, plotting=None, env=None):
        self.type = type
        self.waypoints = waypoints
        self.dt = dt
        self.tf = tf

        self.current_goal_index = 0  # Index of the current goal in the path
        self.reached_threshold = 1.0

        if self.type == 'Unicycle2D':
            self.alpha = 1.0
            self.v_max = 1.0
            self.w_max = 0.5
        elif self.type == 'DynamicUnicycle2D':
            self.alpha1 = 1.5
            self.alpha2 = 1.5
            # v_max is set to 1.0 inside the robot class
            self.a_max = 0.5
            self.w_max = 0.5
            X0 = np.array([X0[0], X0[1], X0[2], 0.0]).reshape(-1, 1)

        self.show_animation = show_animation
        self.plotting = plotting
        self.obs = np.array(env.obs_circle)
        self.unknown_obs = None

        if show_animation:
            # Initialize plotting
            if self.plotting is None:
                self.fig = plt.figure()
                self.ax = plt.axes()
            else:
                # plot the obstacles
                self.ax, self.fig = self.plotting.plot_grid("Path Following")
            plt.ion()
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_aspect(1)

            # Visualize goal and obstacles
            self.ax.scatter(waypoints[:, 0], waypoints[:, 1], c='g', s=10)
        else:
            self.ax = plt.axes() # dummy placeholder

        # Setup control problem
        self.setup_robot(X0)
        self.setup_control_problem()

    def setup_robot(self, X0):
        try:
            from tracking.robot import BaseRobot
        except ImportError:
            from robot import BaseRobot
        self.robot = BaseRobot(X0.reshape(-1, 1), self.dt, self.ax, self.type)

    def setup_control_problem(self):
        self.u = cp.Variable((2, 1))
        self.u_ref = cp.Parameter((2, 1), value=np.zeros((2, 1)))
        self.A1 = cp.Parameter((1, 2), value=np.zeros((1, 2)))
        self.b1 = cp.Parameter((1, 1), value=np.zeros((1, 1)))
        objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref))

        if self.type == 'Unicycle2D':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u[0]) <= self.v_max,
                           cp.abs(self.u[1]) <= self.w_max]
        elif self.type == 'DynamicUnicycle2D':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                            cp.abs(self.u[0]) <= self.a_max,
                            cp.abs(self.u[1]) <= self.w_max]
        self.cbf_controller = cp.Problem(objective, constraints)

    def goal_reached(self, current_position, goal_position):
        return np.linalg.norm(current_position[:2] - goal_position[:2]) < self.reached_threshold

    def set_unknown_obs(self, unknown_obs):
        # set initially
        self.unknown_obs = unknown_obs
        for (ox, oy, r) in self.unknown_obs :
            self.ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='orange',
                    fill=True,
                    alpha=0.4
                )
            )
        self.robot.test_type = 'cbf_qp'

    def get_nearest_obs(self, detected_obs):
        # if there was new obstacle detected, update the obs
        if len(detected_obs) != 0:
            all_obs = np.vstack((self.obs, detected_obs))
            return np.array(detected_obs).reshape(-1, 1)
        else:
            all_obs = self.obs

        radius = all_obs[:, 2]
        distances = np.linalg.norm(all_obs[:, :2] - self.robot.X[:2].T, axis=1)
        min_distance_index = np.argmin(distances-radius)
        nearest_obstacle = all_obs[min_distance_index]
        return nearest_obstacle.reshape(-1, 1)
    
    def is_collide_unknown(self):
        if self.unknown_obs is None:
            return False
        for obs in self.unknown_obs:
            # check if the robot collides with the obstacle
            robot_radius = self.robot.robot_radius
            distance = np.linalg.norm(self.robot.X[:2] - obs[:2])
            return distance < obs[2] + robot_radius


    def run(self, save_animation=False):
        print("===================================")
        print("============  CBF-QP  =============")
        print("Start following the generated path.")
        early_violation = 0
        unexpected_beh = 0

        ani_idx = 0
        for i in range(int(self.tf / self.dt)):
            if self.goal_reached(self.robot.X, np.array(self.waypoints[self.current_goal_index]).reshape(-1, 1)):
                self.current_goal_index += 1
                # Check if all waypoints are reached; if so, stop the loop
                if self.current_goal_index >= len(self.waypoints):
                    print("All waypoints reached.")
                    break
                else:
                    #print(f"Moving to next waypoint: {self.waypoints[self.current_goal_index]}")
                    pass

            goal = np.array(self.waypoints[self.current_goal_index][0:2]) # set goal to next waypoint's (x,y)

            detected_obs = self.robot.detect_unknown_obs(self.unknown_obs)

            nearest_obs = self.get_nearest_obs(detected_obs)
            self.u_ref.value = self.robot.nominal_input(goal)
            if self.type == 'Unicycle2D':
                h, dh_dx = self.robot.agent_barrier(nearest_obs)
                self.A1.value[0,:] = dh_dx @ self.robot.g()
                self.b1.value[0,:] = dh_dx @ self.robot.f() + self.alpha * h
            elif self.type == 'DynamicUnicycle2D':
                h, h_dot, dh_dot_dx = self.robot.agent_barrier(nearest_obs)
                self.A1.value[0,:] = dh_dot_dx @ self.robot.g()
                self.b1.value[0,:] = dh_dot_dx @ self.robot.f() + (self.alpha1+self.alpha2) * h_dot + self.alpha1*self.alpha2*h

            self.cbf_controller.solve(solver=cp.GUROBI, reoptimize=True)
            collide = self.is_collide_unknown()

            if self.cbf_controller.status != 'optimal' or collide:
                print("ERROR in QP")
                unexpected_beh = -1 # reutn with error
                if self.show_animation:
                    self.robot.render_plot()
                    current_position = self.robot.X[:2].flatten()
                    self.ax.text(current_position[0]+0.5, current_position[1]+0.5, '!', color='red', weight='bold', fontsize=22)
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                    plt.pause(5)

                if save_animation:
                    current_directory_path = os.getcwd() 
                    if not os.path.exists(current_directory_path + "/output/animations"):
                        os.makedirs(current_directory_path + "/output/animations")
                    plt.savefig(current_directory_path +
                                "/output/animations/" + "t_step_" + str(ani_idx) + ".png")
                break

            self.robot.step(self.u.value)
            if self.show_animation:
                self.robot.render_plot()

            # update FOV
            self.robot.update_frontier()
            self.robot.update_safety_area()
            if self.current_goal_index > 5: # exclude the first 1 seconds
                beyond_flag = self.robot.is_beyond_frontier()
                if i < int(5.0 / self.dt):
                    early_violation += beyond_flag
                unexpected_beh += beyond_flag
                if beyond_flag and self.show_animation:
                    print("Cumulative unexpected behavior: {}".format(unexpected_beh))

            if self.show_animation:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                plt.pause(0.01)

                if save_animation and i%2==0:
                    current_directory_path = os.getcwd() 
                    if not os.path.exists(current_directory_path + "/output/animations"):
                        os.makedirs(current_directory_path + "/output/animations")
                    plt.savefig(current_directory_path +
                                "/output/animations/" + "t_step_" + str(ani_idx) + ".png")
                    ani_idx += 1

        if self.show_animation and save_animation:
            subprocess.call(['ffmpeg',
                            '-i', current_directory_path+"/output/animations/" + "/t_step_%01d.png",
                            '-r', '60',  # Changes the output FPS to 30
                            '-pix_fmt', 'yuv420p',
                            current_directory_path+"/output/animations/tracking.mp4"])

            for file_name in glob.glob(current_directory_path +
                            "/output/animations/*.png"):
                os.remove(file_name)

        print("=====   Simulation finished  =====")
        print("===================================\n")
        if self.show_animation:
            plt.ioff()
            plt.close()

        return unexpected_beh, early_violation

if __name__ == "__main__":
    dt = 0.05
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from utils import plotting
    from utils import env


    env_type = env.type
    
    if env_type == 1:
        tf = 100
    elif env_type == 2:
        tf = 30

    num_steps = int(tf/dt)

    if env_type == 1:
        # type 1
        path_to_continuous_waypoints = os.getcwd()+"/output/240312-2128_large_env/state_traj_vis_005.npy"
        path_to_continuous_waypoints = os.getcwd()+"/output/240312-2128_large_env/state_traj_ori_001.npy"

    elif env_type == 2:
        path_to_continuous_waypoints = os.getcwd()+"/output/240225-0430/state_traj_ori_016.npy" # fails with QP 34 16
        path_to_continuous_waypoints = os.getcwd()+"/output/240225-0430/state_traj_vis_021.npy"
        #path_to_continuous_waypoints = os.getcwd()+"/output/240225-0430/state_traj_ori_027.npy"

        path_to_continuous_waypoints = os.getcwd()+"/output/env1_visibility.npy"


    waypoints = np.load(path_to_continuous_waypoints, allow_pickle=True)
    waypoints = np.array(waypoints, dtype=np.float64)

    print(waypoints[-1])
    x_init = waypoints[0]
    x_goal = waypoints[-1]

    plot_handler = plotting.Plotting(x_init, x_goal)
    env_handler = env.Env()

    type = 'Unicycle2D'
    #type = 'DynamicUnicycle2D'
    path_follower = UnicyclePathFollower(type, x_init, waypoints, dt, tf, 
                                         show_animation=True,
                                         plotting=plot_handler,
                                         env=env_handler)
    # randomly generate 5 unknown obstacles
    x_range = env_handler.x_range
    y_range = env_handler.y_range
    # unknown_obs = np.random.uniform(low=[x_range[0], y_range[0], 0], high=[x_range[1], y_range[1], 0], size=(20, 3))
    # unknown_obs[:, 2] = 0.5
    # unknown_obs = np.vstack((unknown_obs, [
    #                     [10, 8, 0.5],
    #                     [10.5, 8, 0.5],
    #                     [11, 8, 0.5]])
    # )

    if env_type == 1:
        unknown_obs = np.array([[13.0, 10.0, 0.5],
                                [12.0, 13.0, 0.5],
                                [15.0, 20.0, 0.5],
                                [20.5, 20.5, 0.5],
                                [24.0, 15.0, 0.5]]) # 45 FOV, type 1 (small)
    elif env_type == 2: 
        unknown_obs = np.array([[9.0, 8.8, 0.3]]) # 45 FOV, type 2 (small)

    path_follower.set_unknown_obs(unknown_obs)
    unexpected_beh = path_follower.run(save_animation=True)
