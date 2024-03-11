import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import os
import glob
import subprocess
class UnicyclePathFollower:
    def __init__(self, robot, X0, waypoints, alpha, dt=0.05, tf=100,
                  show_animation=False, plotting=None, env=None):
        self.robot = robot
        self.waypoints = waypoints
        self.alpha = alpha
        self.dt = dt
        self.tf = tf

        self.current_goal_index = 0  # Index of the current goal in the path
        self.reached_threshold = 1.0

        self.v_max = 1.0
        self.w_max = 0.5

        self.show_animation = show_animation
        self.plotting = plotting
        self.obs = np.array(env.obs_circle)

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
        if self.robot == 'unicycle2d':
            try:
                from tracking.robots.unicycle2D import Unicycle2D
            except ImportError:
                from robots.unicycle2D import Unicycle2D
            self.robot = Unicycle2D(X0.reshape(-1, 1), self.dt, self.ax)

    def setup_control_problem(self):
        self.u = cp.Variable((2, 1))
        self.u_ref = cp.Parameter((2, 1), value=np.zeros((2, 1)))
        self.A1 = cp.Parameter((1, 2), value=np.zeros((1, 2)))
        self.b1 = cp.Parameter((1, 1), value=np.zeros((1, 1)))
        objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref))
        constraints = [self.A1 @ self.u + self.b1 >= 0,
                       cp.abs(self.u[0]) <= self.v_max,
                       cp.abs(self.u[1]) <= self.w_max]
        self.cbf_controller = cp.Problem(objective, constraints)

    def goal_reached(self, current_position, goal_position):
        return np.linalg.norm(current_position[:2] - goal_position[:2]) < self.reached_threshold

    def nearest_obs(self):
        radius = self.obs[:, 2]
        distances = np.linalg.norm(self.obs[:, :2] - self.robot.X[:2].T, axis=1)
        min_distance_index = np.argmin(distances-radius)
        nearest_obstacle = self.obs[min_distance_index]
        return nearest_obstacle.reshape(-1, 1)


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

            nearest_obs = self.nearest_obs()
            h, dh_dx = self.robot.agent_barrier(nearest_obs)
            self.u_ref.value = self.robot.nominal_input(goal)
            self.A1.value = dh_dx @ self.robot.g()
            self.b1.value = dh_dx @ self.robot.f() + self.alpha * h
            self.cbf_controller.solve(solver=cp.GUROBI, reoptimize=True)

            if self.cbf_controller.status != 'optimal':
                print("ERROR in QP")
                unexpected_beh = -1 # reutn with error
                break

            self.robot.step(self.u.value)
            if self.show_animation:
                self.robot.render_plot()

            # update FOV
            self.robot.update_frontier()
            self.robot.update_safety_area()
            if i > int(1.0 / self.dt): # exclude the first 1 seconds
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
    alpha = 2.0
    tf = 100
    num_steps = int(tf/dt)
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from utils import plotting
    from utils import env

    path_to_continuous_waypoints = os.getcwd()+"/output/state_traj_ori_000.npy"
    path_to_continuous_waypoints = os.getcwd()+"/output/state_traj_vis_000.npy"
    path_to_continuous_waypoints = os.getcwd()+"/output/state_traj_vis_long.npy"
    path_to_continuous_waypoints = os.getcwd()+"/output/240225-0430/state_traj_ori_021.npy"
    waypoints = np.load(path_to_continuous_waypoints, allow_pickle=True)
    waypoints = np.array(waypoints, dtype=np.float64)

    print(waypoints[-1])
    x_init = waypoints[0]
    x_goal = waypoints[-1]

    plot_handler = plotting.Plotting(x_init, x_goal)

    path_follower = UnicyclePathFollower('unicycle2d', x_init, waypoints,  alpha, dt, tf, 
                                         show_animation=True,
                                         plotting=plot_handler,
                                         env=env.Env())
    unexpected_beh = path_follower.run(save_animation=False)
