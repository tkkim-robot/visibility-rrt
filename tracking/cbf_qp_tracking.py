import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import os
import glob
import subprocess
class UnicyclePathFollower:
    def __init__(self, robot, obs, X0, waypoints, alpha, dt=0.05, tf=100, show_obstacles=False, show_animation=False):
        self.robot = robot
        self.obs = obs
        self.waypoints = waypoints
        self.alpha = alpha
        self.dt = dt
        self.tf = tf

        self.current_goal_index = 0  # Index of the current goal in the path
        self.reached_threshold = 0.4

        self.v_max = 2.0
        self.w_max = 1.0

        self.show_animation = show_animation

        if show_animation:
            # Initialize plotting
            plt.ion()
            self.fig = plt.figure()
            self.ax = plt.axes()
            #self.fig, self.ax = plt.subplots()
            # self.ax.set_xlim((-0.5, 2))
            # self.ax.set_ylim((-0.5, 2))
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_aspect(1)

            # Visualize goal and obstacles
            self.ax.scatter(waypoints[:, 0], waypoints[:, 1], c='g')
            if show_obstacles:
                circ = plt.Circle((obs[0,0], obs[1,0]), obs[2,0], linewidth=1, edgecolor='k', facecolor='k')
                self.ax.add_patch(circ)
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


    def run(self, save_animation=False):
        print("===================================")
        print("============  CBF-QP  =============")
        print("Start following the generated path.")
        unexpected_beh = 0

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

            h, dh_dx = self.robot.agent_barrier(self.obs)
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
            if i > int(3.0 / self.dt):
                beyond_flag = self.robot.is_beyond_frontier()
                unexpected_beh += beyond_flag
                if beyond_flag and self.show_animation:
                    print("Cumulative unexpected behavior: {}".format(unexpected_beh))

            if self.show_animation:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                plt.pause(0.01)

                if save_animation:
                    current_directory_path = os.getcwd() 
                    if not os.path.exists(current_directory_path + "/output/animations"):
                        os.makedirs(current_directory_path + "/output/animations")
                    plt.savefig(current_directory_path +
                                "/output/animations/" + "t_step_" + str(i) + ".png")

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

        return unexpected_beh

if __name__ == "__main__":
    dt = 0.05
    alpha = 2.0
    tf = 100
    num_steps = int(tf/dt)

    path_to_continuous_waypoints = os.getcwd()+"/output/state_traj_ori_000.npy"
    path_to_continuous_waypoints = os.getcwd()+"/output/state_traj_vis_000.npy"
    path_to_continuous_waypoints = os.getcwd()+"/output/20240214-141316/state_traj_ori_003.npy"
    waypoints = np.load(path_to_continuous_waypoints, allow_pickle=True)
    waypoints = np.array(waypoints, dtype=np.float64)
    x_init = waypoints[0]

    obs = np.array([0.8, 10.5, 0.2]).reshape(-1, 1)
    #goal = np.array([1, 1])
    path_follower = UnicyclePathFollower('unicycle2d', obs, x_init, waypoints,  alpha, dt, tf, 
                                         show_obstacles=False,
                                         show_animation=True)
    unexpected_beh = path_follower.run(save_animation=False)
