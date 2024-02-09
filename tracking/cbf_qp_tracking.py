import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import os

from robots.unicycle2D import Unicycle2D

class UnicyclePathFollower:
    def __init__(self, robot, obs, X0, waypoints, alpha, dt=0.05, tf=20, show_obstacles=True):
        self.robot = robot
        self.obs = obs
        self.waypoints = waypoints
        self.alpha = alpha
        self.dt = dt
        self.tf = tf

        self.current_goal_index = 0  # Index of the current goal in the path
        self.reached_threshold = 0.3

        self.v_max = 1.0
        self.w_max = 0.5

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

        # Setup control problem
        self.setup_robot(X0)
        self.setup_control_problem()

    def setup_robot(self, X0):
        if self.robot == 'unicycle2d':
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


    def run(self):
        for i in range(int(self.tf / self.dt)):
            if self.goal_reached(self.robot.X, np.array(self.waypoints[self.current_goal_index]).reshape(-1, 1)):
                self.current_goal_index += 1
                # Check if all waypoints are reached; if so, stop the loop
                if self.current_goal_index >= len(self.waypoints):
                    print("All waypoints reached.")
                    break
                else:
                    print(f"Moving to next waypoint: {self.waypoints[self.current_goal_index]}")

            goal = np.array(self.waypoints[self.current_goal_index][0:2]) # set goal to next waypoint's (x,y)
            print(np.array(self.waypoints[self.current_goal_index][0:2]))
            h, dh_dx = self.robot.agent_barrier(self.obs)
            self.u_ref.value = self.robot.nominal_input(goal)
            self.A1.value = dh_dx @ self.robot.g()
            self.b1.value = dh_dx @ self.robot.f() + self.alpha * h
            self.cbf_controller.solve(solver=cp.GUROBI, reoptimize=True)

            if self.cbf_controller.status != 'optimal':
                print("ERROR in QP")
                break

            print(f"control input: {self.u.value.T}, h:{h}")
            self.robot.step(self.u.value)
            self.robot.render_plot()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)

        print("Simulation finished")
        plt.ioff()
        #plt.show()
        plt.close()

# Assuming Unicycle2D is defined in the same file or imported
if __name__ == "__main__":
    dt = 0.05
    alpha = 2.0
    tf = 100
    num_steps = int(tf/dt)

    path_to_continuous_waypoints = os.getcwd()+"/output/state_traj.npy"
    waypoints = np.load(path_to_continuous_waypoints, allow_pickle=True)
    waypoints = np.array(waypoints, dtype=np.float64)
    x_init = waypoints[0]

    obs = np.array([0.5, 0.3, 0.1]).reshape(-1, 1)
    #goal = np.array([1, 1])
    path_follower = UnicyclePathFollower('unicycle2d', obs, x_init, waypoints,  alpha, dt, tf, show_obstacles=False)
    path_follower.run()