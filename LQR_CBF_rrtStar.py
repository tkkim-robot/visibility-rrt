import math
import time
import copy
import numpy as np

from utils import env, plotting, utils
from utils.node import Node
from LQR_CBF_planning import LQR_CBF_Planner


"""
Created on Jan 23, 2024
# FIXME: write desciprtion
@author: Taekyung Kim

@description:

@note: 

@required-scripts: LQR_CBF_planning.py, env.py

"""

class LQRrrtStar:
    def __init__(self, x_start, x_goal, step_len,
                 goal_sample_rate, search_radius, iter_max, solve_QP=False):
        self.x_start = Node(x_start)
        self.x_goal = Node(x_goal)
        self.step_len = step_len
        self.goal_len = 8
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max
        self.vertex = [self.x_start]
        self.path = []

        self.env = env.Env()
        self.plotting = plotting.Plotting(x_start, x_goal)
        self.utils = utils.Utils()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        # self.obs_circle = self.env.obs_circle

        self.lqr_cbf_planner = LQR_CBF_Planner()
        self.LQR_Gain = dict() # call by reference, so it's modified in LQRPlanner
        self.solve_QP = solve_QP

    def planning(self):
        start_time = time.time()
        for k in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.LQR_steer(node_near, node_rand)


            if k % 100 == 0:
                print('rrtStar sampling iterations: ', k)
                print('rrtStar 1000 iterations sampling time: ', time.time() - start_time)
                start_time = time.time()

            if k % 1000 == 0:
                print('rrtStar sampling iterations: ', k)
                self.plotting.animation_online(self.vertex, "rrtStar", True)

            if node_new and not self.utils.is_collision(node_near, node_new):
                neighbor_index = self.find_near_neighbor(node_new)
                self.vertex.append(node_new)

                if neighbor_index:
                    self.LQR_choose_parent(node_new, neighbor_index)
                    self.rewire(node_new, neighbor_index)
        
        index = self.search_goal_parent()

        if index is None:
            print('No path found!')
            return None

        self.path = self.extract_path(self.vertex[index])
        # visualization
        self.plotting.animation(self.vertex, self.path, "rrt*, N = " + str(self.iter_max))

    def sample_path(self, wx, wy, step=0.2):
        # smooth path
        px, py, traj_costs = [], [], []

        for i in range(len(wx) - 1):
            for t in np.arange(0.0, 1.0, step):
                px.append(t * wx[i+1] + (1.0 - t) * wx[i])
                py.append(t * wy[i+1] + (1.0 - t) * wy[i])

        dx, dy = np.diff(px), np.diff(py)
        traj_costs = [math.sqrt(idx ** 2 + idy ** 2) for (idx, idy) in zip(dx, dy)]
        return px, py, traj_costs

    def LQR_steer(self, node_start, node_goal,exact_steering = False):
        ##balance the distance of node_goal
        dist, theta = self.get_distance_and_angle(node_start, node_goal)
        if not exact_steering: 
            dist = min(self.step_len, dist)
            show_animation = False
        else: 
            show_animation = False
        node_goal.x = node_start.x + dist * math.cos(theta)
        node_goal.y = node_start.y + dist * math.sin(theta)


        wx, wy, _, _, = self.lqr_cbf_planner.lqr_cbf_planning(node_start, node_goal, self.LQR_Gain, solve_QP=self.solve_QP, show_animation=show_animation)
        px, py, traj_cost = self.sample_path(wx, wy)



        if len(wx) == 1:
            return None
        node_new = Node((wx[-1], wy[-1]))
        node_new.parent = node_start
        # calculate cost of each new_node
        node_new.cost = node_start.cost + sum(abs(c) for c in traj_cost)
        #node_new.StateTraj = np.array([px,py]) # Will be needed for adaptive sampling 
        return node_new

    def cal_LQR_new_cost(self, node_start, node_goal):
        wx, wy, _, can_reach = self.lqr_cbf_planner.lqr_cbf_planning(node_start, node_goal, self.LQR_Gain, show_animation=False, solve_QP=self.solve_QP)
        px, py, traj_cost = self.sample_path(wx, wy)
        if wx is None:
            return float('inf'), False
        

        return node_start.cost + sum(abs(c) for c in traj_cost), can_reach

    def LQR_choose_parent(self, node_new, neighbor_index):
        cost = []
        for i in neighbor_index:

            # check if neighbor_node can reach node_new
            _, _, _, can_reach = self.lqr_cbf_planner.lqr_cbf_planning(self.vertex[i], node_new, self.LQR_Gain, show_animation=False, solve_QP=self.solve_QP)

            if can_reach and not self.utils.is_collision(self.vertex[i], node_new):  #collision check should be updated if using CBF
                update_cost, _ = self.cal_LQR_new_cost(self.vertex[i], node_new)
                cost.append(update_cost)
            else:
                cost.append(float('inf'))
        min_cost = min(cost)

        if min_cost == float('inf'):
            print('There is no good path.(min_cost is inf)')
            return None

        cost_min_index = neighbor_index[int(np.argmin(cost))]
        node_new.parent = self.vertex[cost_min_index] 
        node_new.parent.childrenNodeInds.add(len(self.vertex)-1) # Add the index of node_new to the children of its parent. This step is essential when rewiring the tree to project the changes of the cost of the rewired node to its antecessors  
        

    def rewire(self, node_new, neighbor_index):
        for i in neighbor_index:
            node_neighbor = self.vertex[i]

            # check collision and LQR reachabilty
            if not self.utils.is_collision(node_new, node_neighbor):
                new_cost, can_rach = self.cal_LQR_new_cost(node_new, node_neighbor)

                if can_rach and node_neighbor.cost > new_cost:
                    node_neighbor.parent = node_new
                    node_neighbor.cost = new_cost
                    self.updateCosts(node_neighbor)

    def updateCosts(self,node):
        for ich in node.childrenNodeInds: 
            self.vertex[ich].cost = self.cal_LQR_new_cost(node,self.vertex[ich])[0] # FIXME since we already know that this path is safe, we only need to compute the cost 
            self.updateCosts(self.vertex[ich])
            

    def search_goal_parent(self):
        dist_list = [math.hypot(n.x - self.x_goal.x, n.y - self.x_goal.y) for n in self.vertex]
        node_index = [i for i in range(len(dist_list)) if dist_list[i] <= self.goal_len]

        if not node_index:
            return None

        if len(node_index) > 0:
            cost_list = [dist_list[i] + self.vertex[i].cost for i in node_index
                         if not self.utils.is_collision(self.vertex[i], self.x_goal)]
            return node_index[int(np.argmin(cost_list))]

        return len(self.vertex) - 1


    def generate_random_node(self, goal_sample_rate,rce = 0.5,md = 8):
        delta = self.utils.delta

        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                        np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return copy.deepcopy(self.x_goal)
        
    def find_near_neighbor(self, node_new):
        n = len(self.vertex) + 1
        r = min(self.search_radius * math.sqrt((math.log(n) / n)), self.step_len)

        dist_table = [math.hypot(nd.x - node_new.x, nd.y - node_new.y) for nd in self.vertex]
        dist_table_index = [ind for ind in range(len(dist_table)) if dist_table[ind] <= r and
                            not self.utils.is_collision(node_new, self.vertex[ind])]
        return dist_table_index

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]


    def extract_path(self, node_end):
        path = [[self.x_goal.x, self.x_goal.y]]
        node = node_end

        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

if __name__ == '__main__':
    # x_start = (18, 8)  # Starting node
    # x_goal = (37, 18)  # Goal node
    x_start = (2, 2)  # Starting node
    x_goal = (30, 24)  # Goal node

    x_start = (2.0, 2.0)  # Starting node
    x_goal = (30.0, 24.0)  # Goal node
    x_goal = (18.0, 10.0)  # Goal node

    lqr_rrt_star = LQRrrtStar(x_start=x_start, x_goal=x_goal, step_len=10,
                            goal_sample_rate=0.10, search_radius=20, 
                            iter_max=2000, solve_QP=False)
    lqr_rrt_star.planning()