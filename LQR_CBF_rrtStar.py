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

SHOW_ANIMATION = False

class LQRrrtStar:
    def __init__(self, x_start, x_goal, max_sampled_node_dist=10, max_rewiring_node_dist=10,
                 goal_sample_rate=0.1, rewiring_radius=20, iter_max=1000, solve_QP=False, visibility=True):
        # arguments
        self.x_start = Node(x_start)
        self.x_goal = Node(x_goal)
        self.max_sampled_node_dist = max_sampled_node_dist
        self.max_rewiring_node_dist = max_rewiring_node_dist
        self.goal_sample_rate = goal_sample_rate
        self.rewiring_radius = rewiring_radius # [m]
        self.iter_max = iter_max
        self.solve_QP = solve_QP

        # tuning parameters
        self.sample_delta = 0.5 # max dist in random sample [m]
        self.goal_len = 1 

        # initialization
        self.vertex = [self.x_start] # store all nodes in the RRT tre
        self.path = [] # final result of RRT algorithm

        # general setup
        self.env = env.Env()
        self.x_range = self.env.x_range # x range of the env
        self.y_range = self.env.y_range # y range of the env
        self.plotting = plotting.Plotting(x_start, x_goal)
        utils_ = utils.Utils() # in this code, only use is_collision()
        self.is_collision = utils_.is_collision
        # self.obs_circle = self.env.obs_circle
        lqr_cbf_planner = LQR_CBF_Planner(visibility=visibility)
        self.lqr_cbf_planning = lqr_cbf_planner.lqr_cbf_planning
        self.LQR_Gain = dict() # call by reference, so it's modified in LQRPlanner

    def planning(self):
        start_time = time.time()

        for k in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_nearest = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.LQR_steer(node_nearest, node_rand)

            # visualization
            if k % 100 == 0:
                print('rrtStar sampling iterations: ', k)
                print('rrtStar 1000 iterations sampling time: ', time.time() - start_time)
                start_time = time.time()

            if k % 500 == 0:
                print('rrtStar sampling iterations: ', k)
                self.plotting.animation_online(self.vertex, "rrtStar", True)

            # when node_new is feasible and safe
            if node_new and not self.is_collision(node_nearest, node_new):
                # the order of this function should not be changed, otherwise, neighbor might include itself
                neighbor_index = self.find_near_neighbor(node_new) 
                # add node_new to the tree
                self.vertex.append(node_new)

                # rewiring
                if neighbor_index:
                    self.LQR_choose_parent(node_new, neighbor_index)
                    self.rewire(node_new, neighbor_index)

        index = self.search_goal_parent()

        if index is None:
            print('No path found!')
            return None

        # extract path to the end_node
        self.path = self.extract_path(node_end=self.vertex[index])
        # from start to end
        self.path.reverse()
        # visualization
        self.plotting.animation(self.vertex, self.path, "rrt*, N = " + str(self.iter_max))

    def generate_random_node(self, goal_sample_rate=0.1):
        delta = self.sample_delta

        if np.random.random() > goal_sample_rate: # 0 < goal_sample_rate < random < 1.0
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                        np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return copy.deepcopy(self.x_goal) # random < goal_sample_rate
    
    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    def LQR_steer(self, node_start, node_goal, clip_max_dist=True):
        """
            - Steer from node_start to node_goal using LQR CBF planning
            - Only return a new node if the path has at least one safe path segment
            clip_max_dist: if True, clip the distance of node_goal to be at most self.max_sampled_node_dist
        """
        dist, theta = self.get_distance_and_angle(node_start, node_goal)
        if clip_max_dist: 
            dist = min(self.max_sampled_node_dist, dist)
        node_goal.x = node_start.x + dist * math.cos(theta)
        node_goal.y = node_start.y + dist * math.sin(theta)
        node_goal.yaw = theta

        # rtraj = [rx, ry, ryaw]: feasible robot trajectory
        rtraj, _, _, = self.lqr_cbf_planning(node_start, node_goal, self.LQR_Gain, solve_QP=self.solve_QP, show_animation=SHOW_ANIMATION)
        rx, ry, ryaw = rtraj
        if len(rx) == 1:
            return None
        px, py, traj_cost = self.sample_path(rx, ry)

        node_new = Node((rx[-1], ry[-1], ryaw[-1]))
        node_new.parent = node_start

        # calculate cost in terms of trajectory length
        node_new.cost = node_start.cost + sum(abs(c) for c in traj_cost)
        #node_new.StateTraj = np.array([px,py]) # Will be needed for adaptive sampling 
        return node_new


    def find_near_neighbor(self, node_new):
        """
            - for rewiring
        """
        n = len(self.vertex) + 1
        r = min(self.max_rewiring_node_dist, self.rewiring_radius * math.sqrt((math.log(n) / n)))

        dist_table = [math.hypot(nd.x - node_new.x, nd.y - node_new.y) for nd in self.vertex]
        dist_table_index = [ind for ind in range(len(dist_table)) if dist_table[ind] <= r and
                            not self.is_collision(node_new, self.vertex[ind])]
        return dist_table_index
    
    def sample_path(self, rx, ry, step=0.2):
        # smooth path
        px, py, traj_costs = [], [], []

        for i in range(len(rx) - 1):
            for t in np.arange(0.0, 1.0, step):
                px.append(t * rx[i+1] + (1.0 - t) * rx[i])
                py.append(t * ry[i+1] + (1.0 - t) * ry[i])

        dx, dy = np.diff(px), np.diff(py)
        traj_costs = [math.sqrt(idx ** 2 + idy ** 2) for (idx, idy) in zip(dx, dy)]
        return px, py, traj_costs

    def cal_LQR_new_cost(self, node_start, node_goal):
        rtraj, _, can_reach = self.lqr_cbf_planning(node_start, node_goal, self.LQR_Gain, show_animation=False, solve_QP=self.solve_QP)
        rx, ry, ryaw = rtraj
        px, py, traj_cost = self.sample_path(rx, ry)
        if rx is None:
            return float('inf'), False
        

        return node_start.cost + sum(abs(c) for c in traj_cost), can_reach

    def LQR_choose_parent(self, node_new, neighbor_index):
        """
            - before rewiring, choose the best parent for node_new
        """
        cost = []
        for i in neighbor_index:

            # check if neighbor_node can reach node_new
            _, _, can_reach = self.lqr_cbf_planning(self.vertex[i], node_new, self.LQR_Gain, show_animation=False, solve_QP=self.solve_QP)

            if can_reach and not self.is_collision(self.vertex[i], node_new):  #collision check should be updated if using CBF
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
        #print(len(neighbor_index))
        for i in neighbor_index:
            node_neighbor = self.vertex[i]

            # check collision and LQR reachabilty
            print(math.hypot(node_new.x - node_neighbor.x, node_new.y - node_neighbor.y))
            if not self.is_collision(node_new, node_neighbor):
                new_cost, can_rach = self.cal_LQR_new_cost(node_new, node_neighbor)

                if can_rach and node_neighbor.cost > new_cost:
                    node_neighbor.parent = node_new
                    node_neighbor.cost = new_cost
                    self.updateCosts(node_neighbor)

    def updateCosts(self,node):
        #print("update costs: ", node.childrenNodeInds)
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
                         if not self.is_collision(self.vertex[i], self.x_goal)]
            return node_index[int(np.argmin(cost_list))]

        return len(self.vertex) - 1

    def extract_path(self, node_end):
        path = [[self.x_goal.x, self.x_goal.y, self.x_goal.yaw]]
        node = node_end

        while node.parent is not None:
            print(node.x, node.y, node.yaw)
            path.append([node.x, node.y, node.yaw])
            node = node.parent
        path.append([node.x, node.y, node.yaw])

        return path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

if __name__ == '__main__':
    x_start = (5.0, 5.0, math.pi/2)  # Starting node (x, y, yaw)
    #x_goal = (30.0, 24.0)  # Goal node
    x_goal = (18.0, 10.0)  # Goal node
    x_goal = (10.0, 18.0)  # Goal node
    x_goal = (10.0, 3.0)  # Goal node

    # lqr_rrt_star = LQRrrtStar(x_start=x_start, x_goal=x_goal, max_sampled_node_dist=10,
    #                           max_rewiring_node_dist=10,
    #                           goal_sample_rate=0.10,
    #                           rewiring_radius=20, 
    #                           iter_max=1000,
    #                           solve_QP=False)
    lqr_rrt_star = LQRrrtStar(x_start=x_start, x_goal=x_goal, max_sampled_node_dist=1,
                              max_rewiring_node_dist=2,
                              goal_sample_rate=0.00,
                              rewiring_radius=2, 
                              iter_max=500,
                              solve_QP=False,
                              visibility=False)
    lqr_rrt_star.planning()