import math
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

import env, plotting, utils
from LQR_planning import LQRPlanner



class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.cost = 0
        self.StateTraj = None
        self.childrenNodeInds = set([])

class LQRrrtStar:
    def __init__(self, x_start, x_goal, step_len,
                 goal_sample_rate, search_radius, iter_max, AdSamplingFlag=False, solve_QP=False):
        self.s_start = Node(x_start)
        self.s_goal = Node(x_goal)
        self.step_len = step_len
        self.goal_len = 8
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max
        self.vertex = [self.s_start]
        self.path = []

        self.env = env.Env()
        self.plotting = plotting.Plotting(x_start, x_goal)
        self.utils = utils.Utils()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        self.lqr_planner = LQRPlanner()
        self.LQR_Gain = dict()
        self.solve_QP = solve_QP

        # The adaptive sampling attributes: 
        self.Vg_leaves = []
        self.AdSamplingFlag = AdSamplingFlag
        self.adapIter = 1
        self.kde_preSamples = []
        self.kde_currSamples =[]
        self.initEliteSamples = []
        self.curr_Ldist = 0
        self.prev_Ldist = 0
        #Reaching the optimal distribution params:
        #---kde
        self.kdeOpt_flag = False
        self.kde_eliteSamples = []
        self.KDE_fitSamples = None
        self.KDE_pre_gridProbs = None
        self.kde_enabled = True
        #Elite samples and CE computation att
        self.len_frakX = 0
        self.pre_gridProbs = []
        self.SDF_optFlg = False
        self.N_qSamples = 200 
        self.rho = .3
        self.step_size = 0.3
        self.plot_pdf_kde = True

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
            
            # >>> Extend to the goal 
            if self.AdSamplingFlag: 
                # Steering to the goal region: 
                if node_new is None: 
                    continue
                g_node = self.LQR_steer(node_new, self.s_goal, exact_steering = True)
                if g_node is not None and not self.utils.is_collision(node_new, node_new):
                    self.Vg_leaves.append(g_node)
            # <<< End extend to the goal
        
        index = self.search_goal_parent()

        if index is None:
            print('No path found!')
            return None

        self.path = self.extract_path(self.vertex[index])


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


        wx, wy, _, _, = self.lqr_planner.lqr_planning(node_start.x, node_start.y, node_goal.x, node_goal.y, self.LQR_Gain, solve_QP=self.solve_QP, show_animation=show_animation)
        px, py, traj_cost = self.sample_path(wx, wy)



        if len(wx) == 1:
            return None
        node_new = Node((wx[-1], wy[-1]))
        node_new.parent = node_start
        # calculate cost of each new_node
        node_new.cost = node_start.cost + sum(abs(c) for c in traj_cost)
        node_new.StateTraj = np.array([px,py]) # Will be needed for adaptive sampling 
        return node_new

    def cal_LQR_new_cost(self, node_start, node_goal):
        wx, wy, _, can_reach = self.lqr_planner.lqr_planning(node_start.x, node_start.y, node_goal.x, node_goal.y, self.LQR_Gain, show_animation=False, solve_QP=self.solve_QP)
        px, py, traj_cost = self.sample_path(wx, wy)
        if wx is None:
            return float('inf'), False
        

        return node_start.cost + sum(abs(c) for c in traj_cost), can_reach

    def LQR_choose_parent(self, node_new, neighbor_index):
        cost = []
        for i in neighbor_index:

            # check if neighbor_node can reach node_new
            _, _, _, can_reach = self.lqr_planner.lqr_planning(self.vertex[i].x, self.vertex[i].y, node_new.x, node_new.y, self.LQR_Gain, show_animation=False, solve_QP=self.solve_QP)


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
        dist_list = [math.hypot(n.x - self.s_goal.x, n.y - self.s_goal.y) for n in self.vertex]
        node_index = [i for i in range(len(dist_list)) if dist_list[i] <= self.goal_len]

        if not node_index:
            return None

        if len(node_index) > 0:
            cost_list = [dist_list[i] + self.vertex[i].cost for i in node_index
                         if not self.utils.is_collision(self.vertex[i], self.s_goal)]
            return node_index[int(np.argmin(cost_list))]

        return len(self.vertex) - 1


    def generate_random_node(self, goal_sample_rate,rce = 0.5,md = 8):
        delta = self.utils.delta
        adap_flag=self.AdSamplingFlag
        u_rand = np.random.uniform(0, 1)
        Vg_leaves = self.Vg_leaves
        if not adap_flag or (u_rand > rce and len(Vg_leaves)==0): # Uniform sampling form the workspace: 
            if np.random.random() > goal_sample_rate:
                return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                            np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

            return copy.deepcopy(self.s_goal)
        elif len(Vg_leaves) != 0 and not self.SDF_optFlg:
            N_xSmpls = 200 
            # t_min = min([vg.curTime for vg in Vg_leaves]) # The fastest trajectory 
            # h = t_min/md 
            h = .5
            return self.CE_Sample(Vg_leaves,h,self.N_qSamples)
        elif self.SDF_optFlg: # Sampling from the optimal SDF: 
            xySmpl = self.OptSDF.sample()
            return Node(xySmpl[0][0],xySmpl[0][1])
        else: 
            if np.random.random() > goal_sample_rate:
                return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                            np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))
            return copy.deepcopy(self.s_goal)

    def CE_Sample(self,Vg_leaves,h,N_qSamples):
        """
        Exploit the samples of trajectories that reach the goal to adapt the sampling distribution towards the distribution of the rare event.
        The trajectories that reach the goal will be disceretized to extract the elite samples that will be used to adapt (optimize)
        the sampling distribution.

        :param Vg_leaves:
        :param h: The time step to discretize the trajectories
        :param N_qSamples: A threshold indicates the number of samples that are sufficient enough to be exploited (TODO (Doc): How to decide this number)
        :return: None: if the number of points of the discretized trajectories < N_qSamples, (x,y) samples from the estimated distribution
        """
        if len(Vg_leaves)>=(self.adapIter*30): #The acceptable number of trajectories to adapat upon
            frakX = []
            #Find the elite trajectoies then discretize them and use their samples as the elite samples:
            Vg_leaves_costList = [vg.cost for vg in Vg_leaves]
            if (self.adapIter + 3) > 5: 
                d_factor = 15
            elif self.adapIter > 2:
                d_factor = self.adapIter + 3
            else: 
                d_factor = self.adapIter
            
            q = self.rho               # The rho^th quantile  
            cost_rhoth_q = np.quantile(Vg_leaves_costList, q=q)
            elite_Vg_leaves = [vg for vg in Vg_leaves if vg.cost <= cost_rhoth_q]
            if len(elite_Vg_leaves) == 0:
                elite_Vg_leaves = Vg_leaves

            #XXXXXXX
            for vg in elite_Vg_leaves:
                vgcost2come = vg.cost
                #Concatnating the trajectory:
                traj2vg = np.asarray(vg.StateTraj).T
                node = vg
                while node.parent is not None:
                    node = node.parent
                    if node.cost != 0:
                        ParentTraj = node.StateTraj.T
                        traj2vg = np.concatenate((ParentTraj,traj2vg),axis=0)
                # Backtrack the path from vg to v0; extract the sample at certain increments of the time:
                tStep_init1 = int(h/self.step_size)
                tStep_init = 2
                tStep = 10
                tStep_temp = tStep_init1+3
                while tStep < len(traj2vg[:,1]):
                    pi_q_tStep = traj2vg[tStep,:]
                    elite_cddtSample = [pi_q_tStep,vgcost2come] #This tuple contains the actual sample pi_q_tStep and the CostToCome to the goal of the corresponding trajectory
                    frakX.append(elite_cddtSample)
                    tStep = tStep + tStep_temp
            if self.adapIter == 1:
                frakX.extend(self.initEliteSamples)
            # XXXXXXX
            self.len_frakX = len(frakX)
            if len(frakX) == 0:
                ok = 1
            x,y = self.CE_KDE_Sampling(frakX)
        else:
                x = None
                y = None
        if x is None or y is None: 
            delta = self.utils.delta
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                            np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))
        else: 
            return Node((x,y))
    #Density estimate, kernel density or GMM:
    def CE_KDE_Sampling(self,frakX):
        """
        Fit the elite samples to Kernel density estimate (KDE) or a GMM to generate from; and generate an (x,y) sample from the estimated
        distribution. Checks if the CE between the previous density estimate and the current one below some threshold. In the case
        of KDE the expectation similarity measure could be used instead on the CE.

        NOTE to Ahmad:
        You're using the CE with the KDE because you have the logistic probes of the samples and you use them; however, for
        Kernel based distributions the expectation similarity could be used as well. One might reformulate the CE framework
        in terms of nonparametric distributions.

        :param frakX: The elite set of samples with the corresponding trajectory cost.
        :return:
        """
        frakXarr = np.array(frakX)
        N_samples = len(frakX)
        if len(frakXarr.shape) !=2:
            ok =1
        costs_arr = frakXarr[:,1]
        elite_samplesTemp = frakXarr[:,0] #A subset of the samples that are below the elite quantile
        elite_samples = [elite_samplesTemp[i] for i in range(len(elite_samplesTemp))]
        elite_samples_arr = np.asarray(elite_samples)
        elite_costs = costs_arr

        #random point from the estimated distribution:
        if self.kde_enabled:#self.params.kde_enabled:
            kde = KernelDensity(kernel='gaussian', bandwidth=.85)
            # kde.fit(elite_samples_arr,sample_weight=w_arrNorm)
            kde.fit(elite_samples_arr)
            self.adapIter += 1
            xySample = kde.sample()

        if self.kde_enabled:#self.params.kde_enabled:
            x_gridv = np.linspace(-2, 18, 40)
            y_gridv = np.linspace(-2, 18, 40)
            Xxgrid, Xygrid = np.meshgrid(x_gridv, y_gridv)
            XYgrid_mtx = np.array([Xxgrid.ravel(), Xygrid.ravel()]).T
            #Get the probabilities
            grid_probs = np.exp(kde.score_samples(XYgrid_mtx))

            # Find the KL divergence the current samples and the previous ones:
            if self.adapIter > 2:
                KL_div = self.KLdiv(grid_probs)
                if KL_div < .1:
                    self.kdeOpt_flag = True
                    
                self.KDE_fitSamples = kde #This kde object will be used to sample form whn the optimal sampling distribution has been reached

            self.KDE_pre_gridProbs = grid_probs
            #Save the grid points with the corresponding probs, the cost, and the tree to plot them afterwards:
            # saveData(self.goal_costToCome_list, 'adapCBF_RRTstr_Cost', suffix=self.suffix, CBF_RRT_strr_obj=self,
            #          adapDist_iter=self.adapIter-1, enFlag=False)

            # saveData([self.TreeT,self.vg_minCostToCome_list], 'adapCBF_RRTstr_Tree_CDC', suffix=self.suffix, CBF_RRT_strr_obj=self,
            #          adapDist_iter=self.adapIter - 1, enFlag=False)
            # saveData([Xxgrid, Xygrid, grid_probs.reshape(Xxgrid.shape),elite_samples_arr], 'adapCBF_RRTstr_KDEgridProbs_CDC',
            #          suffix=self.suffix, CBF_RRT_strr_obj=self,
            #          adapDist_iter=self.adapIter - 1, enFlag=False)

            #Plot the distribution
            if self.plot_pdf_kde:
                # self.initialize_graphPlot()
                CS = plt.contour(Xxgrid, Xygrid, grid_probs.reshape(Xxgrid.shape))  # , norm=LogNorm(vmin=4.18, vmax=267.1))
                # plt.colorbar(CS, shrink=0.8, extend='both')
                plt.scatter(elite_samples_arr[:, 0], elite_samples_arr[:, 1])
                plt.show()

        return xySample[0][0],xySample[0][1]

    def KLdiv(self,grid_probs):
        """
        Compute the KL divergence
        :param grid_probs: The probabilities of the point in the grid of the current sampling distribution
        :return: the KL divergence
        """
        if self.kde_enabled:#self.params.kde_enabled:
            pre_grid_probs = self.KDE_pre_gridProbs
        else:
            pre_grid_probs = self.pre_gridProbs
        return -sum([pre_grid_probs[i]*np.log2(grid_probs[i]/pre_grid_probs[i]) for i in range(len(pre_grid_probs))])
    # End of Adaptive sampling 

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
        path = [[self.s_goal.x, self.s_goal.y]]
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
    x_goal = (25.0, 8.0)  # Goal node

    rrt_star = LQRrrtStar(x_start=x_start, x_goal=x_goal, step_len=10,
                            goal_sample_rate=0.10, search_radius=20, 
                            iter_max=2000,AdSamplingFlag = False, solve_QP=False)
    rrt_star.planning()