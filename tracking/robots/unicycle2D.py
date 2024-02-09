import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

import sys
# at trackin_mpc/robots/unicycle2D.py, import utils/utils.py
sys.path.append('..')


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)


class Unicycle2D:
    
    def __init__(self,X0,dt,ax):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        id: robot id
        '''
        
        self.type = 'Unicycle2D'
        
        self.X = X0.reshape(-1,1)
        self.X_nominal = np.copy(self.X)
        self.dt = dt
      
        # for exp 
        self.k1 = 0.1
        self.k2 = 0.1

        # FOV parameters
        self.fov_angle = np.deg2rad(80)  # [rad]
        self.cam_range = 3  # [m]

        self.U = np.array([0,0]).reshape(-1,1)
        
        # Plot handles
        self.radii = 0.3
        # Robot's body represented as a scatter plot
        self.body = ax.scatter([],[],s=60,facecolors='b',edgecolors='b') #facecolors='none'
        # Robot's orientation axis represented as a line
        self.axis,  = ax.plot([self.X[0,0],self.X[0,0]+self.radii*np.cos(self.X[2,0])],[self.X[1,0],self.X[1,0]+self.radii*np.sin(self.X[2,0])], color='r')
        # Initialize FOV line handle with placeholder data
        self.fov, = ax.plot([], [], 'k--')  # Unpack the tuple returned by plot
        # Initialize FOV fill handle with placeholder data
        self.fov_fill = ax.fill([], [], 'k', alpha=0.1)[0]  # Access the first element
        self.render_plot()

    def f(self):
        return np.array([0,0,0]).reshape(-1,1)
    
    def g(self):
        return np.array([ [ np.cos(self.X[2,0]), 0],
                          [ np.sin(self.X[2,0]), 0],
                          [0, 1] ]) 
         
    def step(self,U): 
        self.U = U.reshape(-1,1)
        self.X = self.X + ( self.f() + self.g() @ self.U )*self.dt
        self.X[2,0] = angle_normalize(self.X[2,0])
        return self.X
    
    def render_plot(self):
        x = np.array([self.X[0,0],self.X[1,0],self.X[2,0]])
        #self.body._offsets3d = ([[x[0]],[x[1]],[x[2]]])
        self.body.set_offsets([x[0], x[1]])
        
        self.axis.set_ydata([self.X[1,0],self.X[1,0]+self.radii*np.sin(self.X[2,0])])
        self.axis.set_xdata( [self.X[0,0],self.X[0,0]+self.radii*np.cos(self.X[2,0])] )

        # Calculate FOV points
        fov_left, fov_right = self.calculate_fov_points()

        # Define the points of the FOV triangle (including robot's position)
        fov_x_points = [self.X[0, 0], fov_left[0], fov_right[0], self.X[0, 0]]  # Close the loop
        fov_y_points = [self.X[1, 0], fov_left[1], fov_right[1], self.X[1, 0]]

        # Update FOV line handle
        self.fov.set_data(fov_x_points, fov_y_points)  # Update with new data

        # Update FOV fill handle
        self.fov_fill.set_xy(np.array([fov_x_points, fov_y_points]).T)  # Update the vertices of the polygon
    
    def nominal_input(self, G, d_min = 0.05):
        G = np.copy(G.reshape(-1,1))
        k_v = 1.5 #0.5
        k_omega = 1.0 #0.5#2.5
        distance = max(np.linalg.norm( self.X[0:2,0]-G[0:2,0] ) - d_min,0)
        theta_d = np.arctan2(G[1,0]-self.X[1,0],G[0,0]-self.X[0,0])
        error_theta = angle_normalize( theta_d - self.X[2,0] )

        omega = k_omega * error_theta   
        if abs(error_theta) > np.deg2rad(80):
            v = 0
        else:
            v = k_v*( distance )*np.cos( error_theta )

        return np.array([v, omega]).reshape(-1,1)
    
    def sigma(self,s):
        return self.k2 * (np.exp(self.k1-s)-1)/(np.exp(self.k1-s)+1)
    
    def sigma_der(self,s):
        return - self.k2 * np.exp(self.k1-s)/( 1+np.exp( self.k1-s ) ) * ( 1 - self.sigma(s)/self.k2 )
    
    def agent_barrier(self, obs):
        obsX = obs[0:2]
        d_min = obs[2][0] # radius

        beta = 1.01
        theta = self.X[2,0]
        
        h = np.linalg.norm( self.X[0:2] - obsX[0:2] )**2 - beta*d_min**2   
        s = ( self.X[0:2] - obsX[0:2]).T @ np.array( [np.cos(theta),np.sin(theta)] ).reshape(-1,1)
        h = h - self.sigma(s)
        
        der_sigma = self.sigma_der(s)
        dh_dx = np.append( 2*( self.X[0:2] - obsX[0:2] ).T - der_sigma * ( np.array([ [np.cos(theta), np.sin(theta)] ]) ),  - der_sigma * ( -np.sin(theta)*( self.X[0,0]-obsX[0,0] ) + np.cos(theta)*( self.X[1,0] - obsX[1,0] ) ) , axis=1)
        return h, dh_dx
    
    def calculate_fov_points(self):
        """
        Calculate the left and right boundary points of the robot's FOV.
        """
        # Calculate left and right boundary angles
        angle_left = self.X[2,0] - self.fov_angle / 2
        angle_right = self.X[2,0] + self.fov_angle / 2

        # Calculate points at the boundary of the FOV
        fov_left = (self.X[0,0] + self.cam_range * np.cos(angle_left), self.X[1,0] + self.cam_range * np.sin(angle_left))
        fov_right = (self.X[0,0] + self.cam_range * np.cos(angle_right), self.X[1,0] + self.cam_range * np.sin(angle_right))

        return fov_left, fov_right
        
if __name__ == "__main__":

    plt.ion()
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect(1)

    dt = 0.05
    tf = 20
    num_steps = int(tf/dt)
    robot = Unicycle2D( np.array([-1,-1,np.pi/4]).reshape(-1,1), dt, ax )
    obs = np.array([0.5, 0.3, 0.1]).reshape(-1,1)
    goal = np.array([1,1])
    ax.scatter(goal[0], goal[1], c='g')
    circ = plt.Circle((obs[0,0],obs[1,0]),obs[2,0],linewidth = 1, edgecolor='k',facecolor='k')
    ax.add_patch(circ)

    num_constraints = 1
    u = cp.Variable((2,1))
    u_ref = cp.Parameter((2,1), value = np.zeros((2,1)))
    A1 = cp.Parameter((num_constraints,2), value = np.zeros((num_constraints,2)))
    b1 = cp.Parameter((num_constraints,1), value = np.zeros((num_constraints,1)))
    objective = cp.Minimize( cp.sum_squares( u - u_ref ) ) 
    const = [A1 @ u + b1 >= 0]
    alpha = 2.0 #10.0
    const += [ cp.abs( u[0,0] ) <= 1.0 ]
    const += [ cp.abs( u[1,0] ) <= 0.5 ]
    cbf_controller = cp.Problem( objective, const )

    for i in range(num_steps):
        
        h, dh_dx = robot.agent_barrier( obs)
        u_ref.value = robot.nominal_input( goal )
        A1.value[0,:] = dh_dx @ robot.g()
        b1.value[0,:] = dh_dx @ robot.f() + alpha * h
        cbf_controller.solve(solver=cp.GUROBI, reoptimize=True)
        
        if cbf_controller.status!='optimal':
            print("ERROR in QP")
            exit()
                  
        print(f"control input: {u.value.T}, h:{h}")
        robot.step(u.value)
        robot.render_plot()

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)