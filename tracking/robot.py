import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, Point, LineString

def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)


class BaseRobot:
    
    def __init__(self,X0,dt,ax,type='Unicycle2D'):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        '''
        
        self.type = type
        self.test_type = 'gatekeeper' # or 'cbf_qp'
        if type == 'Unicycle2D':
            try:
                from tracking.robots.unicycle2D import Unicycle2D
            except ImportError:
                from robots.unicycle2D import Unicycle2D
            self.robot = Unicycle2D(dt)
        elif type == 'DynamicUnicycle2D':
            try:
                from tracking.robots.dynamic_unicycle2D import DynamicUnicycle2D
            except ImportError:
                from robots.dynamic_unicycle2D import DynamicUnicycle2D
            self.robot = DynamicUnicycle2D(dt)
        else:
            raise ValueError("Invalid robot type")
        
        self.X = X0.reshape(-1,1)
        self.dt = dt
      
        # FOV parameters
        self.fov_angle = np.deg2rad(70)  # [rad]
        self.cam_range = 3.0  # [m]

        self.robot_radius = 0.25 # including padding
        self.max_decel = 0.5 # [m/s^2]
        self.max_ang_decel = 0.25  # [rad/s^2]

        self.U = np.array([0,0]).reshape(-1,1)
        
        # Plot handles
        self.vis_orient_len = 0.3
        # Robot's body represented as a scatter plot
        self.body = ax.scatter([],[],s=60,facecolors='b',edgecolors='b') #facecolors='none'
        # Store the unsafe points and scatter plot
        self.unsafe_points = []
        self.unsafe_points_handle = ax.scatter([],[],s=40,facecolors='r',edgecolors='r')
        # Robot's orientation axis represented as a line
        self.axis,  = ax.plot([self.X[0,0],self.X[0,0]+self.vis_orient_len*np.cos(self.X[2,0])],[self.X[1,0],self.X[1,0]+self.vis_orient_len*np.sin(self.X[2,0])], color='r')
        # Initialize FOV line handle with placeholder data
        self.fov, = ax.plot([], [], 'k--')  # Unpack the tuple returned by plot
        # Initialize FOV fill handle with placeholder data
        self.fov_fill = ax.fill([], [], 'k', alpha=0.1)[0]  # Access the first element
        self.frontier_fill = ax.fill([], [], 'b', alpha=0.1)[0]  # Access the first element
        self.safety_area_fill = ax.fill([], [], 'r', alpha=0.3)[0]  

        self.detected_obs = None
        self.detected_points = []
        self.detected_obs_patch = ax.add_patch(plt.Circle((0, 0), 0, edgecolor='black',facecolor='orange', fill=True))
        self.detected_points_scatter = ax.scatter([],[],s=10,facecolors='r',edgecolors='r') #facecolors='none'
        self.frontier = Polygon() # preserve the union of all the FOV triangles
        self.safety_area = Polygon() # preserve the union of all the safety areas
        self.positions = []  # List to store the positions for plotting

        # initialize the frontier with the initial robot location with radius 1 
        robot_position = Point(self.X[0, 0], self.X[1, 0]).buffer(1)
        self.frontier = self.frontier.union(robot_position)
    
    def f(self):
        return self.robot.f(self.X)
    
    def g(self):
        return self.robot.g(self.X)
    
    def nominal_input(self, goal, d_min=0.05):
        return self.robot.nominal_input(self.X, goal, d_min)
    
    def agent_barrier(self, obs):
        return self.robot.agent_barrier(self.X, obs, self.robot_radius)

    def step(self, U):
        # wrap step function
        self.U = U.reshape(-1,1)
        self.X = self.robot.step(self.X, self.U)
        return self.X
    
    def render_plot(self):
        x = np.array([self.X[0,0],self.X[1,0],self.X[2,0]])
        #self.body._offsets3d = ([[x[0]],[x[1]],[x[2]]])
        self.body.set_offsets([x[0], x[1]])
        if len(self.unsafe_points) > 0 and self.test_type == 'gatekeeper':
            self.unsafe_points_handle.set_offsets(np.array(self.unsafe_points))
        
        self.axis.set_ydata([self.X[1,0],self.X[1,0]+self.vis_orient_len*np.sin(self.X[2,0])])
        self.axis.set_xdata( [self.X[0,0],self.X[0,0]+self.vis_orient_len*np.cos(self.X[2,0])] )

        # Calculate FOV points
        fov_left, fov_right = self.calculate_fov_points()

        # Define the points of the FOV triangle (including robot's robot_position)
        fov_x_points = [self.X[0, 0], fov_left[0], fov_right[0], self.X[0, 0]]  # Close the loop
        fov_y_points = [self.X[1, 0], fov_left[1], fov_right[1], self.X[1, 0]]

        # Update FOV line handle
        self.fov.set_data(fov_x_points, fov_y_points)  # Update with new data

        # Update FOV fill handle
        self.fov_fill.set_xy(np.array([fov_x_points, fov_y_points]).T)  # Update the vertices of the polygon

        if not self.frontier.is_empty:
            frontier_x, frontier_y = self.frontier.exterior.xy
            self.frontier_fill.set_xy(np.array([frontier_x, frontier_y]).T)  # Update the vertices of the polygon
            #ax.fill(frontier_x, frontier_y, alpha=0.1, fc='r', ec='none')
        if not self.safety_area.is_empty:
            if self.safety_area.geom_type == 'Polygon':
                safety_x, safety_y = self.safety_area.exterior.xy
            elif self.safety_area.geom_type == 'MultiPolygon':
                safety_x = [x for poly in self.safety_area.geoms for x in poly.exterior.xy[0]]
                safety_y = [y for poly in self.safety_area.geoms for y in poly.exterior.xy[1]]
            self.safety_area_fill.set_xy(np.array([safety_x, safety_y]).T)
        if self.detected_obs is not None:
            self.detected_obs_patch.center = self.detected_obs[0], self.detected_obs[1]
            self.detected_obs_patch.set_radius(self.detected_obs[2])
        if len(self.detected_points) > 0:
            self.detected_points_scatter.set_offsets(np.array(self.detected_points))
    
    def update_frontier(self):
        fov_left, fov_right = self.calculate_fov_points()
        robot_position = (self.X[0, 0], self.X[1, 0])
        new_area = Polygon([robot_position, fov_left, fov_right])
    
        self.frontier = self.frontier.union(new_area)
        #self.frontier = self.frontier.simplify(0.1)

    def update_safety_area(self):
        theta = self.X[2, 0]  # Current heading angle in radians
        if self.type == 'Unicycle2D':
            v = self.U[0, 0]  # Linear velocity
        elif self.type == 'DynamicUnicycle2D':
            v = self.X[3, 0]
        omega = self.U[1, 0]  # Angular velocity
        
        if omega != 0:
            # Stopping times
            t_stop_linear = v / self.max_decel
            
            # Calculate the trajectory
            trajectory_points = [Point(self.X[0, 0], self.X[1, 0])]
            t = 0  # Start time
            while t <= t_stop_linear:
                v_current = max(v - self.max_decel * t, 0)
                if v_current == 0:
                    break  # Stop computing trajectory once v reaches 0
                omega_current = omega - np.sign(omega) * self.max_ang_decel * t
                if np.sign(omega_current) != np.sign(omega):  # If sign of omega changes, it has passed through 0
                    omega_current = 0  
                theta += omega_current * self.dt
                x = trajectory_points[-1].x + v_current * np.cos(theta) * self.dt
                y = trajectory_points[-1].y + v_current * np.sin(theta) * self.dt
                trajectory_points.append(Point(x, y))
                t += self.dt

            # Convert trajectory points to a LineString and buffer by robot radius
            if len(trajectory_points) >= 2:
                trajectory_line = LineString([(p.x, p.y) for p in trajectory_points])
                self.safety_area = trajectory_line.buffer(self.robot_radius)
            else:
                self.safety_area = Point(self.X[0, 0], self.X[1, 0]).buffer(self.robot_radius)
        else:    
            braking_distance = v**2 / (2 * self.max_decel)  # Braking distance
            # Straight motion
            front_center = (self.X[0, 0] + braking_distance * np.cos(theta),
                            self.X[1, 0] + braking_distance * np.sin(theta))
            self.safety_area = LineString([Point(self.X[0, 0], self.X[1, 0]), Point(front_center)]).buffer(self.robot_radius)
    
    def is_beyond_frontier(self):
        flag = not self.frontier.contains(self.safety_area)
        if flag:
            self.unsafe_points.append((self.X[0, 0], self.X[1, 0]))
        return flag
    
    def find_extreme_points(self, detected_points):
        # Convert points and robot position to numpy arrays for vectorized operations
        points = np.array(detected_points)
        robot_pos = self.X[0:2].reshape(-1)
        robot_yaw = self.X[2, 0]
        vectors_to_points = points - robot_pos
        robot_heading_vector = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
        angles = np.arctan2(vectors_to_points[:, 1], vectors_to_points[:, 0]) - np.arctan2(robot_heading_vector[1], robot_heading_vector[0])

        angles = (angles + np.pi) % (2 * np.pi) - np.pi

        leftmost_index = np.argmin(angles)
        rightmost_index = np.argmax(angles)
        
        # Extract the most left and most right points
        leftmost_point = points[leftmost_index]
        rightmost_point = points[rightmost_index]
        
        return leftmost_point, rightmost_point

    def detect_unknown_obs(self, unknown_obs, obs_margin=0.05):
        if unknown_obs is None:
            return []
        #detected_obs = []
        self.detected_points = []

        # sort unknown_obs by distance to the robot, closest first
        sorted_unknown_obs = sorted(unknown_obs, key=lambda obs: np.linalg.norm(np.array(obs[0:2]) - self.X[0:2].reshape(-1)))
        for obs in sorted_unknown_obs:
            obs_circle = Point(obs[0], obs[1]).buffer(obs[2]-obs_margin)
            intersected_area = self.frontier.intersection(obs_circle)

            # Check each point on the intersected area's exterior

            points = []
            if intersected_area.geom_type == 'Polygon':
                for point in intersected_area.exterior.coords:
                    points.append(point)
            elif intersected_area.geom_type == 'MultiPolygon':
                for poly in intersected_area.geoms:
                    for point in poly.exterior.coords:
                        points.append(point)

            for point in points:
                point_obj = Point(point)
                # Line from robot's position to the current point
                line_to_point = LineString([Point(self.X[0, 0], self.X[1, 0]), point_obj])

                # Check if the line intersects with the obstacle (excluding the endpoints)
                # only consider the front side of the obstacle
                if not line_to_point.crosses(obs_circle):
                    self.detected_points.append(point)
                
            if len(self.detected_points) > 0:
                break

        if len(self.detected_points) == 0:
            self.detected_obs = None
            return []
        leftmost_most, rightmost_point = self.find_extreme_points(self.detected_points)

        # Calculate the center and radius of the circle
        center = (leftmost_most + rightmost_point) / 2
        radius = np.linalg.norm(rightmost_point - leftmost_most) / 2

        self.detected_obs = [center[0], center[1], radius]
        return self.detected_obs
    
    
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

    import matplotlib.pyplot as plt
    import cvxpy as cp

    type = 'DynamicUnicycle2D'

    plt.ion()
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect(1)

    dt = 0.02
    tf = 20
    num_steps = int(tf/dt)
    if type == 'Unicycle2D':
        robot = BaseRobot( np.array([-1,-1,np.pi/4]).reshape(-1,1), dt, ax, type='Unicycle2D')
    elif type == 'DynamicUnicycle2D':
        robot = BaseRobot( np.array([-1,-1,np.pi/4, 0.0]).reshape(-1,1), dt, ax, type='DynamicUnicycle2D')
    obs = np.array([0.5, 0.3, 0.5]).reshape(-1,1)
    goal = np.array([2,0.5])
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
    const += [ cp.abs( u[0,0] ) <= 0.5 ]
    const += [ cp.abs( u[1,0] ) <= 0.5 ]
    cbf_controller = cp.Problem( objective, const )

    for i in range(num_steps):
        u_ref.value = robot.nominal_input( goal )
        print("u ref: ", u_ref.value.T)
        if type == 'Unicycle2D':
            alpha = 5.0 #10.0
            h, dh_dx = robot.agent_barrier( obs)
            A1.value[0,:] = dh_dx @ robot.g()
            b1.value[0,:] = dh_dx @ robot.f() + alpha * h
        elif type == 'DynamicUnicycle2D':
            alpha1 = 2.0
            alpha2 = 2.0
            h, h_dot, dh_dot_dx = robot.agent_barrier( obs)
            A1.value[0,:] = dh_dot_dx @ robot.g()
            b1.value[0,:] = dh_dot_dx @ robot.f() + (alpha1+alpha2) * h_dot + alpha1*alpha2*h

            print(dh_dot_dx)

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