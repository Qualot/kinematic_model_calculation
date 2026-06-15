import numpy as np
import casadi as ca
import skrobot
from math_functions import block_tridiag, compute_normals, compute_hessians
from casadi_sdf import convert_to_casadi_sdf

class Tendon():
    def __init__(self, name, origin, insertion, via, wrap, method='casadi'):
        self.name = name
        self.origin = origin
        self.insertion = insertion
        self.via = via
        self.wrap = wrap
        self.method = method

        self.keypoints = []
        self.points = []
        self.keysegments = 0
        self.segments = 31 # m+1. >40 makes the computation too slow for newton method. 

        # m floating points
        self.m = self.segments - 1

        self.casadi_solver = None
        if self.method == 'casadi':
            # self.casadi_sdf = self._get_casadi_sdf_expr
            self.casadi_sdf = convert_to_casadi_sdf(self.wrap._sdf)
            self._init_casadi_solver()

        self.set_line_string()

    def _split_with_remainder_at_end(self, total: int, n_parts: int) -> np.ndarray:
        base = total // n_parts
        remainder = total % n_parts
        parts = np.full(n_parts, base, dtype=int)
        parts[-1] += remainder
        return parts

    def _divide_segment_with_waypoints(self, points, divisions):
        """
        Divide the points (A, C₁, ..., Cₙ, B) into segments and return the points.

        Parameters:
            points (list or array): Points to go through (E.g. [[x0,y0], [x1,y1], ..., [xn,yn]]) 
            divisions (list of int): Divisions for each sector (Length: len(points)-1) 

        Returns:
            np.ndarray: Points of all sections (Without overwrap) 
        """
        if len(divisions) != len(points) - 1:
            raise ValueError("len(divisions) should be equal to len(points) -1.")

        all_points = []

        for i in range(len(divisions)):
            start = np.array(points[i])
            end = np.array(points[i + 1])
            n = divisions[i]

            segment = np.linspace(start, end, n + 1)

            if i != 0:
                # To avoid overwrapping, starting points are omitted except for the initial sector.
                segment = segment[1:]

            all_points.append(segment)

        return np.vstack(all_points)

    def update_keypoints(self):
        self.keypoints = []
        self.keypoints.append(self.origin.worldpos())
        if not self.via is None:
            if type(self.via) is list:
                self.keysegments = len(self.via)+1
                if len(self.via) == 1:
                    self.keypoints.append(self.via.worldpos())
                else:
                    for i in self.via:
                        self.keypoints.append(i.worldpos())
            else:
                self.keypoints.append(self.via.worldpos())
                self.keysegments = 2
        else:
            self.keysegments = 1

        self.keypoints.append(self.insertion.worldpos())

        initial_points = self._divide_segment_with_waypoints(self.keypoints, self._split_with_remainder_at_end(self.segments, self.keysegments))
        self.points = initial_points

        if self.method == 'casadi':
            p_start = initial_points[0]
            p_end = initial_points[-1]
            internal_init = initial_points[1:-1]
            
            internal_opt = self.wrap_calculation_casadi(internal_init, p_start, p_end)
            self.points = np.vstack([p_start, internal_opt, p_end])
        else:
            # Newton
            self.points = initial_points
            self.points = self.wrap_calculation_newton()

    def _get_casadi_sdf_expr(self, p):
        """
        A bridge method that returns a CasADi-compatible SDF expression 
        based on the geometry information held by self.wrap.
        A cylinder type (fixed values) is used here as an example, but a dynamic 
        switch based on attributes like self.wrap.type would provide a cleaner design.
        """
        # Example: when self.wrap contains cylinder information
        radius = getattr(self.wrap, 'radius', 0.5)
        height = getattr(self.wrap, 'height', 2.0)
        
        d = ca.sqrt(p[0]**2 + p[1]**2) - radius
        return ca.fmax(d, ca.fabs(p[2]) - height / 2.0)


    def _init_casadi_solver(self, k_line=1, k_contact=1):
        """
        Solver initialization
        """

        self.casadi_sdf = convert_to_casadi_sdf(self.wrap._sdf)

        m = self.m
        X_sym = ca.MX.sym('X', 3 * m)
        p_start_sym = ca.MX.sym('p_start', 3)
        p_end_sym = ca.MX.sym('p_end', 3)
        
        E_line = 0.0
        p_first = X_sym[0:3]
        E_line += 0.5 * k_line * ca.sumsqr(p_first - p_start_sym)
        for j in range(m - 1):
            p_current = X_sym[j*3 : (j+1)*3]
            p_next    = X_sym[(j+1)*3 : (j+2)*3]
            E_line += 0.5 * k_line * ca.sumsqr(p_next - p_current)
        p_last = X_sym[(m-1)*3 : m*3]
        E_line += 0.5 * k_line * ca.sumsqr(p_end_sym - p_last)
        
        E_contact = 0.0
        for j in range(m):
            p_j = X_sym[j*3 : (j+1)*3]
            sdf_j = self.casadi_sdf(p_j)
            penetration = ca.fmin(sdf_j, 0.0)
            E_contact += 0.5 * k_contact * (penetration ** 2)

        objective = E_line + E_contact
        nlp = {'x': X_sym, 'f': objective, 'p': ca.vcat([p_start_sym, p_end_sym])}
        opts = {
            'print_time': False, 
            'ipopt.print_level': 0, 
            'ipopt.max_iter': 200, 
            'ipopt.tol': 1e-8,
            
            'ipopt.delta': 0.05,
            'ipopt.acceptable_obj_change_tol': 1e-4,
        }
        self.casadi_solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    def wrap_calculation_casadi(self, init_points, p_start, p_end):
        """
        Fast method to call the pre-compiled CasADi solver.
        """

        try:
            current_pose = self.wrap.worldcoords().T() 
        except AttributeError:
            current_pose = id(self.wrap._sdf)

        # initialize solver for the initial use or the pose change
        if (self.casadi_solver is None) or (not hasattr(self, '_prev_pose')) or (not np.allclose(self._prev_pose, current_pose)):
            self._init_casadi_solver()
            self._prev_pose = current_pose.copy() if isinstance(current_pose, np.ndarray) else current_pose
            
        p_param = np.concatenate([p_start, p_end])
        
        # solver execute
        sol = self.casadi_solver(x0=init_points.flatten(), p=p_param)
        x_opt = np.array(sol['x']).flatten()
        
        return x_opt.reshape(-1, 3)

    def wrap_calculation_newton(self):
        all_points = self.points.reshape(self.points.size)
        iter_points = all_points[3:-3] # remove x_0 and x_{m+1}
        edge_points = np.zeros(len(iter_points))
        edge_points[:3] = all_points[:3]
        edge_points[-3:] = all_points[-3:]
 
        m = len(iter_points)/3

        G_tridiag = block_tridiag(N=int(m), block_size=3)

        # conservative. no divergence
        # k_contact, alpha, iterations = 5.0, 0.01, 200

        # a bit radical. no divergence
        # k_contact, alpha, iterations = 1.0, 0.02, 100

        # current limitation? no divergence
        # k_contact, alpha, iterations = 1.0, 0.04, 50


        k_line = 1.0
        k_contact = 1.0
        d = k_line / (m**2)
        l_tol = 1e-8
        f_tol = k_line * self.get_length()/np.sqrt(m) * l_tol
        alpha = .04

        # Damped Newton method
        x = iter_points.copy()
        iterations = 50
        for i in range(iterations):
            normals = compute_normals(x.reshape(-1,3), self.wrap._sdf.__call__)
            _, distance = self.wrap._sdf.on_surface(x.reshape(-1,3))
            # distance = np.minimum(distance, 0)
            distance_repeat = np.minimum(np.repeat(distance, 3), 0)

            f_contact = -k_contact * distance_repeat * normals.reshape(normals.size)

            f = k_line*G_tridiag @ x + k_line*edge_points + f_contact

            # nabra_f_c
            # Initialize
            nabra_f_contact = np.zeros((3*int(m), 3*int(m)))
            hessian = compute_hessians(x.reshape(-1,3), self.wrap._sdf.__call__)

            # Filling blocks
            for j in range(int(m)):
                if distance[j] < 0:
                    n_j = normals[j]
                    nabra_f_contact[j*3:(j+1)*3, j*3:(j+1)*3] = -k_contact * n_j @ np.transpose(n_j)
                    - k_contact * distance[j] * hessian[j]  # Diag


            K = k_line * G_tridiag +nabra_f_contact

            x = x + alpha * np.linalg.inv(d*np.eye(len(iter_points))-K) @ f

            if np.linalg.norm(f) < f_tol:
                print('converged!')
                # iter_points = x
                break

        all_points[3:-3] = x
        return all_points.reshape(-1,3)


    def set_line_string(self):
        self.update_keypoints()
        self.line_string = skrobot.model.primitives.LineString(np.array(self.points))


    def redraw(self, viewer):
        viewer.delete(self.line_string)
        self.set_line_string()
        viewer.add(self.line_string)


    def get_length(self):
        diffs = np.diff(self.points, axis=0)             # Differences between consecutive points
        segment_lengths = np.linalg.norm(diffs, axis=1)  # Length of each segment
        return np.sum(segment_lengths)


    def update(self, points):
        self.points = points
        
    def get_points(self):
        return self.points
    
    def get_name(self):
        return self.name

    def get_start_point(self):
        return self.points[0]
        
    def get_end_point(self):
        return self.points[-1]




