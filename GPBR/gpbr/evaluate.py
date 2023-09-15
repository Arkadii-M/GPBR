from abc import ABC, abstractmethod
import numpy as np
import scipy

from . import boundary

class GPBREvaluate(ABC):
    def __init__(self, 
                 mfs_helper, norm_func, expr_compiler,
                 g1_exact_vals, g2_radius, g2_dirichlet_vals, eps=1e-15):
        self.expr_compiler = expr_compiler
        self.mfs_helper = mfs_helper
        self.norm_func = norm_func   
        self.g1_exact_vals=g1_exact_vals
        self.g2_radius=g2_radius
        self.g2_dirichlet_vals=g2_dirichlet_vals
        self.eps=eps
    @abstractmethod
    def feasible(self, individual):
        pass
    @abstractmethod
    def eval(self, individual):
        pass



class GPBREvaluate2D(GPBREvaluate):
    def __init__(self, thetas,**kwargs):
        super().__init__(**kwargs)
        self.g1_boundary = boundary.StarLikeCurve(self.mfs_helper.collocation)
        self.thetas=thetas

    @staticmethod
    def feasible2D(cls, individual):
        return cls.feasible(individual)
    @staticmethod
    def eval2D(cls, individual):
        return cls.eval(individual)


    def feasible(self, individual):
        func_vect = np.vectorize(self.expr_compiler(expr=individual))
        try:
            f_vals = func_vect(self.thetas)
        except Exception as e:
            return False
        
        # Save calculated values to evaluate the error or distance
        individual.eval_values = f_vals[:-1]

        # Check if all values are finite
        if not np.isfinite(f_vals).all():
            return False
        # Singular boundary
        if (abs(f_vals) < self.eps).all(): 
            return False
        
        # Interior boundary has to be inside outter boundary
        if not (abs(f_vals) < self.g2_radius).all():
            return False
        
        # Check if function is periodic
        if abs(f_vals[0] - f_vals[-1]) > self.eps:
            return False
        
        return True

    def eval(self, individual):
        # Calculate points of the boundary
        g1_approx_values = self.g1_boundary(individual.eval_values)

        # Form system
        A = self.mfs_helper.form_matrix(g1_approx_values)
        b = self.mfs_helper.form_column(g1_approx_values)

        # If there is no solution return 'invalid' fitness
        try:
            lambda_ = scipy.linalg.lstsq(A,b)[0]
        except:
            print(f'No solution for expression:{individual}')
            return 1e10,

        g2_dirichlet_approx = self.mfs_helper.uApprox(lambda_,self.mfs_helper.g2_coll,g1_approx_values)
        individual.error_values = self.g2_dirichlet_vals-g2_dirichlet_approx
        individual.last_fitness = self.norm_func(individual.error_values)

        return individual.last_fitness,


class GPBREvaluate3D(GPBREvaluate):
    def __init__(self, thetas, phis, **kwargs):
        super().__init__(**kwargs)
        self.g1_boundary = boundary.StarLikeSurface(self.mfs_helper.collocation)
        self.thetas=thetas
        self.phis=phis
    
    @staticmethod
    def feasible3D(cls, individual):
        return cls.feasible(individual)
    @staticmethod
    def eval3D(cls, individual):
        return cls.eval(individual)
    
    def feasible(self, individual):
        """
            Feasibility function for the individual. Returns True if feasible False
            otherwise.
            
            thetas has shape (NxN) and next structure:
            
            t0  t1  t1  ...  tn
            t0  t1  t1  ...  tn
            .   .   .  ...  .
            t0  t1  t1  ...  tn
            
            phis has shape (NxN) and next structure:
            
            p0  p0  p0  ...  p0
            p1  p1  p1  ...  p1
            .   .   .  ...  .
            pn  pn  pn  ...  pn
            
            
            """    
            # Evaluate function values
        func_vect = np.vectorize(self.expr_compiler.compile(expr=individual))

        try:
            f_vals = func_vect(theta=self.thetas, phi=self.phis)
        except:
            return False
        
        # Save calculated values to evaluate the error or distance
        individual.eval_values = f_vals
        
        # Check if all values are finite
        if not np.isfinite(f_vals).all():
            return False
        
        # Check for singular surface
        if (abs(f_vals) < self.eps).all():
            return False
        
        # Interior boundary has to be inside outter boundary
        if not (abs(f_vals) < self.g2_radius).all():
            return False
        
        # Check if surface is closed
        
        # r(0,phi_1) = r(0,phi_2) = ... = r(0, phi_n)
        thetas_0 = f_vals[:,0]
        if not np.isclose(thetas_0, thetas_0[0], atol=self.eps).all():
            return False
        
        # r(pi,phi_1) = r(pi,phi_2) = ... = r(pi, phi_n)
        thetas_pi = f_vals[:,-1]
        if not np.isclose(thetas_pi, thetas_pi[0], atol=self.eps).all():
            return False    
        
        # r(thetas,0) = r(thetas,2pi)
        if not np.allclose(f_vals[0], f_vals[-1], atol=self.eps):
            return False
        
        return True
    
    def eval(self, individual):
        g1_approx_values = self.g1_boundary(individual.eval_values.diagonal()) # take digagonal values (theta_0, phi_0), ..., (theta_n,phi_n)

        A =self.mfs_helper.form_matrix(g1_approx_values)
        b =self.mfs_helper.form_column(g1_approx_values)

        try:
            lambda_ = scipy.linalg.lstsq(A,b)[0]
        except:
            print(f'No solution for expression {individual}')
            return 1e10,

        g2_dirichlet_approx = self.mfs_helper.uApprox(lambda_,self.mfs_helper.g2_coll,g1_approx_values)

        individual.error_values = self.g2_dirichlet_vals-g2_dirichlet_approx
        individual.last_fitness = self.norm_func(individual.error_values)

        return individual.last_fitness,
