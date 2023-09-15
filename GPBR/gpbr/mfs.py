from abc import ABC, abstractmethod
from . import collocation as coll, boundary, boundary_condition as bc
import numpy as np



class MFS(ABC):
    def __init__(self,
                 ndim: int,
                 collocation: coll.Collocation,
                 g2Boundary:  boundary.Boundary,
                 g1DirichletCondition: bc.BoundaryCondition,
                 g2NeumanCondition: bc.BoundaryCondition):
        
        self.__ndim = ndim
        self.__n = collocation.n
        self.__collocation = collocation
        self.__g1_cond = g1DirichletCondition
        self.__g2_cond = g2NeumanCondition
        self.__g2_values = g2Boundary()

    '''
    Properties
    '''
    @property
    def n(self):
        return self.__n
    
    @property
    def collocation(self):
        return self.__collocation
    
    @property
    def g1_cond(self):
        return self.__g1_cond
    
    @property
    def g2_cond(self):
        return self.__g2_cond
    
    @property
    def g2_coll(self):
        return self.__g2_values
    
    '''
    Helper functions
    '''
    def repmat(self, X):
        '''
        Input array has next structure:
        X = [
            [x1, x2, x3,..., xn]
            [y1, y2, y3,..., yn]
            [z1, z2, z3,..., zn]]
            
        This function transfer to the next structure:
        [[  ['x1', 'x1', 'x1', 'x1', ...],
            ['x2', 'x2', 'x2', 'x2', ...],
            ['x3', 'x3', 'x3', 'x3', ...],
            .............................
            ['xn', 'xn', 'xn', 'xn', ...] ],

       [    ['y1', 'y1', 'y1', 'y1', ...],
            ['y2', 'y2', 'y2', 'y2', ...],
            ['y3', 'y3', 'y3', 'y3', ...],
            .............................
            ['yn', 'yn', 'yn', 'yn', ...] ],
            
        [   ['z1', 'z1', 'z1', 'z1', ...],
            ['z2', 'z2', 'z2', 'z2', ...],
            ['z3', 'z3', 'z3', 'z3', ...],
            .............................
            ['zn', 'zn', 'zn', 'zn', ...] ]]
        
        '''
        return np.tile(X.reshape(self.__ndim,self.n,1),self.n)
    '''
    Abstract methods ( for 2D and 3D problems)
    '''
    @abstractmethod
    def module(self, X: np.ndarray,Y: np.ndarray) -> np.ndarray:
        pass
    @abstractmethod
    def phi(self, X: np.ndarray,Y: np.ndarray) -> np.ndarray:
        pass    
    @abstractmethod
    def d_normal_phi(self, X: np.ndarray,Y: np.ndarray) -> np.ndarray:
        pass
    @abstractmethod
    def nu(self):
        pass
    
    '''
    Main functions
    '''    
    def form_column(self, g1BoundaryValues: np.ndarray) -> np.ndarray:
        '''
        Forms the right side column.
        
        g1BoundaryValues has shape (1xN)
        '''
        return np.concatenate((
            self.g1_cond(g1BoundaryValues),
            self.g2_cond(self.g2_coll)))
    
    def form_sources(self, X1, X2):
        '''
        Forms source points.
        Source points lie on pseudo boundaries.
        Note: pseudo boundary shapes can be adjusted
        '''
        X1_s=np.copy(X1)
        X2_s=np.copy(X2)

        X1_s[:,::2] = 0 ## null odd
        X2_s[:,1::2] = 0 ## null even
        return 0.5*X1_s + 2*X2_s
    
    def form_matrix(self, X1: np.ndarray):
        '''
        Forms the matrix for MFS       
        
        g1BoundaryValues ((2 or 3)xN) are values in collocation points.
        
        g1BoundaryValues = (x1_1, x1_2, ... , x1_n)
        
        Output matrix has next form:
        
        Phi(x1_1,y1)  Phi(x1_1,y2)  ...  Phi(x1_1,yn) 
        
        Phi(x1_2,y1)  Phi(x1_2,y2)  ...  Phi(x1_2,yn) 
            ...          ...        ...    ...
        Phi(x1_n,y1)  Phi(x1_n,y2)  ...  Phi(x1_n,yn) 
        ---------------------------------------------        
        DPhi(x2_1,y1) DPhi(x2_1,y2) ... DPhi(x2_1,yn) 
        
        DPhi(x2_2,y1) DPhi(x2_2,y2) ... DPhi(x2_2,yn) 
            ...          ...      ...    ...
        DPhi(x2_n,y1) DPhi(x2_n,y2) ... DPhi(x2_n,yn)        
        
        '''        
        sources = self.form_sources(X1, self.g2_coll)
        sourcesRep = np.transpose(self.repmat(sources), axes=(0,2,1))

        return np.concatenate((
                self.phi(self.repmat(X1),sourcesRep),
                self.d_normal_phi(self.repmat(self.g2_coll),sourcesRep)))
    
    def uApprox(self, lambda_: np.array, X: np.array, X1: np.array):
        '''
        Calculates approx values of function in specific points
        '''
        sources = self.form_sources(X1, self.g2_coll)
        sourcesRep = np.transpose(self.repmat(sources), axes=(0,2,1))
        return np.dot(self.phi(self.repmat(X),sourcesRep), lambda_)



class MFS3D(MFS):
    def __init__(self,
                 collocation: coll.Collocation,
                 g2Boundary:  boundary.Boundary,
                 g1DirichletCondition: bc.BoundaryCondition,
                 g2NeumanCondition: bc.BoundaryCondition):
        super().__init__(3,collocation,g2Boundary,g1DirichletCondition,g2NeumanCondition)
        self.__unit_sphere = boundary.Sphere(collocation, 1.0)
    '''
    Override
    '''
    def module(self, X: np.ndarray,Y: np.ndarray) -> np.ndarray:
        assert X.shape == Y.shape, f'X [{X.shape}] and Y [{Y.shape}] shape dimensions missmatch'
        return np.sqrt(np.sum((X-Y)**2,axis=0))
    
    def phi(self, X: np.ndarray,Y: np.ndarray) -> np.ndarray:
        assert X.shape == Y.shape, f'X [{X.shape}] and Y [{Y.shape}] shape dimensions missmatch'
        return 1.0 / (4.0 * np.pi * self.module(X,Y));
    
    def d_normal_phi(self, X: np.ndarray,Y: np.ndarray) -> np.ndarray:
        assert X.shape == Y.shape, f'X [{X.shape}] and Y [{Y.shape}] shape dimensions missmatch'
        return (np.sum( (Y-X)*self.nu(), axis = 0)) / (4.0 * np.pi * self.module(X, Y)**3)

    def nu(self): ## For the spherer
        return super().repmat(self.__unit_sphere())


# ### Helper for 2D problem




class MFS2D(MFS):
    def __init__(self,
                 collocation: coll.Collocation,
                 g2Boundary:  boundary.Boundary,
                 g1DirichletCondition: bc.BoundaryCondition,
                 g2NeumanCondition: bc.BoundaryCondition):
        super().__init__(2,collocation,g2Boundary,g1DirichletCondition,g2NeumanCondition)
        self.__unit_circle = boundary.Circle(collocation, 1.0)
    '''
    Override
    '''
    def module(self, X: np.ndarray,Y: np.ndarray) -> np.ndarray:
        assert X.shape == Y.shape, f'X [{X.shape}] and Y [{Y.shape}] shape dimensions missmatch'
        return np.sqrt(np.sum((X-Y)**2, axis = 0))
    
    def phi(self, X: np.ndarray,Y: np.ndarray) -> np.ndarray:
        assert X.shape == Y.shape, f'X [{X.shape}] and Y [{Y.shape}] shape dimensions missmatch'
        return -np.log(self.module(X, Y))/(2.0*np.pi)
    
    def d_normal_phi(self, X: np.ndarray,Y: np.ndarray) -> np.ndarray:
        assert X.shape == Y.shape, f'X [{X.shape}] and Y [{Y.shape}] shape dimensions missmatch'
        return (np.sum((Y-X)*self.nu(), axis =0))/(2.0 * np.pi * self.module(X,Y)**2)

    def nu(self): ## for the circle
        return super().repmat(self.__unit_circle())
    