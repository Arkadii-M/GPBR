#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from abc import ABC, abstractmethod
from functools import lru_cache, cached_property

from typing import List


# ## Define collocation

# In[2]:


class Collocation:
    def __init__(self, n: int):
        self.__n = n
        
    @property
    def n(self):
        return self.__n


# In[3]:


class Collocation2D(Collocation):
    def __init__(self, n: int):
        Collocation.__init__(self, n)
        self.__thetas = [(2.0*np.pi*i)/self.n for i in range(self.n)] # [0,2pi)
        self.__thetas_closed = self.__thetas + [2.0*np.pi] # [0,2pi]
        
    @property
    def thetas(self):
        return self.__thetas
    
    @property
    def thetas_closed(self):
        return self.__thetas_closed


# In[4]:


class Collocation3D(Collocation):
    def __init__(self, n: int):
        Collocation.__init__(self, n)
        self.__thetas = np.linspace(0,np.pi,self.n) # [0,pi]
        self.__phis = [(2.0*np.pi*i)/self.n for i in range(self.n)] # [0,2pi)
        self.__phis_closed = self.__phis + [2.0*np.pi] # [0,2pi]
        
    @property
    def thetas(self):
        return self.__thetas
    
    @property
    def phis(self):
        return self.__phis
    
    @property
    def phis_closed(self):
        return self.__phis_closed
    
#     @property
#     def mesh(self):
#         return np.meshgrid(self.__thetas,self.__phis)
    
    @property
    def mesh_closed(self):
        return np.meshgrid(self.__thetas,self.__phis_closed)


# ## Define boundaries

# In[5]:


class Boundary:  
    def __init__(self, collocation: Collocation):
        
        if isinstance(collocation, Collocation2D):
            self.__circle_collocation = np.array([
                    np.cos(collocation.thetas),
                    np.sin(collocation.thetas)
                ], dtype=float)
        elif isinstance(collocation, Collocation3D):
            self.__sphere_collocation = np.array([
                    np.sin(collocation.thetas)*np.cos(collocation.phis),
                    np.sin(collocation.thetas)*np.sin(collocation.phis),
                    np.cos(collocation.thetas)
                ], dtype=float)
            
            self.__sphere_mesh = np.array([
                np.outer(np.cos(collocation.phis), np.sin(collocation.thetas)),
                np.outer(np.sin(collocation.phis), np.sin(collocation.thetas)),
                np.outer(np.ones(np.size(collocation.phis)), np.cos(collocation.thetas)),
            ], dtype=float)
        
    @property
    def circleCollocation(self):
        return self.__circle_collocation
    
    @property
    def sphereCollocation(self):
        return self.__sphere_collocation
        
    @property
    def sphereMesh(self):
        return self.__sphere_mesh


# ### Two-dimensional

# In[6]:


class Circle(Boundary):
    def __init__(self, collocation: Collocation, radius: float):
        Boundary.__init__(self, collocation)
        self.__radius = radius
        
    def __call__(self) -> np.ndarray:
        return super().circleCollocation * self.__radius  


# In[7]:


class StarLikeCurve(Boundary):
    def __init__(self, collocation: Collocation):
        Boundary.__init__(self, collocation)
    
    def __call__(self, r_vals: np.array) -> np.ndarray:
        return super().circleCollocation * r_vals


# ### Three-dimensional

# In[8]:


class Sphere(Boundary):
    def __init__(self, collocation: Collocation, radius: float):
        Boundary.__init__(self, collocation)
        self.__radius = radius
    
    def mesh(self) -> np.ndarray:
        return super().sphereMesh * self.__radius
    
    def __call__(self) -> np.ndarray:
        return super().sphereCollocation * self.__radius


# In[9]:


class StarLikeSurface(Boundary):
    def __init__(self, collocation: Collocation):
        Boundary.__init__(self, collocation)
        
    def mesh(self, r_vals: np.array) -> np.ndarray:
        return super().sphereMesh * r_vals
    
    def __call__(self,r_vals: np.array) -> np.ndarray:
        return super().sphereCollocation * r_vals


# ## Boundary condition

# In[10]:


class BoundaryCondition:
    def __init__(self, func):
        self.__func = func
        
    def __call__(self,X: np.array):
        return self.__func(X)


# ## Method of fundamental solutions (MFS) helper abstract

# In[11]:


class MFSHelper(ABC):
    def __init__(self,
                 ndim: int,
                 collocation: Collocation,
                 g2Boundary:  Boundary,
                 g1DirichletCondition: BoundaryCondition,
                 g2NeumanCondition: BoundaryCondition):
        
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
    def thetas_phis(self):
        return np.copy(self.collocation.thetas), np.copy(self.collocation.phis)
    @property
    def g1_cond(self):
        return self.__g1_cond
    @property
    def g2_cond(self):
        return self.__g2_cond
    @property
    def g2_coll(self):
        return np.copy(self.__g2_values)
    
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
    Abstract function ( for 2D and 3D problems)
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
        X1[:,::2] = 0 ## null odd
        X2[:,1::2] = 0 ## null even
        return 0.5*X1 + 2*X2
    
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
        sources = self.form_sources(np.copy(X1), self.g2_coll)
        sourcesRep = np.transpose(self.repmat(sources), axes=(0,2,1))

        return np.concatenate((
                self.phi(self.repmat(X1),sourcesRep),
                self.d_normal_phi(self.repmat(self.g2_coll),sourcesRep)))
    
    def uApprox(self, lambda_: np.array, X: np.array, X1: np.array):
        '''
        Calculates approx values of function in specific points
        '''
        sources = self.form_sources(np.copy(X1), self.g2_coll)
        sourcesRep = np.transpose(self.repmat(sources), axes=(0,2,1))
        return np.dot(self.phi(self.repmat(X),sourcesRep), lambda_)


# ### Helper for 3D problem

# In[12]:


class MFSHelper3D(MFSHelper):
    def __init__(self,
                 collocation: Collocation,
                 g2Boundary:  Boundary,
                 g1DirichletCondition: BoundaryCondition,
                 g2NeumanCondition: BoundaryCondition):
        MFSHelper.__init__(self, 3,collocation,g2Boundary,g1DirichletCondition,g2NeumanCondition)
        self.__unit_sphere = Sphere(collocation, 1.0)
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
        return super().repmat(np.copy(self.__unit_sphere()))


# ### Helper for 2D problem

# In[13]:


class MFSHelper2D(MFSHelper):
    def __init__(self,
                 collocation: Collocation,
                 g2Boundary:  Boundary,
                 g1DirichletCondition: BoundaryCondition,
                 g2NeumanCondition: BoundaryCondition):
        MFSHelper.__init__(self, 2,collocation,g2Boundary,g1DirichletCondition,g2NeumanCondition)
        self.__unit_circle = Circle(collocation, 1.0)
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
        return super().repmat(np.copy(self.__unit_circle()))
    