#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from abc import ABC, abstractmethod


# ## Define collocation

# In[2]:


class Collocation:
    def __init__(self, n: int):
        self.__n = n
    @property
    def n(self):
        return self.__n


# In[3]:


class Collocation3D(Collocation):
    def __init__(self, n: int):
        super().__init__(n)
        self.__calc_collocation()
        
    def __calc_collocation(self):
        self.__thetas = np.linspace(0,np.pi,self.n)
        self.__phis = [(2.0*np.pi*i)/self.n for i in range(self.n)]
        self.__phis_2pi = np.linspace(0.0,2.0*np.pi,self.n+1)
    @property
    def thetas(self):
        return self.__thetas
    @property
    def phis(self):
        return self.__phis
    @property
    def phis2pi(self):
        return self.__phis_2pi
    @property
    def thetas_phis(self):
        return self.__thetas,self.__phis


# In[4]:


class Collocation2D(Collocation):
    def __init__(self, n: int):
        super().__init__(n)
        self.__calc_collocation()
        
    def __calc_collocation(self):
        self.__thetas = [(2.0*np.pi*i)/self.n for i in range(self.n)]
        self._thetas_2pi = self.__thetas + [2*np.pi]
    @property
    def thetas(self):
        return self.__thetas
    @property
    def thetas2pi(self):
        return self._thetas_2pi


# ## Define boundaries

# In[5]:


class Boundary:   
    def circleBase(self,thetas: np.array) -> np.ndarray:
        return np.array([ np.cos(thetas),np.sin(thetas)])
    
    def sphereBase(self,thetas: np.array, phis: np.array) -> np.ndarray:
        sin_thetas = np.sin(thetas)
        return np.array([
            sin_thetas * np.cos(phis),
            sin_thetas * np.sin(phis),
            np.cos(thetas)
        ], dtype=float)   


# ### Two-dimensional

# In[6]:


class Circle(Boundary):
    def __init__(self, radius: float):
        self.__radius = radius
    def __call__(self,thetas: np.array) -> np.ndarray:
        return Boundary.circleBase(self,thetas)*self.__radius  


# In[7]:


class StarLikeCurve(Boundary):
    def __call__(self,r_vals: np.array, thetas: np.array) -> np.ndarray:
        return Boundary.circleBase(self,thetas)[:] * r_vals


# ### Three-dimensional

# In[8]:


class SphereBoundary(Boundary):
    def __init__(self, radius: float):
        self.__radius = radius
    def __call__(self,thetas: np.array, phis: np.array) -> np.ndarray:
        return Boundary.sphereBase(self,thetas,phis)*self.__radius


# In[9]:


class StarLikeSurface(Boundary):
    def __call__(self,r_vals: np.array, thetas: np.array, phis: np.array) -> np.ndarray:
        return Boundary.sphereBase(self,thetas,phis)[:] * r_vals


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
                 g2NeumanCondition: BoundaryCondition,
                 g2Collocation):
        self.__ndim = ndim
        self.__n = collocation.n
        self.__collocation = collocation
        self.__g1_cond = g1DirichletCondition
        self.__g2_cond = g2NeumanCondition
        self.__g2_func = g2Boundary
        self.__g2_collocation = g2Collocation

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
        return np.copy(self.__collocation.thetas_phis)
    @property
    def g1_cond(self):
        return self.__g1_cond
    @property
    def g2_cond(self):
        return self.__g2_cond
    @property
    def g2_coll(self):
        return np.copy(self.__g2_collocation)
    
    '''
    Helper functions
    '''
    def null_even(self,matrix: np.array):
        matrix[1::2] = 0
        return matrix
    def null_odd(self,matrix: np.array):
        matrix[::2] = 0
        return matrix
    def repmat(self,values: np.array,n = None):
        if n is None:
            n = self.n
        assert values.shape == (self.__ndim,n), f'repmat: shape must be ({self.__ndim},{n}) but {values.shape} was given'
        return np.tile(values.reshape(self.__ndim,n,1),n) ## TODO: check
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
    def form_column(self, g1BoundaryValues: np.array):
        return np.concatenate((
            self.g1_cond(g1BoundaryValues),
            self.g2_cond(self.g2_coll)))
    
    def form_matrix(self, g1BoundaryValues: np.ndarray):
        g1Repeat = self.repmat(g1BoundaryValues)
        g2Repeat = self.repmat(self.g2_coll)
        sourcesRepeat = self.__form_sources(np.copy(g1Repeat), np.copy(g2Repeat))
        return np.concatenate((
                self.phi(g1Repeat,sourcesRepeat),
                self.d_normal_phi(g2Repeat,sourcesRepeat)))
    
    def __form_sources(self,g1Repeat: np.array,g2Repeat: np.array):
        np.apply_along_axis(self.null_odd,1,g1Repeat) ## null odd rows   
        np.apply_along_axis(self.null_even,1,g2Repeat) ## null even rowss
        sources = g1Repeat*0.5 + g2Repeat*2
        ## TOOD: change
#         return np.array(list(map(lambda m: m.T,sources)), dtype=float) ## transpose each slice
        return np.transpose(sources, axes=(0, 2, 1))
    
    def uApprox(self, lambda_: np.array, X: np.array, g1BoundaryValues: np.array):
        n_x = X.shape[1]
        x_repeat = self.repmat(X,n_x)
        g1_repeat = self.repmat(g1BoundaryValues)
        g2_repeat = self.repmat(self.g2_coll)
#         n_x = X.shape[1]
# #         x_repeat = self.repmat(X,n_x)
# #         g1_repeat = self.repmat(g1BoundaryValues)
# #         g2_repeat = self.repmat(self.g2_coll)
        
#         x_repeat = np.tile(X.reshape(3,1,n_x),(self.n,1))
# #         g1_repeat =np.tile(g1BoundaryValues.reshape(3,n_x,1),(1,self.n))
#         g1_repeat =np.tile(g1BoundaryValues.reshape(3,1,N),(n_x,1))
#         g2_repeat = np.tile(self.g2_coll.reshape(3,1,N),(n_x,1))
        
        sources_repeat = self.__form_sources(g1_repeat,g2_repeat)
        
        phi = self.phi(x_repeat,sources_repeat)
        return np.dot(phi,lambda_)


# ### Helper for 3D problem

# In[12]:


class MFSHelper3D(MFSHelper):
    def __init__(self,
                 collocation: Collocation,
                 g2Boundary:  Boundary,
                 g1DirichletCondition: BoundaryCondition,
                 g2NeumanCondition: BoundaryCondition):
        super().__init__(3,collocation,g2Boundary,g1DirichletCondition,g2NeumanCondition,g2Boundary(collocation.thetas,collocation.phis))
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
        en = np.sum( (X-Y)*self.nu(), axis = 0) * -1
        den = 4.0 * np.pi * self.module(X, Y)**3
        return en/den
    
#     def nu(self): ## For the sphere
#         n = self.n
#         thetas, phis = self.thetas_phis
#         sin_thetas = np.sin(thetas)
#         X = np.array( sin_thetas * np.cos(phis)).reshape((n,1))
#         Y = np.array(sin_thetas * np.sin(phis)).reshape((n,1))
# #         Z = np.array(np.sign(sin_thetas) * np.cos(thetas)).reshape((n,1))
#         Z = np.array(np.cos(thetas)).reshape((n,1))
        
#         return np.array([
#             np.tile(X,(1,n)),
#             np.tile(Y,(1,n)),
#             np.tile(Z,(1,n))],dtype = float)
    def nu(self): ## For the sphere
        n = self.n
        thetas, phis = self.thetas_phis
        sin_thetas = np.sin(thetas)
        X = np.array( sin_thetas * np.cos(phis))
        Y = np.array(sin_thetas * np.sin(phis))
        Z = np.array(np.cos(thetas))
        
        return np.array([
            np.tile(X,(n,1)),
            np.tile(Y,(n,1)),
            np.tile(Z,(n,1))],dtype = float)


# ### Helper for 2D problem

# In[13]:


class MFSHelper2D(MFSHelper):
    def __init__(self,
                 collocation: Collocation,
                 g2Boundary:  Boundary,
                 g1DirichletCondition: BoundaryCondition,
                 g2NeumanCondition: BoundaryCondition):
        super().__init__(2,collocation,g2Boundary,g1DirichletCondition,g2NeumanCondition,g2Boundary(collocation.thetas))
    '''
    Override
    '''
    def module(self, X: np.ndarray,Y: np.ndarray) -> np.ndarray:
        assert X.shape == Y.shape, f'X [{X.shape}] and Y [{Y.shape}] shape dimensions missmatch'
        return np.sqrt(np.sum((X-Y)**2,axis=0))
    
    def phi(self, X: np.ndarray,Y: np.ndarray) -> np.ndarray:
        assert X.shape == Y.shape, f'X [{X.shape}] and Y [{Y.shape}] shape dimensions missmatch'
        return (-1.0/(2.0 * np.pi)) * np.log(self.module(X,Y))
    
    def d_normal_phi(self, X: np.ndarray,Y: np.ndarray) -> np.ndarray:
        assert X.shape == Y.shape, f'X [{X.shape}] and Y [{Y.shape}] shape dimensions missmatch'
        en = np.sum( (X-Y)*self.nu(), axis = 0) * -1
        den = 2.0 * np.pi *self.module(X, Y)**2
        return en/den
    
    def nu(self): ## for the circle
        n = self.n
        thetas = self.collocation.thetas
        X = np.cos(thetas)
        Y = np.sin(thetas)
        return np.array([
            np.tile(X,(n,1)),
            np.tile(Y,(n,1))],dtype = float)
