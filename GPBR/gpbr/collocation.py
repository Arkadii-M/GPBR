import numpy as np

class Collocation:
    def __init__(self, n: int):
        self.__n = n
        
    @property
    def n(self):
        return self.__n
    
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