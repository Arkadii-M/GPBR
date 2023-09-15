from . import collocation as coll
import numpy as np

class Boundary:  
    def __init__(self, collocation: coll.Collocation):
        
        if isinstance(collocation, coll.Collocation2D):
            self.__circle_collocation = np.array([
                    np.cos(collocation.thetas),
                    np.sin(collocation.thetas)
                ], dtype=float)
        elif isinstance(collocation, coll.Collocation3D):
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


class Circle(Boundary):
    def __init__(self, collocation: coll.Collocation, radius: float):
        Boundary.__init__(self, collocation)
        self.__radius = radius
        
    def __call__(self) -> np.ndarray:
        return super().circleCollocation * self.__radius  





class StarLikeCurve(Boundary):
    def __init__(self, collocation: coll.Collocation):
        Boundary.__init__(self, collocation)
    
    def __call__(self, r_vals: np.array) -> np.ndarray:
        return super().circleCollocation * r_vals




class Sphere(Boundary):
    def __init__(self, collocation: coll.Collocation, radius: float):
        Boundary.__init__(self, collocation)
        self.__radius = radius
    
    def mesh(self) -> np.ndarray:
        return super().sphereMesh * self.__radius
    
    def __call__(self) -> np.ndarray:
        return super().sphereCollocation * self.__radius





class StarLikeSurface(Boundary):
    def __init__(self, collocation: coll.Collocation):
        Boundary.__init__(self, collocation)
        
    def mesh(self, r_vals: np.array) -> np.ndarray:
        return super().sphereMesh * r_vals
    
    def __call__(self,r_vals: np.array) -> np.ndarray:
        return super().sphereCollocation * r_vals