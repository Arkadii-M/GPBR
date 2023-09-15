import numpy as np

class BoundaryCondition:
    def __init__(self, func):
        self.__func = func
        
    def __call__(self, X: np.array):
        return self.__func(X)