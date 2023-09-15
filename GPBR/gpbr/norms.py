import numpy as np
import scipy
from scipy import integrate


def MSE_norm(values):
    return np.sum(values**2)/len(values) ## MSE error


def L2_norm_2d(values):
    return np.sqrt((2*np.pi/len(values))*np.sum(values**2)) ## L2 norm


def L2_norm_3d(f_vals, r_vals, thetas, phis):
    # thetas in [0,pi]
    # phis in [0,2pi]
    return np.sqrt(integrate.trapezoid(integrate.trapezoid(f_vals**2 * r_vals**2,phis)*np.sin(thetas),thetas))