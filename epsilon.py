import numpy as np
import params
from numba import njit
import matplotlib.pyplot as pl
#from matplotlib import patches
#from scipy.integrate import ode
#from scipy.special import erf
#from sys import exit
from matplotlib import rc
rc("text", usetex=False)

def initial_condition_spinorbit(y0,e_fermi,temperature,bandstruct,i_k,dynamics_type):
    e_v = bandstruct[i_k,1]
    e_c = bandstruct[i_k,2]
    if (temperature > 1e-5):
        fermi_function_e_c = 1/(np.exp((e_c-e_fermi)/temperature)+1)
        fermi_function_e_v = 1/(np.exp((e_v-e_fermi)/temperature)+1)
        y0.extend([fermi_function_e_v,0.0,0.0,fermi_function_e_c,0.0,0.0,0.0,0.0])
    else:
        if (e_c>0):
            y0_e_plus = 0
        else:
            y0_e_plus = 1

        if (e_v>0):
            y0_e_minus = 0
        else:
            y0_e_minus = 1

        y0.extend([y0_e_minus,0.0,0.0,y0_e_plus,0.0,0.0,0.0,0.0])


def epsilon(Nk_in_Path, angle_inc_E_field, paths, dk, E_dir):
    length = -paths[0,0,0]

    bandstruct = np.zeros(shape=(100,3))
    bandstruct[:,0] = np.arange(-length, length+dk, dk)
    
    k_x = (bandstruct[:,0])
    k_y = (paths[0,0,1])
   
    if params.structure_type == "zinc-blende":
        bandstruct[:,1] = -1*params.eV_conv*(np.cos((k_x+k_y)/length*np.pi) +  np.sqrt((k_x**2+k_y**2)))  # e_minus or e_v
        bandstruct[:,2] = -1*params.eV_conv*(np.cos((k_x+k_y)/length*np.pi) -  np.sqrt((k_x**2+k_y**2)))  # e_plus or e_c
    return bandstruct


def dipole():
    part1 = np.ones(100, dtype=np.complex128)
    part0 = np.zeros(100, dtype=np.complex128)
    di_x = np.concatenate((part0,part1,part1,part0)).reshape(2,2,100)
    di_y = np.zeros((2,2,100), dtype=np.complex128)
    return di_x, di_y
