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
        fermi_function = 1/(np.exp((e_c[i_k])/temperature)+1)
        y0.extend([1.0,0.0,0.0,fermi_function,0.0,0.0,0.0,0.0])
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

    
    alpha = 0.1                                     # strength of spin-orbit coupling

    length = -paths[0,0,0]
   

    bandstruct = np.zeros(shape=(100,3))
    bandstruct[:,0] = np.arange(-length, length+dk, dk)
    
    k_x2 = (bandstruct[:,0])**2
    k_y2 = (paths[0,0,1])**2
    
    bandstruct[:,1] = -1*(np.cos(bandstruct[:,0]/length*np.pi) +  np.sqrt((k_x2+k_y2)))  # e_minus or e_v
    bandstruct[:,2] = -1*(np.cos(bandstruct[:,0]/length*np.pi) -  np.sqrt((k_x2+k_y2)))  # e_plus or e_c

    return bandstruct


def dipole():
    part1 = np.ones(100, dtype=np.complex128)
    part0 = np.zeros(100, dtype=np.complex128)
    di_x = np.concatenate((part0,part1,part1,part0)).reshape(2,2,100)
    di_y = np.zeros((2,2,100), dtype=np.complex128)

    '''
    print("------- di_x ---------")
    print(*di_x,sep='\n')
        
    print("------- di_y ---------")
    print(*di_y,sep='\n')

    print(np.shape(di_x))
    '''

    return di_x, di_y
