import numpy as np
import params
#from numba import njit
import matplotlib.pyplot as pl
#from matplotlib import patches
#from scipy.integrate import ode
#from scipy.special import erf
#from sys import exit



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
    bandstruct[:,1] = np.cos(bandstruct[:,0]/length*np.pi) -  alpha
    bandstruct[:,2] = np.cos(bandstruct[:,0]/length*np.pi) + alpha

    print(*bandstruct, sep='\n')
    '''
    x_val = [x[0] for x in bandstruct]
    y_val_m = [x[1] for x in bandstruct]
    y_val_p = [x[2] for x in bandstruct]

    pl.plot(x_val,y_val_m) 
    pl.plot(x_val,y_val_p)
    pl.show()
    '''
   
    return bandstruct
