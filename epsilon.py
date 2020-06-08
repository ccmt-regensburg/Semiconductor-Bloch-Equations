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





def epsilon(params):

    
    alpha = 0.1                                     # strength of spin-orbit coupling
    Nk_in_path = params.Nk_in_path                  # Number of kpoints in each of the two paths

    angle_inc_E_field = params.angle_inc_E_field
    E_dir = np.array([np.cos(np.radians(angle_inc_E_field)), np.sin(np.radians(angle_inc_E_field))])
    dk, kpnts, paths = mesh(params, E_dir)
    
    bandstruct = np.zeros(shape=(100,3))
   # print(*bandstruct, sep='\n')
   # return bandstruct
    length = -paths[0,0,0]

    for i in range(0,Nk_in_path):

        k=-length+dk*i
        e_k = np.cos(k/length*np.pi)
        e_minus = e_k - alpha
        e_plus = e_k + alpha

        bandstruct[i] = [k,e_minus,e_plus]

    
    '''
    x_val = [x[0] for x in bandstruct]
    y_val_m = [x[1] for x in bandstruct]
    y_val_p = [x[2] for x in bandstruct]

    pl.plot(x_val,y_val_m) 
    pl.plot(x_val,y_val_p)
    pl.show()  
    '''
   
    return bandstruct



def mesh(params, E_dir):
    Nk_in_path = params.Nk_in_path                    # Number of kpoints in each of the two paths
    rel_dist_to_Gamma = params.rel_dist_to_Gamma      # relative distance (in units of 2pi/a) of both paths to Gamma
    a = params.a                                      # Lattice spacing
    length_path_in_BZ = params.length_path_in_BZ      #

    alpha_array = np.linspace(-0.5 + (1/(2*Nk_in_path)), 0.5 - (1/(2*Nk_in_path)), num = Nk_in_path)
    vec_k_path = E_dir*length_path_in_BZ

    vec_k_ortho = 2.0*np.pi/a*rel_dist_to_Gamma*np.array([E_dir[1], -E_dir[0]])

    # Containers for the mesh, and BZ directional paths
    mesh = []
    paths = []

    # Create the kpoint mesh and the paths
    for path_index in [-1, 1]:
        # Container for a single path
        path = []
        for alpha in alpha_array:
            # Create a k-point
            kpoint = path_index*vec_k_ortho + alpha*vec_k_path

            mesh.append(kpoint)
            path.append(kpoint)

        # Append the a1'th path to the paths array
        paths.append(path)

    dk = 1.0/Nk_in_path*length_path_in_BZ

    return dk, np.array(mesh), np.array(paths)
