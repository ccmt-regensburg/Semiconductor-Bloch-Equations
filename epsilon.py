import numpy as np
import params
#from numba import njit
import matplotlib.pyplot as pl
#from matplotlib import patches
#from scipy.integrate import ode
#from scipy.special import erf
#from sys import exit

def epsilon(params):

    angle_inc_E_field = params.angle_inc_E_field
    E_dir = np.array([np.cos(np.radians(angle_inc_E_field)), np.sin(np.radians(angle_inc_E_field))])
    dk, kpnts, paths = mesh(params, E_dir)

    epsilon = []
    length = -paths[0,0,0]
    for i in range(0,100):

        k=paths[0,i,0]
        epsk = [k,np.cos(k/length*np.pi)]
        epsilon.append(epsk)


    print(*epsilon, sep='\n')

    x_val = [x[0] for x in epsilon]
    y_val = [x[1] for x in epsilon]

    pl.plot(x_val,y_val)
    pl.show()

    return epsilon 



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

epsilon(params)
