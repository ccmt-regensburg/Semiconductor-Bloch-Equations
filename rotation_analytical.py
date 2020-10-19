import numpy as np
import params
import os
import efield
from efield import driving_field

import systems as sys
from hfsbe.utility import evaluate_njit_matrix as ev_mat

def main():
    return 0

def rotation(nir_t0):
    fs_conv = params.fs_conv
    eV_conv = params.eV_conv
    e_fermi = params.e_fermi*eV_conv
    w       = efield.nir_w
    alpha   = efield.nir_sigma
    gamma2  = 1/(params.T2*fs_conv)

    angle_inc_E_field   = params.angle_inc_E_field
    rel_dist_to_Gamma   = params.rel_dist_to_Gamma

    E_dir = np.array([np.cos(np.radians(angle_inc_E_field)), np.sin(np.radians(angle_inc_E_field))])
    dk, kpnts, paths = mesh(params, E_dir)

    pulse   = 0
    for path in paths:
        pulse   += kx_integral(path[:, 0], path[:, 1])

    k0          = np.array([-efield.A_transient_t0(t0) for t0 in nir_t0] )

    pulse       *= 4*(2*e_fermi)**2
    pulse       *= ((2*gamma2*(2*np.pi*w)**2)*(-k0) + ((2*np.pi*w)**2 -(2*e_fermi)**2)*efield.transient(nir_t0))/((2*np.pi*w)**2 -(2*e_fermi)**2 +gamma2**2)**2

    A   = 0
    for i, path in enumerate(paths):
        if i >= len(paths)/2:
            break
        kx_in_path  = path[:,0]
        ky_in_path  = path[:,1]
        dk          = kx_in_path[1]-kx_in_path[0]

        bandstruct  = sys.system.evaluate_energy(kx_in_path, ky_in_path)
        ecv_in_path = bandstruct[1] - bandstruct[0]
        ev_in_path  = -ecv_in_path/2
        ec_in_path  = ecv_in_path/2
        
        ec = bandstruct[1]
        k_F_ind     = np.argmin(np.abs(ec[:int(len(ec)/2)]-e_fermi) )

        if k_F_ind < len(ec)/2:
            k_F_ind = len(kx_in_path) -1 -k_F_ind

        h_deriv_x   = ev_mat(sys.h_deriv[0], kx=kx_in_path, ky=ky_in_path) 
        U   = sys.wf  (kx=kx_in_path, ky=ky_in_path)                                                   
        U_h = sys.wf_h(kx=kx_in_path, ky=ky_in_path)                                                  

        A           += 2*np.matmul(U_h[:,:,k_F_ind], np.matmul(h_deriv_x[:,:,k_F_ind], U[:,:,k_F_ind]))[1,1].real
        #pulse   += 2*np.diff(ec)[k_F_ind]/dk
    
    integral    = 0
    for path in paths:
        integral    += kx_integral_real(path[:,0], path[:,1])

    integral    *= 2*e_fermi
    integral    *= 4*k0
    integral    *= (2*np.pi*w)**2/((2*np.pi*w)**2 - (2*e_fermi)**2)
    return pulse/(A+1*integral)

def I_exact_offd_E_dir(t=[]):
    fs_conv = params.fs_conv
    w       = efield.nir_w
    alpha   = efield.nir_sigma
    nir_t0  = params.nir_mu*fs_conv
    angle_inc_E_field   = params.angle_inc_E_field
    rel_dist_to_Gamma   = params.rel_dist_to_Gamma
    eV_conv = params.eV_conv
    e_fermi = params.e_fermi*eV_conv

    if len(t) == 0:
        dt = 1/(21*2*params.w)/3
        t   = nir_t0 + np.arange(-9*alpha, 9*alpha, dt)

    E_dir = np.array([np.cos(np.radians(angle_inc_E_field)), np.sin(np.radians(angle_inc_E_field))])
    dk, kpnts, paths = mesh(params, E_dir)

    integral    = 0
    for path in paths:
        integral    += kx_integral_real(path[:,0], path[:,1])

    k0          = -efield.A_transient_t0(nir_t0)
    integral    *= 4*k0
    integral    *= 2*e_fermi
    integral    *= (2*np.pi*w)**2/((2*np.pi*w)**2 - (2*e_fermi)**2)
    integral    *= 2*rel_dist_to_Gamma/(2*np.pi)**2
    return efield.A_nir(t)*integral

def kx_integral_real(kx_in_path, ky_in_path):
    bandstruct  = sys.system.evaluate_energy(kx_in_path, ky_in_path)
    ecv_in_path = bandstruct[1] - bandstruct[0]
    ev_in_path  = -ecv_in_path/2
    ec_in_path  = ecv_in_path/2
    e_fermi     = params.e_fermi*params.eV_conv
    
    ec = bandstruct[1]
    k_F_ind     = np.argmin(np.abs(ec[:int(len(ec)/2)]-e_fermi) )
    if k_F_ind < len(ec)/2:
        k_F_ind = len(kx_in_path) -1 -k_F_ind

    di_x = sys.di_01xjit(kx=kx_in_path, ky=ky_in_path)

    return np.abs(di_x[k_F_ind] )**2

def I_exact_offd_ortho(t=[]):
    fs_conv = params.fs_conv
    w       = efield.nir_w
    nir_t0  = params.nir_mu*fs_conv
    alpha   = efield.nir_sigma
    gamma2  = 1/(params.T2*fs_conv)
    eV_conv = params.eV_conv
    e_fermi = params.e_fermi*eV_conv

    angle_inc_E_field   = params.angle_inc_E_field
    rel_dist_to_Gamma   = params.rel_dist_to_Gamma

    if len(t) == 0:
        dt = 1/(21*2*params.w)/3
        t   = nir_t0 + np.arange(-9*alpha, 9*alpha, dt)

    E_dir = np.array([np.cos(np.radians(angle_inc_E_field)), np.sin(np.radians(angle_inc_E_field))])
    dk, kpnts, paths = mesh(params, E_dir)

    pulse   = 0
    for path in paths:
        pulse   += kx_integral(path[:, 0], path[:, 1])
    pulse       *= 2*rel_dist_to_Gamma/(2*np.pi)**2

    A_thz       = [efield.A_transient_t0(time ) for time in t]
    E_nir       = efield.nir(t )
    A_nir       = efield.A_nir(t )
    h_t         = ((2*gamma2*(2*np.pi*w)**2)*A_nir + ((2*np.pi*w)**2 -(2*e_fermi)**2 -gamma2**2)*E_nir)/((2*np.pi*w)**2 -(2*e_fermi)**2 +gamma2**2)**2

    pulse       *= -4*h_t*A_thz

    return pulse

def kx_integral(kx_in_path, ky_in_path):
    bandstruct  = sys.system.evaluate_energy(kx_in_path, ky_in_path)
    ecv_in_path = bandstruct[1] - bandstruct[0]
    ev_in_path  = -ecv_in_path/2
    ec_in_path  = ecv_in_path/2
    e_fermi     = params.e_fermi*params.eV_conv
    
    ec = bandstruct[1]
    k_F_ind     = np.argmin(np.abs(ec[:int(len(ec)/2)]-e_fermi) )
    if k_F_ind < len(ec)/2:
        k_F_ind = len(kx_in_path) -1 -k_F_ind

    di_x = sys.di_01xjit(kx=kx_in_path, ky=ky_in_path)
    di_y = sys.di_01yjit(kx=kx_in_path, ky=ky_in_path)

    return (2*e_fermi)*np.imag(np.conj(di_x[k_F_ind])*di_y[k_F_ind ] )

def I_exact_diag_E_dir(t=[]):
    fs_conv = params.fs_conv
    eV_conv = params.eV_conv
    gamma2  = 1/(params.T2*fs_conv)
    nir_t0  = params.nir_mu*fs_conv
    w       = efield.nir_w
    alpha   = efield.nir_sigma

    angle_inc_E_field   = params.angle_inc_E_field
    rel_dist_to_Gamma   = params.rel_dist_to_Gamma

    if len(t) == 0:
        dt = 1/(21*2*params.w)/3
        t   = nir_t0 + np.arange(-9*alpha, 9*alpha, dt)

    E_dir = np.array([np.cos(np.radians(angle_inc_E_field)), np.sin(np.radians(angle_inc_E_field))])
    dk, kpnts, paths = mesh(params, E_dir)

    pulse       = 0
    for i, path in enumerate(paths):
        if i >= len(paths)/2:
            break
        kx_in_path  = path[:,0]
        ky_in_path  = path[:,1]
        dk          = kx_in_path[1]-kx_in_path[0]

        bandstruct  = sys.system.evaluate_energy(kx_in_path, ky_in_path)
        ecv_in_path = bandstruct[1] - bandstruct[0]
        ev_in_path  = -ecv_in_path/2
        ec_in_path  = ecv_in_path/2
        e_fermi     = params.e_fermi*eV_conv
        
        ec = bandstruct[1]
        k_F_ind     = np.argmin(np.abs(ec[:int(len(ec)/2)]-e_fermi) )

        if k_F_ind < len(ec)/2:
            k_F_ind = len(kx_in_path) -1 -k_F_ind

        h_deriv_x   = ev_mat(sys.h_deriv[0], kx=kx_in_path, ky=ky_in_path) 
        U   = sys.wf  (kx=kx_in_path, ky=ky_in_path)                                                   
        U_h = sys.wf_h(kx=kx_in_path, ky=ky_in_path)                                                  
 
        pulse       += 2*np.matmul(U_h[:,:,k_F_ind], np.matmul(h_deriv_x[:,:,k_F_ind], U[:,:,k_F_ind]))[1,1].real
        #pulse   += 2*np.diff(ec)[k_F_ind]/dk

    pulse   *= 2*rel_dist_to_Gamma/((2*np.pi)**2)
    pulse   *= -(efield.A_nir(t ) + efield.A_transient_t0(nir_t0 ) )
    return pulse


def mesh(params, E_dir):
    Nk_in_path        = params.Nk_in_path                    # Number of kpoints in each of the two paths
    rel_dist_to_Gamma = params.rel_dist_to_Gamma      # relative distance (in units of 2pi/a) of both paths to Gamma
    a                 = params.a                                      # Lattice spacing
    length_path_in_BZ = params.length_path_in_BZ      #
    num_paths         = params.num_paths

    alpha_array = np.linspace(-0.5 + (1/(2*Nk_in_path)), 0.5 - (1/(2*Nk_in_path)), num = Nk_in_path)
    vec_k_path = E_dir*length_path_in_BZ

    vec_k_ortho = 2.0*np.pi/a*rel_dist_to_Gamma*np.array([E_dir[1], -E_dir[0]])

    # Containers for the mesh, and BZ directional paths
    mesh = []
    paths = []

    # Create the kpoint mesh and the paths
#    for path_index in [-1, 1]:
    for path_index in np.linspace(-num_paths+1,num_paths-1, num = num_paths):

        print("path_index",path_index)

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


if __name__ == "__main__":
    main()
