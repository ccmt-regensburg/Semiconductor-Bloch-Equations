import numpy as np
import params
import os
import efield
import matplotlib.pyplot as pl
from matplotlib import patches

import data_directory
import rotation_analytical

def main():
    user_out            = params.user_out
    print_J_P_I_files   = params.print_J_P_I_files   
    energy_plots        = params.energy_plots        
    dipole_plots        = params.dipole_plots        
    test                = params.test                
    matrix_method       = params.matrix_method       
    emission_wavep      = params.emission_wavep      
    Bcurv_in_B_dynamics = params.Bcurv_in_B_dynamics 
    store_all_timesteps = params.store_all_timesteps 
    fitted_pulse        = params.fitted_pulse        
    KK_emission         = params.KK_emission         
    normalize_emission  = params.normalize_emission  
    normalize_f_valence = params.normalize_f_valence 

    a = params.a                                      # Lattice spacing
    b1 = params.b1                                        # Reciprocal lattice vectors
    b2 = params.b2

    BZ_type             = params.BZ_type                          # Type of Brillouin zone to construct
    angle_inc_E_field   = params.angle_inc_E_field      # Angle of driving electric field
    B0                  = params.B0

    fs_conv         = params.fs_conv
    THz_conv        = params.THz_conv
    E_conv          = params.E_conv

    E0              = efield.nir_E0
    alpha           = efield.nir_sigma
    nir_t0          = efield.nir_mu
    w               = efield.nir_w
    phase           = efield.nir_phi
    dt = 1/(21*2*w)/3

    transient_number    = params.transient_number

    gauge           = params.gauge
    T2              = params.T2*fs_conv
    Nk1             = params.Nk_in_path
    Nk2             = params.num_paths
    length_path_in_BZ = params.length_path_in_BZ      #


    ############ load the data from the files in the given path ###########
    old_directory   = os.getcwd()


    minTime         = -900
    maxTime         = 900
    delayTimes      = np.arange(minTime, maxTime+1, 25)

    result          = []
    result_a        = []
    params.with_nir = False
    nir_mu          = 0
    folder  = data_directory.change_to_data_directory()
    params.with_nir = True
     
    if not os.path.exists(folder):
        ref_data    = False
        print("Failing to load: ", folder)
        print("This parameter configuration has not yet been calculated")
        print()
    else:
        ref_data    = False
        os.chdir(folder)
        t_ref, A_field, I_exact_E_dir_ref, I_exact_ortho_ref, I_exact_diag_E_dir, I_exact_diag_ortho, I_exact_offd_E_dir, I_exact_offd_ortho        = np.transpose(np.loadtxt('time.txt') )


    for nir_t0 in delayTimes:
        params.nir_mu   = nir_t0
        folder  = data_directory.change_to_data_directory()
        print("Loading data out of " + folder)
    
        if not os.path.exists(folder):
            print("Failing to load: ", folder)
            print("This parameter configuration has not yet been calculated")
            print()
            continue
        else:
            os.chdir(folder)
    
        t, A_field, I_exact_E_dir, I_exact_ortho, I_exact_diag_E_dir, I_exact_diag_ortho, I_exact_offd_E_dir, I_exact_offd_ortho        = np.transpose(np.loadtxt('time.txt') )

        #pl.plot(t, I_exact_E_dir)
        #pl.plot(t, I_exact_ortho)

        time_window     = 9*alpha/fs_conv
        time_indices    = np.where(np.abs(np.array(t)-nir_t0) < time_window)[0]
        I_exact_E_dir   = I_exact_E_dir[time_indices]
        I_exact_ortho   = I_exact_ortho[time_indices]
        I_exact_offd_ortho   = I_exact_offd_ortho[time_indices]
        t               = t[time_indices]
    
        I_exact_x       = (I_exact_E_dir + I_exact_ortho)/np.sqrt(2)
        I_exact_y       = (I_exact_E_dir - I_exact_ortho)/np.sqrt(2)

        t               *= fs_conv
        dx = t[1]-t[0]

        I_offd_ortho_a  = rotation_analytical.I_exact_offd_ortho(t)
        I_exact_x_a     = (I_exact_E_dir + I_offd_ortho_a)/np.sqrt(2)
        I_exact_y_a     = (I_exact_E_dir - I_offd_ortho_a)/np.sqrt(2)
    
        integrated_x    = np.trapz((np.diff(I_exact_x)/dx)**2, dx = dx)
        integrated_y    = np.trapz((np.diff(I_exact_y)/dx)**2, dx = dx)

        integrated_x_a  = np.trapz((np.diff(I_exact_x_a)/dx)**2, dx = dx)
        integrated_y_a  = np.trapz((np.diff(I_exact_y_a)/dx)**2, dx = dx)

        result.append(np.array([nir_t0, (integrated_x-integrated_y)/(integrated_x+integrated_y) ] ))
        result_a.append(np.array([nir_t0, (integrated_x_a-integrated_y_a)/(integrated_x_a+integrated_y_a) ] ))
        os.chdir("..")

    result      = np.array(result)
    result_a    = np.array(result_a)
    folder      = "polarization_rotation_" + str('Nk1-{}_Nk2-{}_w{:4.2f}_E{:4.2f}_a{:4.2f}_ph{:3.2f}_T2-{:05.2f}').format(Nk1,Nk2,w/THz_conv,E0/E_conv,alpha/fs_conv,phase,T2/fs_conv)
    if not os.path.exists(folder):
        os.makedirs(folder)
    os.chdir(folder)
    np.savetxt("polarization_rotation_" + str(transient_number) + ".txt", result)
    t       = np.linspace(minTime, maxTime, 1000)*fs_conv
    fig, ax = pl.subplots()

    #ax.plot(result[:,0], result[:,1]*1e6, marker="x", linestyle="", label="numerical", color="k")
    ax.plot(result_a[:,0], result_a[:,1]*1e6, marker=".", linestyle="", label="analytical", color="b")

    pl.legend()
    ax.grid(True)
    ax.set_xlabel("Delay time in fs")
    ax.set_ylabel("polarization of the nir-pulse in mikro rad")
    pl.savefig("polarization_rotation_" + str(transient_number) + ".pdf")
    pl.show()

    return 0

    

def Gaussian_envelope(t, alpha, mu):
    '''
    Function to multiply a Function f(t) before Fourier transform
    to ensure no step in time between t_final and t_final + delta
    '''
    return np.exp(-(t-mu)**2.0/(2.0*alpha)**2)

if __name__ == "__main__":
    main()
