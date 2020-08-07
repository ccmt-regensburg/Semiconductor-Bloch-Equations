import numpy as np
import params
import os
import efield
from efield import simple_transient
import matplotlib.pyplot as pl
from matplotlib import patches

import scipy.integrate as integrate
import data_directory

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

    transient_number    = params.transient_number

    gauge           = params.gauge
    T2              = params.T2*fs_conv
    Nk1             = params.Nk_in_path
    Nk2             = params.num_paths
    length_path_in_BZ = params.length_path_in_BZ      #


    ############ load the data from the files in the given path ###########
    old_directory   = os.getcwd()


    minTime         = 0
    maxTime         = minTime+1
    delayTimes      = np.arange(minTime, maxTime, 900)

    result          = []
    params.with_nir = False
    folder  = data_directory.change_to_data_directory(nir_t0 = nir_t0*fs_conv)
    params.with_nir = True
     
    if not os.path.exists(folder):
        print("Failing to load: ", folder)
        print("This parameter configuration has not yet been calculated")
        print()
    else:
        os.chdir(folder)
        t_ref, A_field, I_exact_E_dir_ref, I_exact_ortho_ref, I_exact_diag_E_dir, I_exact_diag_ortho, I_exact_offd_E_dir, I_exact_offd_ortho        = np.transpose(np.loadtxt('time.txt') )


    for nir_t0 in delayTimes:
        folder  = data_directory.change_to_data_directory(nir_t0 = nir_t0*fs_conv)
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
    
        ref_indices     = np.where(np.abs(np.array(t_ref)-nir_t0) < time_window)[0]
        time_indices    = np.where(np.abs(np.array(t)-nir_t0) < time_window)[0]
        I_exact_E_dir   = 1*I_exact_E_dir[time_indices] - 0*I_exact_E_dir_ref[ref_indices]
        I_exact_ortho   = 1*I_exact_ortho[time_indices] - 0*I_exact_ortho_ref[ref_indices]
        t               = t[time_indices]
    
        angles          = np.arctan((I_exact_ortho)/(((I_exact_E_dir)+0e-10j ) ) )
        pl.plot(t[0:], np.real(angles) )
        pl.show()

        I_exact_x       = (I_exact_E_dir - I_exact_ortho)/np.sqrt(2)
        I_exact_y       = (I_exact_E_dir + I_exact_ortho)/np.sqrt(2)
        
        t       *= fs_conv
        dx = t[1]-t[0]
    
        integrated_x    = np.trapz((np.diff(I_exact_x)/dx)**2, dx = t[1]-t[0])
        integrated_y    = np.trapz((np.diff(I_exact_y)/dx)**2, dx = t[1]-t[0])
        result.append(np.array([nir_t0, (integrated_x-integrated_y)/(integrated_x+integrated_y)*1e3 ] ))
        os.chdir("..")

    result      = np.array(result)
    folder      = "polarization_rotation_" + str('Nk1-{}_Nk2-{}_w{:4.2f}_E{:4.2f}_a{:4.2f}_ph{:3.2f}_T2-{:05.2f}').format(Nk1,Nk2,w/THz_conv,E0/E_conv,alpha/fs_conv,phase,T2/fs_conv)
    if not os.path.exists(folder):
        os.makedirs(folder)
    os.chdir(folder)
    np.savetxt("polarization_rotation_" + str(transient_number) + ".txt", result)
    t       = np.linspace(minTime, maxTime, 1000)*fs_conv
    fig, ax = pl.subplots()

    ax.plot(result[:,0], -result[:,1], marker="x", linestyle="", label="Polarization rotation", color="k")

    ax2     = ax.twinx()
    ax2.plot(t/fs_conv, efield.simple_A_field(t), label="A(t)", color="b")

    if params.transient_number < 0:
        ax.plot(t/fs_conv, efield.simple_transient(t), label="Transient")

    pl.legend()
    ax.grid(True)
    ax.set_xlabel("Delay time in fs")
    ax.set_ylabel("polarization of the nir-pulse in mrad")
    ax2.set_ylabel("Gauge field A(t)", color="b")
    pl.savefig("polarization_rotation_" + str(transient_number) + ".pdf")
    #pl.show()

    return 0

    

def Gaussian_envelope(t, alpha, mu):
    '''
    Function to multiply a Function f(t) before Fourier transform
    to ensure no step in time between t_final and t_final + delta
    '''
    return np.exp(-(t-mu)**2.0/(2.0*alpha)**2)

if __name__ == "__main__":
    main()
