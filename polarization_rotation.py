import numpy as np
import params
import os
import efield
import matplotlib.pyplot as pl
from matplotlib import patches

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
    substract_offset    = params.substract_offset    
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

    gauge           = params.gauge
    T2              = params.T2*fs_conv
    Nk1             = params.Nk_in_path
    Nk2             = params.num_paths
    length_path_in_BZ = params.length_path_in_BZ      #


    ############ load the data from the files in the given path ###########
    old_directory   = os.getcwd()

    data_base       = "/loctmp/nim60855/generated_data/"
    if not os.path.exists(data_base):
        data_base   = "/home/maximilian/Documents/studium/generated_data/"
    os.chdir(data_base)

    if params.realistic_system:
        directory      = "realistic_ham/"
    else:
        directory      = "sym_ham/"

    if params.with_transient:
        directory      += "with_transient/"
    else:
        directory      += "without_transient/"

    directory           += gauge + "/"

    directory           += str('Nk1-{}_Nk2-{}_w{:4.2f}_E{:4.2f}_a{:4.2f}_ph{:3.2f}_t0-{:4.2f}_T2-{:05.2f}').format(Nk1,Nk2,w/THz_conv,E0/E_conv,alpha/fs_conv,phase,nir_t0/fs_conv,T2/fs_conv)
    print("Loading data out of " + directory)

    if not os.path.exists(directory):
        print("Failing to load: ", directory)
        print("This parameter configuration has not yet been calculated")
        print()
        return 0
    else:
        os.chdir(directory)


    t, A_field, I_exact_E_dir, I_exact_ortho, I_exact_diag_E_dir, I_exact_diag_ortho, I_exact_offd_E_dir, I_exact_offd_ortho        = np.transpose(np.loadtxt('time.txt') )
    freq, Int_exact_E_dir, Int_exact_ortho, Int_exact_diag_E_dir, Int_exact_diag_ortho, Int_exact_offd_E_dir, Int_exact_offd_ortho  = np.transpose(np.loadtxt('frequency.txt') )

    Int_exact_E_dir = np.around(Int_exact_E_dir, decimals=12)
    Int_exact_ortho = np.around(Int_exact_ortho, decimals=12)
    t       *= fs_conv
    freq    *= w
    w_min       = np.argmin(np.abs(freq/w - 0.5 ) )
    integrated_super    = 4*np.trapz(np.sqrt(Int_exact_E_dir[w_min:]*Int_exact_ortho[w_min:]), dx = freq[1]-freq[0])
    integrated_E_dir    = 2*np.trapz(Int_exact_E_dir[w_min:], dx = freq[1]-freq[0])
    print("Power of the parallel component:", integrated_E_dir)

    integrated_ortho    = 2*np.trapz(Int_exact_ortho[w_min:], dx = freq[1]-freq[0])
    print("Power of the orthogonal component:", integrated_ortho)
    print("Resulting angle in mrad:", integrated_super/(integrated_ortho+integrated_E_dir)*1e3 )
    print()


if __name__ == "__main__":
    main()
