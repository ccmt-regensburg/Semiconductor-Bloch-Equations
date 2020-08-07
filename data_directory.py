import params
import os
import efield

def change_to_data_directory():
    transient_number    = params.transient_number 
    realistic_system    = params.realistic_system 
    with_transient      = params.with_transient   
    angle_inc_E_field   = params.angle_inc_E_field
    align               = params.align
    gauge               = params.gauge            
    BZ_type             = params.BZ_type
    semicl_model        = params.semicl_model

    fs_conv         = params.fs_conv
    THz_conv        = params.THz_conv
    E_conv          = params.E_conv
    eV_conv         = params.eV_conv

    E0              = efield.nir_E0
    alpha           = efield.nir_sigma
    w               = efield.nir_w
    phase           = efield.nir_phi
    nir_t0          = params.nir_mu*fs_conv

    T2              = params.T2*fs_conv
    temperature     = params.temperature*eV_conv          # Temperature for initial conditions

    nir_fac         = params.nir_fac
    tra_fac         = params.tra_fac
    with_nir        = params.with_nir

    if BZ_type == 'full':
        Nk1   = params.Nk1                                # Number of kpoints in b1 direction
        Nk2   = params.Nk2                                # Number of kpoints in b2 direction
        Nk    = Nk1*Nk2                                   # Total number of kpoints
        align = params.align                              # E-field alignment
        Nk_in_path  = Nk1
    elif BZ_type == 'full_for_velocity':
        Nk1   = params.Nk1_vel                            # Number of kpoints in b1 direction
        Nk2   = params.Nk2_vel                            # Number of kpoints in b2 direction
        Nk    = Nk1*Nk2                                   # Total number of kpoints
        Nk_in_path  = Nk1
        angle_inc_E_field = params.angle_inc_E_field      # Angle of driving electric field
    elif BZ_type == '2line':
        Nk_in_path = params.Nk_in_path                    # Number of kpoints in each of the two paths
        rel_dist_to_Gamma = params.rel_dist_to_Gamma      # relative distance (in units of 2pi/a) of both paths to Gamma
        length_path_in_BZ = params.length_path_in_BZ      # Length of a single path in the BZ
        angle_inc_E_field = params.angle_inc_E_field      # Angle of driving electric field
        Nk1   = params.Nk_in_path                         # for printing file names, we use Nk1 and ...
        Nk2   = params.num_paths                          # ... and Nk2 = 2
        Nk    = Nk1*Nk2                                   # Total number of k points, we have 2 paths

    print("Beginning to construct directory.")
    data_base       = "/loctmp/nim60855/generated_data/"
    if not os.path.exists(data_base):
        data_base   = "/home/maximilian/Documents/studium/generated_data/"
    os.chdir(data_base)

    next_step           = BZ_type + "/"
    if BZ_type == "full_for_velocity":
        next_step = "full/"
    
    if not os.path.exists(next_step):
        os.makedirs(next_step)
    os.chdir(next_step)

    if with_transient == False:
        next_step       = "without_transient/"
    else:
        next_step       = "transient_" + str(transient_number) + "/"
    
    if not os.path.exists(next_step):
        os.makedirs(next_step)
    os.chdir(next_step)

    if params.realistic_system:
        next_step      = "realistic_ham/"
    else:
        next_step      = "sym_ham/"
    
    if not os.path.exists(next_step):
        os.makedirs(next_step)
    os.chdir(next_step)

    if angle_inc_E_field==30:
        next_step       = "gamma-m/"
    else:
        next_step       = "gamma-k/"

    if not os.path.exists(next_step):
        os.makedirs(next_step)
    os.chdir(next_step)

    if semicl_model:
        next_step = "semicl/"
        if not os.path.exists(next_step):
            os.makedirs(next_step)
        os.chdir(next_step)

    if BZ_type == "2line":
        next_step = gauge + "/"
    elif BZ_type == "full":
        next_step = "length/"
    else:
        next_step = "velocity/"

    if not os.path.exists(next_step):
        os.makedirs(next_step)
    os.chdir(next_step)

    directory           = str('Nk1-{}_Nk2-{}_w{:4.2f}_E{:4.2f}_a{:4.2f}_ph{:3.2f}_t0-{:4.2f}_T2-{:05.2f}').format(Nk1,Nk2,w/THz_conv,E0/E_conv,alpha/fs_conv,phase,nir_t0/fs_conv,T2/fs_conv)

    if (temperature < 1e-5):
        directory       += str('_T-{:05.2f}').format(temperature*eV_conv)

    if (tra_fac != 1):
        directory       += str('_fac-{:4.2f}').format(tra_fac)

    if not with_nir:
        directory       += "_reference"


    return directory


