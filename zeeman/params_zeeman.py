# Input parameters for SBE.py
import numpy as np


class params:
# System parameters
#########################################################################
    a                   = 8.28834     # Lattice spacing in atomic units (4.395 A)
    # a = 8.308
    e_fermi             = 0.2         # Fermi energy in eV
    temperature         = 0.03        # Temperature in eV

    # Model Hamiltonian parameters
    # Brillouin zone parameters
    ##########################################################################
    # Type of Brillouin zone
    # 'full' for full hexagonal BZ, '2line' for two lines with adjustable size
    BZ_type = 'full'
    gauge = 'velocity_extra'

    # Reciprocal lattice vectors
    b1 = (2*np.pi/(a*np.sqrt(3)))*np.array([np.sqrt(3), -1])
    b2 = (4*np.pi/(a*np.sqrt(3)))*np.array([0, 1])

    # full BZ parametes
    Nk1                 =  4          # Number of kpoints in b1 direction
    Nk2                 =  4          # Number of kpoints in b2 direction (number of paths)

    # 2line BZ parameters
    Nk_in_path          = 800         # Number of kpoints in each of the two paths
    rel_dist_to_Gamma   = 0.05        # relative distance (in units of 2pi/a) of both paths to Gamma
    length_path_in_BZ   = 3*np.pi/a   # Length of path in BZ K-direction
    # length_path_in_BZ   = 5*np.pi/(np.sqrt(3)*a) # Length of path in BZ M-direction
    angle_inc_E_field   = 30          # incoming angle of the E-field in degree

    # Driving field parameters
    ##########################################################################
    align               = 'M'         # E-field direction (gamma-'K' or gamma-'M')
    E0                  = 5.00        # Pulse amplitude (MV/cm)
    w                   = 25.0        # Pulse frequency (THz)
    chirp               = 0.0         # Pulse chirp ratio (chirp = c/w) (THz)
    alpha               = 25.0        # Gaussian pulse width (femtoseconds)
    phase               = (0/1)*np.pi # Carrier envelope phase (edited by cep-scan.py)

    B0                  = 10          # Magnetic field Amplitude (T)
    incident_angle      = 45          # Theta angle to the z-axis
    mu_x                = 1           # Magnetic dipole moment in x direction (mu_b)
    mu_y                = 1           # Magnetic dipole moment in y direction (mu_b)
    mu_z                = 1           # Magnetic dipole moment in z direction (mu_b)

    # Time scales (all units in femtoseconds)
    ##########################################################################
    T1    = 1E3    # Phenomenological diagonal damping time
    T2    = 1      # Phenomenological polarization damping time
    t0    = -1000  # Start time *pulse centered @ t=0, use t0 << 0
    tf    = 1000   # End time
    dt    = 0.01    # Time step

    # Unit conversion factors
    ##########################################################################
    fs_conv = 41.34137335                  #(1fs    = 41.341473335 a.u.)
    E_conv = 0.0001944690381               #(1MV/cm = 1.944690381*10^-4 a.u.)
    B_conv = 4.25531e-6                    #(1T     = 4.255e-6 a.u.)
    THz_conv = 0.000024188843266           #(1THz   = 2.4188843266*10^-5 a.u.)
    amp_conv = 150.97488474                #(1A     = 150.97488474)
    eV_conv = 0.03674932176                #(1eV    = 0.036749322176 a.u.)
    muB_conv = 0.5                         #(1mu_b    = 0.5 a.u.)

    # Flags for testing and features
    ##########################################################################
    user_out      = True   # Set to True to get user plotting and progress output
    calc_exact    = True   # Calculate exact emission (Careful takes long!)
    normal_plots  = False  # Standard plots of P, J, I and w-dependency
    polar_plots   = False  # Higher harmonic polarization rotation
    save_file     = True   # Save all data files
    save_full     = False  # Save full information

