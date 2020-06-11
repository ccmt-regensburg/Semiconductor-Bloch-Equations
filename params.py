# Input parameters for SBE.py
import numpy as np


class params:
# System parameters
#########################################################################
    a                   = 8.28834     # Lattice spacing in atomic units (4.395 A)
    e_fermi             = 0.0         # Fermi energy in eV
    temperature         = 0.03        # Temperature in eV

    # Model Hamiltonian parameters
    # Brillouin zone parameters
    ##########################################################################
    # Type of Brillouin zone
    # 'full' for full hexagonal BZ, '2line' for two lines with adjustable size
    BZ_type = '2line'

    # Reciprocal lattice vectors
    b1 = (2*np.pi/(a*np.sqrt(3)))*np.array([np.sqrt(3), -1])
    b2 = (4*np.pi/(a*np.sqrt(3)))*np.array([0, 1])

    # full BZ parametes
    Nk1                 = 400         # Number of kpoints in b1 direction
    Nk2                 = 20          # Number of kpoints in b2 direction (number of paths)

    # 2line BZ parameters
    Nk_in_path          = 800        # Number of kpoints in each of the two paths
    rel_dist_to_Gamma   = 0.01       # relative distance (in units of 2pi/a) of both paths to Gamma
    length_path_in_BZ   = 3*np.pi/a  # Length of path in BZ K-direction
    # length_path_in_BZ   = 4*np.pi/(np.sqrt(3)*a) # Length of path in BZ M-direction
    angle_inc_E_field   = 0        # incoming angle of the E-field in degree

    # Driving field parameters
    ##########################################################################
    align               = 'M'         # E-field direction (gamma-'K' or gamma-'M')
    E0                  = 5.00        # Pulse amplitude (MV/cm)
    w                   = 25.0        # Pulse frequency (THz)
    chirp               = -0.92        # Pulse chirp ratio (chirp = c/w) (THz)
    alpha               = 25.0        # Gaussian pulse width (femtoseconds)
    phase               = (0/1)*np.pi # Carrier envelope phase (edited by cep-scan.py)

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
    THz_conv = 0.000024188843266           #(1THz   = 2.4188843266*10^-5 a.u.)
    amp_conv = 150.97488474                #(1A     = 150.97488474)
    eV_conv = 0.03674932176                #(1eV    = 0.036749322176 a.u.)

    # Flags for testing and features
    ##########################################################################
    user_out      = True   # Set to True to get user plotting and progress output
    save_file     = True   # To save exact data
    save_full     = False  # Save full information
    test          = False  # Test plots of exact data

