#Input parameters for SBE.py
import numpy as np

# Brillouin zone parameters
##########################################################################
Nk_in_path          = 200         # Number of kpoints in each of the two paths
rel_dist_to_Gamma   = 0.05        # relative distance (in units of 2pi/a) of both paths to Gamma
a                   = 8.308       # Lattice spacing in atomic units!! (4.395 A)
length_path_in_BZ   = np.pi/a       # 
E_dir               = np.array([0.0, 1.0])   # normalized direction of the E-field and therefore, direction of the path

# Driving field parameters
##########################################################################
E0    = 12.0  # Pulse amplitude (MV/cm)
w     = 30.0  # Pulse frequency (THz)
alpha = 48.0  # Gaussian pulse width (femtoseconds)
align = 'M'   # Direction of field polarization ('M'=gamma-M, 'K'=gamma-K)

# Time scales (all units in femtoseconds)
##########################################################################
T2    = 10.0   # Phenomenological polarization damping time 
t0    = -1000    # Start time *pulse centered @ t=0, use t0 << 0
tf    = 1500     # End time
dt    = 0.02     # Time step

# Unit conversion factors
##########################################################################
fs_conv = 41.34137335                  #(1fs    = 41.341473335 a.u.)
E_conv = 0.0001944690381               #(1MV/cm = 1.944690381*10^-4 a.u.) 
THz_conv = 0.000024188843266           #(1THz   = 2.4188843266*10^-5 a.u.)
amp_conv = 150.97488474                #(1A     = 150.97488474)
eV_conv = 0.03674932176                #(1eV    = 0.036749322176 a.u.)

# Flags for testing and features
##########################################################################
test          = False  # Set to True to output travis testing parameters
matrix_method = False  # Set to True to use old matrix method for solving

