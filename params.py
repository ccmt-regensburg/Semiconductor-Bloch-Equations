#Input parameters for SBE.py
import numpy as np

# Brillouin zone parameters
##########################################################################
Nk1   = 5                     # Monkhorst-Pack mesh points in b1 direction
Nk2   = 1                     # "                           " b2 direction
a = 1.0                       # Lattice spacing                                   
b1 = 4.0*np.pi/(np.sqrt(3)*a)*np.array([0,1])# Reciprocal lattice vectors         
b2 = 2.0*np.pi/(np.sqrt(3)*a)*np.array([np.sqrt(3),-1])

# Driving field parameters
##########################################################################
E0    = 12.0  # Pulse amplitude (MV/cm)
w     = 30.0  # Pulse frequency (THz)
alpha = 48.0  # Gaussian pulse width (femtoseconds)
align = 'M'   # Direction of field polarization ('M'=gamma-M, 'K'=gamma-K)

# Time scales (all units in femtoseconds)
##########################################################################
T2    = 1.0   # Phenomenological polarization damping time 
t0    = -1000 # Start time *pulse centered @ t=0, use t0 << 0
tf    = 1000  # End time
dt    = 0.02  # Time step

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

