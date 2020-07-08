#Input parameters for SBE.py
import numpy as np

# System parameters
#########################################################################
#a                   = 10.259       # Lattice spacing in atomic units (4.395 A)
a                   = 8.308
# Galium Selenide  lattice spacing = 5.429 Angstrom = 10.259 a.u.
# Galium Arsenic   lattice spacing = 5.653 angstrom = 10.683 a.u.
# Bismuth Teluride lattice spacing = 4.395 angstrom = 8.308
e_fermi             = 0       # Fermi energy in eV
temperature         = 0.0       # Temperature in eV

# Model Hamiltonian parameters
C0                  = 0          # Dirac point position
C2                  = 0           # k^2 coefficient
A                   = 0.1974      # Fermi velocity
R                   = 5.53        # k^3 coefficient
k_cut               = 0.05       # Model hamiltonian cutoff

# Brillouin zone parameters
##########################################################################
# Type of Brillouin zone
# 'full' for full hexagonal BZ, '2line' for two lines with adjustable size
# 'full_for_velocity' for Monkhorst-Pack mesh for velocity gauge
BZ_type = '2line'

# Reciprocal lattice vectors
b1 = (2*np.pi/(a*np.sqrt(3)))*np.array([np.sqrt(3),-1])
b2 = (4*np.pi/(a*np.sqrt(3)))*np.array([0,1])

# full_for_velocity BZ parametes
Nk1_vel             = 10          # Number of kpoints in b1 direction
Nk2_vel             = 10          # Number of kpoints in b2 direction
angle_inc_E_field   = 0           # incoming angle of the E-field in degree

# full BZ parametes
Nk1                 = 10        # Number of kpoints in b1 direction
Nk2                 = 2         # Number of kpoints in b2 direction (number of paths)

# 2line BZ parameters
Nk_in_path          = 100         # Number of kpoints in each of the two paths
rel_dist_to_Gamma   = 0.05        # relative distance (in units of 2pi/a) of both paths to Gamma
length_path_in_BZ   = 2*np.pi/a   # Length of path in BZ
angle_inc_E_field   = 0           # incoming angle of the E-field in degree

# Gauge
gauge               = 'length'
#gauge               = 'velocity'    # 'length': use length gauge with gradient_k present
                                  # 'velocity': use velocity gauge with absent gradient_k

# Driving field parameters
##########################################################################
align               = 'K'          # E-field direction (gamma-'K' or gamma-'M'), 
                                   # or angle (30 for 30 degrees, only works with velocity gauge) 
E0                  = 5.0          # Pulse amplitude (MV/cm)
B0                  = 0           # B-Field strength (T)
w                   = 25.0         # Pulse frequency (THz)
chirp               = 0.0          # Pulse chirp ratio (chirp = c/w) (THz)
alpha               = 25.0         # Gaussian pulse width (femtoseconds)
phase               = (0/5)*np.pi  # Carrier envelope phase (edited by cep-scan.py)

# scaling of the dipole
scale_dipole_eq_mot = 1
scale_dipole_emiss  = 1

# Time scales (all units in femtoseconds)
##########################################################################
T1    = 1E3  # Phenomenological damping time for diagonal occupations
T2    = 1       # Phenomenological damping time for off-diagonal polarizations
t0    = -1000 # Start time *pulse centered @ t=0, use t0 << 0
tf    = 1000  # End time
dt    = 0.1  # Time step

# Unit conversion factors
##########################################################################
fs_conv = 41.34137335                  #(1fs    = 41.341473335 a.u.)
E_conv = 0.0001944690381               #(1MV/cm = 1.944690381*10^-4 a.u.)
B_conv = 4.25531E-6                    #(1T     = 4.25531*10^-6 a.u.)
THz_conv = 0.000024188843266           #(1THz   = 2.4188843266*10^-5 a.u.)
amp_conv = 150.97488474                #(1A     = 150.97488474)
eV_conv = 0.03674932176                #(1eV    = 0.036749322176 a.u.)

# Flags for testing and features
##########################################################################
user_out            = True  # Set to True to get user plotting and progress output
print_J_P_I_files   = True  # Set to True to get plotting of interband (P), intraband (J) contribution and emission
energy_plots        = False  # Set to True to plot 3d energy bands and contours
dipole_plots        = False  # Set tp True to plot dipoles (currently not working?)
test                = False  # Set to True to output travis testing parameters
matrix_method       = False  # Set to True to use old matrix method for solving
emission_wavep      = False  # additionally compute emission quasiclassically using wavepacket dynamics (
Bcurv_in_B_dynamics = False  # decide when appying B-field whether Berry curvature is used for dynamics
store_all_timesteps = True
save_figures        = True
structure_type      = 'zinc-blende' 
