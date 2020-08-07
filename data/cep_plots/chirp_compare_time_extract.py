import matplotlib.pyplot as plt
import numpy as np
from plot_utilities import read_data, total_fourier, find_base_freq


fs_conv = 41.34137335                  # (1fs    = 41.341473335 a.u.)
E_conv = 0.0001944690381               # (1MV/cm = 1.944690381*10^-4 a.u.)
THz_conv = 0.000024188843266           # (1THz   = 2.4188843266*10^-5 a.u.)
amp_conv = 150.97488474                # (1A     = 150.97488474)
eV_conv = 0.03674932176                # (1eV    = 0.036749322176 a.u.)

# Phase evaluation
phases = np.linspace(0, np.pi, 20)
orderpath = '../data-sbe/dirac/cep_phase_diagram/0.03_dist_to_gamma/'

# Evaluation parameters for fast scanning (phase diagram)
mlist = np.linspace(0, 0.0165372, 7)
chirplist = np.linspace(-0.92, 0.92, 11)
phaselist = np.linspace(0, np.pi, 20)

chirp = chirplist[0]
mz = mlist[0]
dist = '0.03'

mzstring = 'mz_' + '{:.7f}'.format(mz)
dirpath = mzstring + '/'

parampaths = ['chirp_{:.3f}'.format(chirp) + '/' + 'phase_{:.2f}'.format(phase) + '/'
              for phase in phaselist]
paramlegend = [m.strip('/').replace('_', '=') for m in parampaths]

Idata, Iexactdata, Jdata, Pdata = read_data(orderpath, dirpath, parampaths)
# breakpoint()

time = Iexactdata[:, 0]
Int_exact_E_dir = Iexactdata[:, 1]
Int_exact_ortho = Iexactdata[:, 2]

size = np.size(time, axis=1)
Int_exact_output = np.real(Int_exact_E_dir + Int_exact_ortho).T
Output = np.hstack((np.real(time[0])[:, np.newaxis], Int_exact_output))

chirpname = '_chirp_{:.3f}'.format(chirp)
np.savetxt("dirac_time_intensity" + chirpname + ".dat", Output)
