import matplotlib.pyplot as plt
import numpy as np
from plot_utilities import read_data, total_fourier, find_base_freq


fs_conv = 41.34137335                  # (1fs    = 41.341473335 a.u.)
E_conv = 0.0001944690381               # (1MV/cm = 1.944690381*10^-4 a.u.)
THz_conv = 0.000024188843266           # (1THz   = 2.4188843266*10^-5 a.u.)
amp_conv = 150.97488474                # (1A     = 150.97488474)
eV_conv = 0.03674932176                # (1eV    = 0.036749322176 a.u.)


plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['font.size'] = 20

# Phase evaluation
phases = np.linspace(0, np.pi, 20)
orderpath = '../data-sbe/dirac/cep_phase_diagram/0.03_dist_to_gamma/'

# Evaluation parameters for fast scanning (phase diagram)
mlist = np.linspace(0, 0.0275620, 6)
chirplist = np.linspace(-0.92, 0.92, 11)

mz = mlist[0]
dist = '0.03'

mzstring = 'mz_' + '{:.7f}'.format(mz)

dirpath = mzstring + '/'
# dirpath += chirpstring + '/'

# # Evluation parameters for simple plots
# dist = '0.07'
# dirpath = dist + '_dist_to_gamma/'

parampaths = ['chirp_' + '{:.3f}'.format(chirp) + '/phase_0.00/' for chirp in chirplist]
paramlegend = [m.strip('/').replace('_', '=') for m in parampaths]

Idata, Iexactdata, Jdata, Pdata = read_data(orderpath, dirpath, parampaths)
# breakpoint()

freqw = Iexactdata[:, 3]
Int_exact_E_dir = Iexactdata[:, 6]
Int_exact_ortho = Iexactdata[:, 7]

Int_exact_base_freq = find_base_freq(freqw, Int_exact_E_dir, Int_exact_ortho)

size = np.size(freqw, axis=1)
Int_pos_total = Int_exact_E_dir[:, size//2:] + Int_exact_ortho[:, size//2:]
Int_exact_max = np.max(Int_pos_total, axis=1)

Int_exact_E_dir = (Int_exact_E_dir.T/Int_exact_max).T
Int_exact_ortho = (Int_exact_ortho.T/Int_exact_max).T


Int_exact_output = np.real(Int_exact_E_dir + Int_exact_ortho).T
print(np.min(Int_exact_output))
Output = np.hstack((np.real(freqw.T[:, 0, np.newaxis]), Int_exact_output))
np.savetxt("intensity.dat", Output)
