import matplotlib.pyplot as plt
import numpy as np
from plot_utilities import read_data, find_max_intens, cep_plot


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
chirplist = np.linspace(0, 0.92, 6)
chirplist[1:] *= -1

mz = mlist[5]
chirp = chirplist[5]
dist = '0.03'

mzstring = 'mz_' + '{:.7f}'.format(mz)
chirpstring = 'chirp_' + '{:.3f}'.format(chirp) 

dirpath = mzstring + '/'
dirpath += chirpstring + '/'

# # Evluation parameters for simple plots
# dist = '0.07'
# dirpath = dist + '_dist_to_gamma/'


parampaths = ['phase_{:1.2f}/'.format(p) for p in phases]
dirname = dirpath.strip('/').replace('_', '-').replace('/', '-')
paramlegend = [m.strip('/').replace('_', '=') for m in parampaths]

Idata, Iexactdata, Jdata, Pdata = read_data(orderpath, dirpath, parampaths)

freqw = Iexactdata[:, 3]
Int_exact_E_dir = Iexactdata[:, 6]
Int_exact_ortho = Iexactdata[:, 7]

# Int_exact_E_dir = (Int_exact_E_dir.T/Int_exact_base_freq).T
# Int_exact_ortho = (Int_exact_ortho.T/Int_exact_base_freq).T

Int_max = find_max_intens(freqw, Int_exact_E_dir, Int_exact_ortho)

mztitle = mzstring.replace('_', '=')
chirptitle = chirpstring.replace('_', '=')

cep_plot(freqw, phases, Int_exact_E_dir + Int_exact_ortho,
         mztitle + r'H ' + chirptitle + r'$\mathrm{MV}/\mathrm{cm}$',
         max=Int_max)
