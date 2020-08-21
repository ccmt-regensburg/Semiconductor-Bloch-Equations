import matplotlib.pyplot as plt
import numpy as np
from plot_utilities import find_max_intens, cep_plot

from data_paths import fullpath, suptitle, title
from hfsbe.plotting import read_datasets
from hfsbe.utility import conversion_factors as co

plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['font.size'] = 20


# Phase evaluation
phaselist = np.linspace(-np.pi, np.pi, 201)

chirplist = [-1.500, -0.920, 0.000]
chirp = chirplist[1]
chirpnumber = '{:.3f}'.format(chirp)
chirpstring = 'chirp_' + chirpnumber

fullpath += chirpstring + '/'
title += r'$\omega_\mathrm{chirp} = \SI{' + chirpnumber + r'}{THz}$'

# Evaluation parameters for fast scanning (phase diagram)
parampaths = [fullpath + 'phase_{:1.2f}/'.format(p) for p in phaselist]

# Idata, Iexactdata, Jdata, Pdata = read_data(orderpath, dirpath, parampaths)
Iexactdata, Iapprox, Sol = read_datasets(parampaths)

########################################
# TIME DATA
########################################
time = Iexactdata[:, 0]
I_exact_E_dir = Iexactdata[:, 1]
I_exact_ortho = Iexactdata[:, 2]

########################################
# EMISSION DATA
########################################
freqw = Iexactdata[:, 3]
Int_exact_E_dir = Iexactdata[:, 6]
Int_exact_ortho = Iexactdata[:, 7]

Int_avg, Int_max = find_max_intens(freqw, Int_exact_E_dir, Int_exact_ortho)

Int_data = Int_exact_E_dir + Int_exact_ortho

# phaselist = np.linspace(4*np.pi, 0, 80)# [10:-10] + np.linspace
# phaselist = phaselist[10:-10] - np.pi
# Int_data = np.vstack((Int_data, Int_data, Int_data, Int_data))[10:-10]
# breakpoint()
cep_plot(freqw, phaselist, Int_data, xlim=(0, 21), max=Int_avg, show=False,
         min=1e-19, suptitle=suptitle, title=title)

plt.savefig('unnamed.png')
