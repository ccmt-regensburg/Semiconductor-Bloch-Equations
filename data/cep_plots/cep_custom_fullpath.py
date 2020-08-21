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
phaselist = np.linspace(0, np.pi, 20)

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
# EMISSION DATA
########################################
horder = 13
lorder = 21
# find highest order index
freqw = Iexactdata[:, 3]

hidx = np.where(freqw[0, :].real > 13)[0][0]
lidx = np.where(freqw[0, :].real < 21)[0][-1] + 1

freqw = freqw[:, hidx:lidx]
Int_exact_E_dir = Iexactdata[:, 6][:, hidx:lidx]
Int_exact_ortho = Iexactdata[:, 7][:, hidx:lidx]

Int_data = Int_exact_E_dir + Int_exact_ortho

Int_data_max = np.max(Int_data, axis=1)
Int_data_avg_max = np.average(Int_data_max).real

Int_data_min = np.min(Int_data, axis=1)
Int_data_avg_min = np.average(Int_data_min).real

cep_plot(freqw, phaselist, Int_data, xlim=(horder, lorder), max=Int_data_avg_max,
         show=False, min=Int_data_avg_min/Int_data_avg_max, suptitle=suptitle, title=title)

plt.savefig('unnamed.png')
