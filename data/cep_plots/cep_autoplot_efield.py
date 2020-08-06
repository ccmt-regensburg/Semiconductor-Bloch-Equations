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
############################################################################
# DIRAC
############################################################################
# orderpath = '../data-sbe/dirac/cep_phase_diagram/' + \
#             '0.03_dist_to_gamma/E_field_sweep/' + \
#             'mz_0.0000000/'
# min_intens = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-4, 1e-4]
# min_intens = [1e-14, 1e-12, 1e-8]

########################################
# DIRAC NEGATIVE PHASE
########################################
# orderpath = '../data-sbe/dirac/cep_phase_diagram/' + \
#             '0.03_dist_to_gamma/E_5.0_negative_phase/' + \
#             'mz_0.0000000/'
# min_intens = [1e-14]

############################################################################
# SEMICONDUCTOR QUADRATIC
############################################################################
orderpath = '../data-sbe/semiconductor_hamiltonian/cep_phase_diagram/' + \
            '0.03_dist_to_gamma/E_field_sweep/' + \
            'mz_0.3457844_A_0.0478252/'
min_intens = [1e-22, 1e-18, 1e-16, 1e-14, 1e-12, 1e-10, 1e-10, 1e-10]
# min_intens = [1e-22, 1e-18, 1e-14]
############################################################################
# SEMICONDUCTOR FLAT
############################################################################
# orderpath = '../data-sbe/semiconductor_hamiltonian/cep_phase_diagram/' + \
#             '0.03_dist_to_gamma/E_field_sweep/' + \
#             'mx_0.0165372/'
# min_intens = [1e-22, 1e-18, 1e-16, 1e-14, 1e-12, 1e-10, 1e-10, 1e-10]
# min_intens = [1e-24, 1e-20, 1e-14]
############################################################################
#
# Evaluation parameters for fast scanning (phase diagram)
# Elist = np.linspace(2.5, 20, 8)
Elist = [5.0]
chirplist = np.linspace(-0.92, 0.92, 11)
# chirplist = [-0.92]

# mz = mlist[1]

k = 0
for i, mz in enumerate(Elist):

    Estring = 'E_' + '{:.1f}'.format(mz)

    for j, chirp in enumerate(chirplist):
        chirpstring = 'chirp_' + '{:.3f}'.format(chirp)

        dirpath = Estring + '/'
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

        Int_avg_max, Int_max = find_max_intens(freqw, Int_exact_E_dir,
                                               Int_exact_ortho)

        mztitle = Estring.replace('_', '=')
        chirptitle = chirpstring.replace('_', '=')

        # Int_data = ((Int_exact_E_dir + Int_exact_ortho).T/np.real(Int_max)).T
        Int_data = (Int_exact_E_dir + Int_exact_ortho)
        # freqw *= rescaledata[i]
        cep_plot(freqw, phases, Int_data,
                 xlim=(0, 30), max=Int_avg_max, min=min_intens[i], show=False,
                 title=mztitle + ' ' + chirptitle)

        numberstring = '{:02d}'.format(k)
        k += 1
        plt.savefig(numberstring + '_' + Estring + '_' + chirpstring + '.png')
        plt.clf()
