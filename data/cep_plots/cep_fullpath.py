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

########################################
# DIRAC
########################################

# Negative Phase
# orderpath = '../data-sbe/dirac/cep_phase_diagram/0.03_dist_to_gamma/'
# dirpath = 'E_5.0_negative_phase/'

# Positive Phase
# orderpath = '../data-sbe/dirac/cep_phase_diagram/0.03_dist_to_gamma/'
# dirpath = 'E_field_sweep/mz_0.0000000/E_5.0/'
# orderpath = '/mnt/storage/Storage/cep_data_huber/dirac/0.03_dist_to_gamma_full_Nk1_1000/velocity_gauge/'
# dirpath = 'dipole_on/'
# suptitle = 'Dirac'
# title = r'$E=5 \si{MV/cm}$ ' + \
#         r'$\omega_\mathrm{carrier} = 25 \si{THz} $ ' + \
#         r'$\omega_\mathrm{chirp} = -0.92 \si{THz}$'


########################################
# Semiconductor Quadratic
########################################
# orderpath = '/mnt/storage/Storage/cep_data_huber/semiconductor_quadratic/0.03_dist_to_gamma_quadratic_Nk1_1000/' + \
#             'velocity_gauge/dipole_on/'
# dirpath = 'mz_0.2179616_A_0.0505814/E_5.0/'
# suptitle = 'Semiconductor, Low Dipole'
# title = r'$E=5 \si{MV/cm}$ ' + \
#         r'$\omega_\mathrm{carrier} = 25 \si{THz}$ ' + \
#         r'$\omega_\mathrm{chirp} = -0.92 \si{THz}$ ' + \
#         r'$\epsilon_\mathrm{gap} = 0.607 \si{\eV}$ ' + \
#         r'$|d_\mathrm{max}| = 0.21 \si{e \angstrom}$'

########################################
# Semiconductor High Dipole
########################################
orderpath = '/mnt/storage/Storage/cep_data_huber/semiconductor_high_dipole/0.03_dist_to_gamma_high_dipole_Nk1_1000/' + \
            'velocity_gauge/dipole_on/'
dirpath = 'mz_0.0607538_A_0.0544401/'
suptitle = 'Semiconductor, High Dipole'
title = r'$E=5 \si{MV/cm}$ ' + \
        r'$\omega_\mathrm{carrier} = 25 \si{THz}$ ' + \
        r'$\omega_\mathrm{chirp} = -0.92 \si{THz}$ ' + \
        r'$\epsilon_\mathrm{gap} = 0.485 \si{\eV}$ ' + \
        r'$|d_\mathrm{max}| = 2.03 \si{e \angstrom}$'

########################################
# RESUMMED
########################################
# orderpath = '../data-sbe/resummed_hamiltonian/cep_phase_diagramm/0.03_dist_to_gamma/'
# dirpath = 'E_sweep/E_5.0/'

chirplist = [-0.920, 0.000]
chirp = chirplist[0]
chirpstring = 'chirp_' + '{:.3f}'.format(chirp)

dirpath += chirpstring + '/'

# Evaluation parameters for fast scanning (phase diagram)
parampaths = ['phase_{:1.2f}/'.format(p) for p in phases]

Idata, Iexactdata, Jdata, Pdata = read_data(orderpath, dirpath, parampaths)

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
cep_plot(freqw, phases, Int_data, xlim=(0, 30), max=Int_avg, show=False,
         min=1e-19, suptitle=suptitle, title=title)

plt.savefig('unnamed.png')
