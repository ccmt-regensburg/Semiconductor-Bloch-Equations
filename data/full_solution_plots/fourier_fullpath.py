import matplotlib.pyplot as plt
import numpy as np
from plot_utilities import read_specific, total_fourier
import seaborn as sns
sns.set_palette(sns.color_palette("gist_ncar", 11))


fs_to_au = 41.34137335                  # (1fs    = 41.341473335 a.u.)
E_to_au = 0.0001944690381               # (1MV/cm = 1.944690381*10^-4 a.u.)
THz_to_au = 0.000024188843266           # (1THz   = 2.4188843266*10^-5 a.u.)
ev_to_au = 0.03674932176                # (1eV    = 0.036749322176 a.u.)
au_to_as = 0.529177


plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['font.size'] = 20

##############################################################################
# DATA READING
##############################################################################
Estring = 'E_2.5'
chirpstring = 'chirp_-0.920'

datapath1 = '/mnt/storage/Storage/dirac/dipole_on/' \
    + Estring + '/' + chirpstring + '/' + 'phase_0.00/'
datapath2 = '/mnt/storage/Storage/dirac/dipole_off/' \
    + Estring + '/' + chirpstring + '/' + 'phase_0.00/'

Iexact1, Solution1 = read_specific(datapath1)
Iexact2, Solution2 = read_specific(datapath2)

Icontainer = [Iexact1, Iexact2]
#############################################################################

freqw_container = []
Int_exact_E_dir_container = []
Int_exact_ortho_container = []

for i in range(len(Icontainer)):
    freqw = Icontainer[i][3]
    Int_exact_E_dir = Icontainer[i][6]
    Int_exact_ortho = Icontainer[i][7]

    freqw_container.append(freqw)
    Int_exact_E_dir_container.append(Int_exact_E_dir)
    Int_exact_ortho_container.append(Int_exact_ortho)


freqw_container = np.array(freqw_container)
Int_exact_E_dir_container = np.array(Int_exact_E_dir_container)
Int_exact_ortho_container = np.array(Int_exact_ortho_container)

dipole_legend = ['dipole on', 'dipole off']
total_fourier(freqw_container, Int_exact_E_dir_container,
              Int_exact_ortho_container, ylim=(10e-12, 1),
              ylabel=r'$[I_\mathrm{exact}](\omega)$ intensity in a.u.',
              paramlegend=dipole_legend,
              savename='fourier_' + chirpstring + '_' + Estring + '.png')
