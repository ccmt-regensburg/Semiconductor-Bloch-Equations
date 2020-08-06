import matplotlib.pyplot as plt
import numpy as np
from plot_utilities import read_specific, plot_time_grid
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
# Phase evaluation
# datapath1 = '/mnt/storage/Storage/dirac/dipole_on/' \
#     + 'E_10.0/chirp_-0.920/phase_0.00/'
# datapath2 = '/mnt/storage/Storage/dirac/dipole_off/' \
#     + 'E_10.0/chirp_-0.920/phase_0.00/'

Estring = 'E_10.0'
chirpstring = 'chirp_-0.920'

datapath1 = '/mnt/storage/Storage/dirac/dipole_on/' \
    + Estring + '/' + chirpstring + '/' + 'phase_0.00/'
datapath2 = '/mnt/storage/Storage/dirac/dipole_off/' \
    + Estring + '/' + chirpstring + '/' + 'phase_0.00/'
datapath3 = '/mnt/storage/Storage/dirac/dipole_on/' \
    + Estring + '/' + chirpstring + '/' + 'phase_1.49/'
datapath4 = '/mnt/storage/Storage/dirac/dipole_off/' \
    + Estring + chirpstring + 'phase_1.49/'

Iexact1, Solution1 = read_specific(datapath1)
Iexact2, Solution2 = read_specific(datapath2)
# Iexact3, Solution3 = read_specific(datapath3)
# Iexact4, Solution4 = read_specific(datapath4)

Icontainer = [Iexact1, Iexact2] #, Iexact3, Iexact4]
Solcontainer = [Solution1, Solution2] #, Solution3, Solution4]
#############################################################################

Int_data_container = []
density_center_container = []
standard_deviation_container = []

for i in range(len(Icontainer)):
    time = Icontainer[i][0]
    Int_exact_E_dir = Icontainer[i][1]
    Int_exact_ortho = Icontainer[i][2]

    Int_data = (Int_exact_E_dir + Int_exact_ortho)

    Solution = Solcontainer[i][1]
    electric_field = Solcontainer[i][2]
    path = Solcontainer[i][3]

    first_path = path[0]

    f_e = Solution[:, 0, :, 3].real

    f_h_sum = np.sum(f_e)
    f_e_sum = np.sum(f_e, axis=0)

    kx_first_path = first_path[:, 0]/au_to_as
    ky_first_path = first_path[:, 1]/au_to_as

    f_e_kx_weight = kx_first_path*f_e.T.real
    f_e_kx_weight_sq = (kx_first_path**2)*f_e.T.real

    f_e_variance = np.sum(f_e_kx_weight_sq - f_e_kx_weight**2, axis=1)/f_e_sum
    f_e_standard_deviation = np.sqrt(f_e_variance)
    f_e_density_center = np.sum(f_e_kx_weight, axis=1)/f_e_sum

    Int_data_container.append(Int_data)
    density_center_container.append(f_e_density_center)
    standard_deviation_container.append(f_e_standard_deviation)

Int_data_container = np.array(Int_data_container)
density_center_container = np.array(density_center_container)
standard_deviation_container = np.array(standard_deviation_container)


########################################
# Band structure
########################################
def dirac_conduction(kx, ky):
    # kx *= au_to_as
    # ky *= au_to_as
    return 2.84134*np.sqrt(kx**2 + ky**2)


A_field = -7.80951974*np.cumsum(electric_field)

density_center_container = np.vstack((density_center_container, A_field))

dipole_legend = ['dipole on', 'dipole off']
band_structure = dirac_conduction(kx_first_path, ky_first_path)
plot_time_grid(time, kx_first_path, electric_field, Int_data_container,
               band_structure, density_center_container,
               standard_deviation=standard_deviation_container,
               electric_field_legend=[r'$\omega_\mathrm{chirp} = -0.920\si{THz}$'],
               current_legend=dipole_legend,
               density_center_legend=dipole_legend + ['A Field'],
               standard_deviation_legend=dipole_legend,
               timelim=(-150, 150),
               energylim=(0, 3.0),
               bzboundary=1.07442,
               savename='full_' + chirpstring + '_' + Estring + '.png')

# plt.plot(time/fs_to_au, standard_deviation_container.T)
# plt.xlabel(r'$t \text{ in } \si{fs}$')
# plt.ylabel(r'$k_x \text{ in } \si{1/\angstrom}$')
# plt.legend(dipole_legend)
# plt.title('Standard Deviation')
# plt.show()
