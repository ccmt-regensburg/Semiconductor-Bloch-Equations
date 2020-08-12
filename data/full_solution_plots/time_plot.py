import matplotlib.pyplot as plt
import numpy as np
from hfsbe.plotting import read_dataset, time_grid
# import seaborn as sns
# sns.set_palette(sns.color_palette("gist_ncar", 11))


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

Estring = 'E_5.0'
chirpstring = 'chirp_-0.920'

datapath1 = '/mnt/storage/Storage/dirac/velocity_gauge/dipole_off/' \
     + Estring + '/' + chirpstring + '/' + 'phase_0.00/'
# datapath2 = '/mnt/storage/Storage/dirac/dipole_off/velocity_gauge/dipole_off/' \
#     + Estring + '/' + chirpstring + '/' + 'phase_1.49/'
# datapath1 = '/mnt/storage/Storage/dirac/dipole_off/' \
#     + Estring + '/' + chirpstring + '/' + 'phase_0.00/'

########################################
# Extra path for comparisons
########################################
datapath2 = '/mnt/storage/Storage/dirac/length_gauge/dipole_off/' \
    + Estring + '/' + chirpstring + '/' + 'phase_0.00/'

Iexact1, Solution1 = read_dataset(datapath1)
Iexact2, Solution2 = read_dataset(datapath2)
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
    path = Solcontainer[i][2]
    electric_field = Solcontainer[i][3]

    first_path = path[0]

    f_e = Solution[:, 0, :, 3].real

    f_h_sum = np.sum(f_e)
    f_e_sum = np.sum(f_e, axis=0)

    kx_first_path = first_path[:, 0]/au_to_as
    ky_first_path = first_path[:, 1]/au_to_as

    # Occupation probability
    p_e = f_e.real/f_e_sum
    p_e_kx_weight = kx_first_path*p_e.T
    p_e_kx_weight_sq = (kx_first_path**2)*p_e.T

    p_e_variance = np.sum(p_e_kx_weight_sq, axis=1) - \
        np.sum(p_e_kx_weight, axis=1)**2
    # p_e_variance /= f_e_sum
    p_e_standard_deviation = np.sqrt(p_e_variance)
    p_e_density_center = np.sum(p_e_kx_weight, axis=1)

    Int_data_container.append(Int_data)
    density_center_container.append(p_e_density_center)
    standard_deviation_container.append(p_e_standard_deviation)

dkx = np.abs(kx_first_path[0] - kx_first_path[1])
Int_data_container = (au_to_as*dkx/(2*np.pi))*np.array(Int_data_container)/2
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

# dipole_legend = ['sbe full', 'sbe semiclas.', 'analytic semiclas.']
dipole_legend = ['velocity gauge', 'length gauge', 'analytic']
band_structure = dirac_conduction(kx_first_path, ky_first_path)


########################################
# Exta Analytical Dirac result
########################################
# Band Structure
knum = np.size(kx_first_path)
band_structure_gamma = dirac_conduction(kx_first_path, np.zeros(knum))
band_structure = np.vstack((band_structure, band_structure_gamma))
# Read Current
analytical_current = np.load('current_analytical.npy')
# Read Density
# analytical_density = np.load('density_analytical.npy')
Int_data_container = np.vstack((Int_data_container, analytical_current))

time_grid(time, kx_first_path, electric_field, Int_data_container,
          band_structure, density_center_container,
          standard_deviation=standard_deviation_container,
          electric_field_legend=[r'$\omega_\mathrm{chirp} = -0.920\si{THz}$'],
          current_legend=dipole_legend,
          density_center_legend=dipole_legend + ['A Field'],
          standard_deviation_legend=dipole_legend,
          timelim=(-150, 150),
          energylim=(0, 1.5),
          bzboundary=1.07442)
               # savename='full_' + chirpstring + '_' + Estring + '.png')
