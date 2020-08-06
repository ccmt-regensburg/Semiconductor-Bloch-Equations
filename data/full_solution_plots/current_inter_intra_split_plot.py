import matplotlib.pyplot as plt
import numpy as np
from plot_utilities import read_specific, plot_time_grid
from special_utilities import emission_exact
import seaborn as sns

import hfsbe.example

sns.set_palette(sns.color_palette("gist_ncar", 11))

fs_to_au = 41.34137335                  # (1fs    = 41.341473335 a.u.)
E_to_au = 0.0001944690381               # (1MV/cm = 1.944690381*10^-4 a.u.)
THz_to_au = 0.000024188843266           # (1THz   = 2.4188843266*10^-5 a.u.)
ev_to_au = 0.03674932176                # (1eV    = 0.036749322176 a.u.)
au_to_as = 0.529177

##############################################################################
# DATA READING
##############################################################################
Estring = 'E_5.0'
chirpstring = 'chirp_-0.920'

datapath1 = '/mnt/storage/Storage/dirac/dipole_on/' \
    + Estring + '/' + chirpstring + '/' + 'phase_0.00/'

Iexact1, Solution1 = read_specific(datapath1)

time = Iexact1[0]
I_exact_E_dir = Iexact1[1]
I_exact_ortho = Iexact1[2]


solution = Solution1[1]
electric_field = Solution1[2]
paths = Solution1[3]

########################################
# DIRAC HAMILTONIAN
########################################
system = hfsbe.example.BiTe(C0=0, C2=0, A=0.19732, R=0, mz=0)
system.eigensystem(gidx=1)


########################################
# KSPACE INDEXRANGE TO EVALUATE CURRENT
########################################
# gammaidx = (379, 421)
gammaidx = (300, 500)
leftidx = (0, gammaidx[0])
rightidx = (gammaidx[1], 800)

# KSPACE BOUNDARY FOR THE GAMMA CURRENT
kb = np.abs(paths[0, gammaidx[0], 0]/au_to_as)

# Left of Fermi
I_E_dir_left, I_ortho_left = emission_exact(system, paths, solution, [1, 0],
                                            idxrange=leftidx)

# Fermi wave vector
I_E_dir_fermi, I_ortho_fermi = emission_exact(system, paths, solution, [1, 0],
                                              idxrange=gammaidx)

# Right of Fermi
I_E_dir_right, I_ortho_right = emission_exact(system, paths, solution, [1, 0],
                                              idxrange=rightidx)

ax1 = plt.subplot2grid((2, 1), (0, 0))
ax1.plot(time/fs_to_au, electric_field/E_to_au)
ax1.set_xlabel(r'$t \text{ in } \si{fs}$')
ax1.set_ylabel(r'$E \text{ in } \si{MV/cm}$')
ax1.set_title(r'Electric Field')
ax1.grid(which='major', axis='x', linestyle='--')

ax2 = plt.subplot2grid((2, 1), (1, 0))
ax2.plot(time, I_E_dir_fermi + I_ortho_fermi)
ax2.plot(time, I_E_dir_left + I_E_dir_right + I_ortho_left + I_ortho_right)
ax2.plot(time, I_exact_E_dir + I_exact_ortho)
ax2.set_xlabel(r'$t \text{ in } \si{fs}$')
ax2.set_ylabel(r'$j$ in atomic units')
ax2.legend([r'$\Gamma$-point current $k_b=' + '{:.2f}'.format(kb) + r' \si{1/\angstrom}$',
            r'Edge current', r'Full current'])
plt.show()
# dipole_legend = ['dipole on', 'dipole off']

# plot_time_grid(time, kx_first_path, electric_field, Int_data_container,
#                band_structure, density_center_container,
#                standard_deviation=standard_deviation_container,
#                electric_field_legend=[r'$\omega_\mathrm{chirp} = -0.920\si{THz}$'],
#                current_legend=dipole_legend,
#                density_center_legend=dipole_legend + ['A Field'],
#                standard_deviation_legend=dipole_legend,
#                timelim=(-150, 150),
#                energylim=(0, 3.0),
#                bzboundary=1.07442,
#                savename='full_' + chirpstring + '_' + Estring + '.png')
