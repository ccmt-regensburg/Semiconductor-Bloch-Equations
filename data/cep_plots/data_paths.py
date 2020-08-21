import numpy as np

global_dir = '/mnt/storage/Storage/'
##############################################################################
# DATA READING
##############################################################################

########################################
# Dirac
########################################

####################################
# 2line
####################################

################################
# narrow
################################
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

############################
# 25THz better resolution
############################
# orderpath = '/mnt/storage/Storage/dirac_25THz/'
# dirpath = 'E_5.0/'
# suptitle = 'Dirac'
# title = r'$E=5 \si{MV/cm}$ ' + \
#         r'$\omega_\mathrm{carrier} = 25 \si{THz} $ ' + \
#         r'$\omega_\mathrm{chirp} = -0.92 \si{THz}$'

# T1 10 T2 1
# fullpath = global_dir + 'dirac_25THz_alpha_25fs/cep_0.03_dist_to_gamma/E_5.0/T1_10_T2_1/'
# suptitle = r'Dirac $E=\SI{5}{MV/cm}$'
# title = r'$T_1=\SI{10}{fs}$ $T_2=\SI{1}{fs}$ ' + \
#         r'$\omega_\mathrm{carrier} = \SI{25}{THz} $ ' + \
#         r'$\omega_\mathrm{chirp} = \SI{-0.92}{THz}$'

# # T1 15 T2 1
# fullpath = global_dir + 'dirac_25THz_alpha_25fs/cep_0.03_dist_to_gamma/E_5.0/T1_15_T2_1/'
# suptitle = r'Dirac $E=\SI{5}{MV/cm}$'
# title = r'$T_1=\SI{15}{fs}$ $T_2=\SI{1}{fs}$ ' + \
#         r'$\omega_\mathrm{carrier} = \SI{25}{THz} $ ' + \
#         r'$\omega_\mathrm{chirp} = \SI{-0.92}{THz}$'

# # T1 20 T2 1
# fullpath = global_dir + 'dirac_25THz_alpha_25fs/cep_0.03_dist_to_gamma/E_5.0/T1_20_T2_1/'
# suptitle = r'Dirac $E=\SI{5}{MV/cm}$'
# title = r'$T_1=\SI{20}{fs}$ $T_2=\SI{1}{fs}$ ' + \
#         r'$\omega_\mathrm{carrier} = \SI{25}{THz} $ ' + \
#         r'$\omega_\mathrm{chirp} = \SI{-0.92}{THz}$'

# T1 10 T2 2
# fullpath = global_dir + 'dirac_25THz_alpha_25fs/cep_0.03_dist_to_gamma/E_5.0/T1_10_T2_2/'
# suptitle = r'Dirac $E=\SI{5}{MV/cm}$'
# title = r'$T_1=\SI{10}{fs}$ $T_2=\SI{2}{fs}$ ' + \
#         r'$\omega_\mathrm{carrier} = \SI{25}{THz} $ ' + \
#         r'$\omega_\mathrm{chirp} = \SI{-0.92}{THz}$'

# T1 15 T2 2
# fullpath = global_dir + 'dirac_25THz_alpha_25fs/cep_0.03_dist_to_gamma/E_5.0/T1_15_T2_2/'
# suptitle = r'Dirac $E=\SI{5}{MV/cm}$'
# title = r'$T_1=\SI{15}{fs}$ $T_2=\SI{2}{fs}$ ' + \
#         r'$\omega_\mathrm{carrier} = \SI{25}{THz} $ ' + \
#         r'$\omega_\mathrm{chirp} = \SI{-0.92}{THz}$'

# # T1 20 T2 2
# fullpath = global_dir + 'dirac_25THz_alpha_25fs/cep_0.03_dist_to_gamma/E_5.0/T1_20_T2_2/'
# suptitle = r'Dirac $E=\SI{5}{MV/cm}$' 
# title = r'$T_1=\SI{20}{fs}$ $T_2=\SI{2}{fs}$ ' + \
#         r'$\omega_\mathrm{carrier} = \SI{25}{THz} $ ' + \
#         r'$\omega_\mathrm{chirp} = \SI{-0.92}{THz}$'

################################
# broad
################################
# 30Thz broad pulse
# orderpath = '/mnt/storage/Storage/dirac_30THz_alpha_45fs/cep_0.03_dist_to_gamma/'
# dirpath = orderpath + 'E_5.0/'
# suptitle = None
# title = None

########################################
# BiTe
########################################

####################################
# 2line
####################################

################################
# narrow
################################

############################
# 25THz better resolution
############################
# # 0.01_dist_to_gamma
# T1 1000 T2 1
# fullpath = global_dir + 'bite_25THz_alpha_25fs/cep_0.01_dist_to_gamma/E_5.0/T1_1000_T2_1/'
# suptitle = r'$\mathrm{Bi}_2\mathrm{Te}_3$ $k^3$ ' + r'$E=\SI{5}{MV/cm}$ ' + \
#            r'$k_y=0.01 \frac{2\pi}{a}$'
# title = r'$T_1=\SI{1000}{fs}$ $T_2=\SI{1}{fs}$ ' + \
#         r'$\omega_\mathrm{carrier} = \SI{25}{THz} $ '

# T1 10 T2 1
# fullpath = global_dir + 'bite_25THz_alpha_25fs/cep_0.01_dist_to_gamma/E_5.0/T1_10_T2_1/'
# suptitle = r'$\mathrm{Bi}_2\mathrm{Te}_3$ $k^3$ ' + r'$E=\SI{5}{MV/cm}$ ' + \
#            r'$k_y=0.01 \frac{2\pi}{a}$'
# title = r'$T_1=\SI{10}{fs}$ $T_2=\SI{1}{fs}$ ' + \
#         r'$\omega_\mathrm{carrier} = \SI{25}{THz} $ '

# # 0.03_dist_to_gamma
# T1 1000 T2 1
# fullpath = global_dir + 'bite_25THz_alpha_25fs/cep_0.03_dist_to_gamma/E_5.0/T1_1000_T2_1/'
# suptitle = r'$\mathrm{Bi}_2\mathrm{Te}_3$ $k^3$ ' + r'$E=\SI{5}{MV/cm}$ ' + \
#            r'$k_y=0.03 \frac{2\pi}{a}$'
# title = r'$T_1=\SI{1000}{fs}$ $T_2=\SI{1}{fs}$ ' + \
#         r'$\omega_\mathrm{carrier} = \SI{25}{THz} $ '

# T1 10 T2 1
# fullpath = global_dir + 'bite_25THz_alpha_25fs/cep_0.03_dist_to_gamma/E_5.0/T1_10_T2_1/'
# suptitle = r'$\mathrm{Bi}_2\mathrm{Te}_3$ $k^3$ ' + r'$E=\SI{5}{MV/cm}$ ' + \
#            r'$k_y=0.03 \frac{2\pi}{a}$'
# title = r'$T_1=\SI{10}{fs}$ $T_2=\SI{1}{fs}$ ' + \
#         r'$\omega_\mathrm{carrier} = \SI{25}{THz} $ '

# # 0.05_dist_to_gamma
# T1 1000 T2 1
# fullpath = global_dir + 'bite_25THz_alpha_25fs/cep_0.05_dist_to_gamma/E_5.0/T1_1000_T2_1/'
# suptitle = r'$\mathrm{Bi}_2\mathrm{Te}_3$ $k^3$ ' + r'$E=\SI{5}{MV/cm}$ ' + \
#            r'$k_y=0.05 \frac{2\pi}{a}$'
# title = r'$T_1=\SI{1000}{fs}$ $T_2=\SI{1}{fs}$ ' + \
#         r'$\omega_\mathrm{carrier} = \SI{25}{THz} $ ' + \
#         r'$\omega_\mathrm{chirp} = \SI{-1.50}{THz}$'

# T1 10 T2 1
# fullpath = global_dir + 'bite_25THz_alpha_25fs/cep_0.05_dist_to_gamma/E_5.0/T1_10_T2_1/'
# suptitle = r'$\mathrm{Bi}_2\mathrm{Te}_3$ $k^3$ ' + r'$E=\SI{5}{MV/cm}$ ' + \
#            r'$k_y=0.05 \frac{2\pi}{a}$'
# title = r'$T_1=\SI{10}{fs}$ $T_2=\SI{1}{fs}$ ' + \
#         r'$\omega_\mathrm{carrier} = \SI{25}{THz} $ ' + \
#         r'$\omega_\mathrm{chirp} = \SI{-1.50}{THz}$'


########################################
# RESUMMED
########################################

########################################
# 2line
########################################

# narrow
########################################
# T1 10
# fullpath = global_dir + 'resummed_25THz_alpha_25fs/cep_0.03_dist_to_gamma/E_5.0/T1_10/'
# suptitle = r'DFT Hamiltonian'
# title = r'$E=\SI{5}{MV/cm}$ ' + \
#         r'$T_1=\SI{10}{fs}$ ' + \
#         r'$\omega_\mathrm{carrier} = \SI{25}{THz} $ ' + \
#         r'$\omega_\mathrm{chirp} = \SI{-0.92}{THz}$'

# broad 
########################################
# T1 1000 T2 1
fullpath = '/mnt/storage/Storage/resummed_30THz_alpha_45fs/cep_0.03_dist_to_gamma/E_5.0/T1_1000_T2_1/'
suptitle = r'DFT Hamiltonian ' + r'$E=\SI{5}{MV/cm}$ ' + r'$k_y=0.03 \frac{2\pi}{a}$'
title = r'$T_1=\SI{1000}{fs}$ ' + \
        r'$\omega_\mathrm{carrier} = \SI{30}{THz} $ '

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

####################################
# 2line
####################################

################################
# narrow
################################

############################
# 40THz better resolution
############################

# # 0.03_dist_to_gamma
# T1 1000 T2 1
# fullpath = '/mnt/storage/Storage/semiconductor_high_40Thz_alpha_25fs/cep_0.03_dist_to_gamma/E_5.0/T1_1000_T2_1/'
# suptitle = r'Semiconductor, High Dipole ' + r'$E=5 \si{MV/cm}$ ' + \
#            r'$k_y=0.03 \frac{2\pi}{a}$'
# title = r'$\omega_\mathrm{carrier} = 40 \si{THz}$ ' + \
#         r'$|d_\mathrm{max}| = 2.03 \si{e \angstrom}$ '
#         # r'$\epsilon_\mathrm{gap} = 0.485 \si{\eV}$ ' + \


# T1 10 T2 1
# fullpath = '/mnt/storage/Storage/semiconductor_high_40Thz_alpha_25fs/cep_0.03_dist_to_gamma/E_5.0/T1_10_T2_1/'
# suptitle = r'Semiconductor, High Dipole ' + r'$E=5 \si{MV/cm}$ ' + \
#            r'$k_y=0.03 \frac{2\pi}{a}$'
# title = r'$\omega_\mathrm{carrier} = 40 \si{THz}$ ' + \
#         r'$|d_\mathrm{max}| = 2.03 \si{e \angstrom}$ '
#         # r'$\epsilon_\mathrm{gap} = 0.485 \si{\eV}$ ' + \

############################
# 25THz standard resolution
############################
# # 25 THz
# orderpath = '/mnt/storage/Storage/cep_data_huber/semiconductor_high_dipole/0.03_dist_to_gamma_high_dipole_Nk1_1000/' + \
#             'velocity_gauge/dipole_on/'
# dirpath = 'mz_0.0607538_A_0.0544401/'
# suptitle = 'Semiconductor, High Dipole'
# title = r'$E=5 \si{MV/cm}$ ' + \
#         r'$\omega_\mathrm{carrier} = 25 \si{THz}$ ' + \
#         r'$\omega_\mathrm{chirp} = -0.92 \si{THz}$ ' + \
#         r'$\epsilon_\mathrm{gap} = 0.485 \si{\eV}$ ' + \
#         r'$|d_\mathrm{max}| = 2.03 \si{e \angstrom}$'

################################
# 40THz standard resolution
################################
# 40THz
# orderpath = '/mnt/storage/Storage/semiconductor_high_40Thz/'
# dirpath = 'mz_0.0607538_A_0.0544401/'
# suptitle = 'Semiconductor, High Dipole'
# title = r'$E=5 \si{MV/cm}$ ' + \
#         r'$\omega_\mathrm{carrier} = 40 \si{THz}$ ' + \
#         r'$\omega_\mathrm{chirp} = -0.92 \si{THz}$ ' + \
#         r'$\epsilon_\mathrm{gap} = 0.485 \si{\eV}$ ' + \
#         r'$|d_\mathrm{max}| = 2.03 \si{e \angstrom}$'

# 40THz T1 10
# fullpath = global_dir + '/semiconductor_high_40Thz_alpha_25fs/cep_0.03_dist_to_gamma/E_5.0/T1_10/'
# suptitle = 'Semiconductor, High Dipole'
# title = r'$E=5 \si{MV/cm}$ ' + \
#         r'$\omega_\mathrm{carrier} = 40 \si{THz}$ ' + \
#         r'$\omega_\mathrm{chirp} = -0.92 \si{THz}$ ' + \
#         r'$T_1 = \SI{10}{fs}$ ' + \
#         r'$\epsilon_\mathrm{gap} = 0.485 \si{\eV}$ ' + \
#         r'$|d_\mathrm{max}| = 2.03 \si{e \angstrom}$'


