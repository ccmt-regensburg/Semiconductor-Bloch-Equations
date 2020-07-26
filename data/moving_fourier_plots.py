import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftshift
from plot_utilities import read_data, simple_fourier
import seaborn as sns
sns.set_palette(sns.color_palette("gist_ncar", 11))


fs_conv = 41.34137335                  # (1fs    = 41.341473335 a.u.)
E_conv = 0.0001944690381               # (1MV/cm = 1.944690381*10^-4 a.u.)
THz_conv = 0.000024188843266           # (1THz   = 2.4188843266*10^-5 a.u.)
amp_conv = 150.97488474                # (1A     = 150.97488474)
eV_conv = 0.03674932176                # (1eV    = 0.036749322176 a.u.)


plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['font.size'] = 20

##############################################################################
# DATA READING
##############################################################################
# Phase evaluation
phases = np.linspace(0, np.pi, 20)
phases = phases[0::2]
# orderpath = '../data-sbe/dirac/cep_phase_diagram/0.03_dist_to_gamma/'
orderpath = 'data-sbe/dirac/cep_phase_diagram/' + \
            '0.03_dist_to_gamma_full/dipole_on/'

# Evaluation parameters for fast scanning (phase diagram)
mlist = np.linspace(0, 0.0165372, 7)
mz = mlist[0]
chirplist = np.linspace(-0.92, 0.92, 11)
chirp = chirplist[0]

dist = '0.03'

mzstring = 'mz_' + '{:.7f}'.format(mz)
chirpstring = 'chirp_' + '{:.3f}'.format(chirp)

dirpath = mzstring + '/'
dirpath += chirpstring + '/'

parampaths = ['phase_{:1.2f}/'.format(p) for p in phases]
dirname = dirpath.strip('/').replace('_', '-').replace('/', '-')
paramlegend = [m.strip('/').replace('_', '=') for m in parampaths]

Idata, Iexactdata, Jdata, Pdata = read_data(orderpath, dirpath, parampaths)
#############################################################################


#############################################################################
# Evaluate fourier transform for moving narrow gaussian (~ 0.7 * alpha)
#############################################################################
def gaussian_envelope(t, t0):
    # alpha = 1033.53 Original 25fs envelope
    alpha = 700
    return np.exp(-(t-t0)**2.0/(2.0*alpha)**2)


time = Iexactdata[:, 0]
Int_exact_E_dir = Iexactdata[:, 1]
Int_exact_ortho = Iexactdata[:, 2]
freqw = Iexactdata[:, 3]

Int_data = (Int_exact_E_dir + Int_exact_ortho)
timelist = time[0]
print(timelist)

# Do a different Fourier transform per Gaussian (Gabor transform)
test_data = []
time_shifts = np.linspace(0, 7000, 11)
for t0 in time_shifts:
    # Iw_exact_full = fftshift(fft(gaussian_envelope(timelist, t0),
    #                          norm='ortho'))
    Iw_exact_full = fftshift(fft(Int_data*gaussian_envelope(timelist, t0),
                                 norm='ortho'))
    # Int_exact_full = (freqw[0]**2)*np.abs(Iw_exact_full)**2
    Int_exact_full = freqw**2*np.abs(Iw_exact_full)**2
    test_data.append(Int_exact_full[0])
test_data = np.array(test_data)

#############################################################################
# DATA PLOTTING
#############################################################################
# simple_fourier(freqw[:, 2048:], test_data[5:, 2048:], paramlegend=np.arange(5),
               # ylim=(1e-15, 1e5), xlim=(0, 1000))
time_shifts_labels = ['{:.2f} fs'.format(ts) for ts in time_shifts/fs_conv]
simple_fourier(freqw, test_data, paramlegend=time_shifts_labels, ylim=(1e-15, 1e5),
               xlim=(0, 30), dirname=r'Shifted Gaussian $\mathrm{FWHM}=56 \si{fs}$')
# mztitle = mzstring.replace('_', '=')
# chirptitle = chirpstring.replace('_', '=')
#############################################################################
