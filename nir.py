import numpy as np
from numba import njit
import matplotlib.pyplot as pl
from scipy import optimize, special, interpolate, signal
from scipy.signal import hilbert, butter, lfilter, sosfilt

import params

def main():
    opt_pulses()

def opt_pulses():
    #Prepare pyplot axes
    fig, ax         = pl.subplots(1, 1)
    fs_conv         = params.fs_conv
    E_conv          = params.E_conv
    THz_conv        = params.THz_conv

    #Load THz Pulse data
    thzPulse        = np.loadtxt("Transient_25THz.txt", delimiter=",")
    thzPulse[:,0]   *= fs_conv                                              #Recalculation of s into a.u.

    initThz         = [1, 100*fs_conv, 0, 25*THz_conv, 0, 0]
    tOpt, tCov      = optimize.curve_fit(transient, thzPulse[:,0], thzPulse[:,1], p0=initThz)
    ax.plot(thzPulse[:,0], thzPulse[:,1], label="THz-Pulse")
    ax.plot(thzPulse[:,0], transient(thzPulse[:,0], *tOpt), label="THz-Pulse fitted")

    ax.set_ylabel("electrical field normed to maximum")
    ax.set_xlabel("time in atomic units")
    ax.legend()
    
#    pl.show()
    pl.clf()
    pl.close()
    return tOpt

@njit
def transient(t, A, alpha, mu, w, chirp, phase):
    return A*np.exp(-(t-mu)**2.0/(2.0*alpha)**2)*np.sin(2.0*np.pi*w*t*(1 + chirp*t) + phase)

#Appendix to use main function
if __name__ == "__main__":
    main()
