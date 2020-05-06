import numpy as np
from numba import njit
from scipy import optimize

import params

def opt_pulses():
    #Prepare pyplot axes
    fs_conv         = params.fs_conv
    E_conv          = params.E_conv
    THz_conv        = params.THz_conv
    vel_light       = params.c

    #Load THz Pulse data
    thzPulse        = np.loadtxt("Transient.dat")
    thzPulse[:,0]   *= 1e15*fs_conv                             #Recalculation of s into a.u.
    thzPulse[:,1]   *= 1e-8*E_conv                                     #Recalculation of V/m into MV/cm
    initThz         = [1, 1000*fs_conv, 0, 1*THz_conv, 0]
    tOpt, tCov      = optimize.curve_fit(transient, thzPulse[:,0], thzPulse[:,1], p0=initThz)

    #Load experimental frequency spectrum
    pulseFreq       = np.loadtxt("NIR_spectrum.dat")
    pulseFreq[:,0]  = vel_light/(pulseFreq[:,0]*1e-9)*1e-12*THz_conv
    pulseFreq[:,1]  *= E_conv
    peak            = pulseFreq[np.argmax(pulseFreq[:,1] ) ]
    initNir         = [peak[1], 100*THz_conv, peak[0]]
    fOpt, fCov      = optimize.curve_fit(gaussSpec, pulseFreq[:,0], np.abs(pulseFreq[:,1]), p0=initNir )

    nOpt            = list(fOpt[:] )
    nOpt[0]         = fOpt[0]*fOpt[1]
    nOpt[1]         = 1/nOpt[1]
    nOpt.append(nOpt[2])
    nOpt[2]         = 0
    nOpt.append(0)

    return [tOpt, nOpt]

@njit
def transient(x, aT, sigmaT, muT, freqT, chirpT):
    return aT*np.exp(-((x-muT)/sigmaT)**2/2)*np.cos(2*np.pi*(1+chirpT*x)*freqT*x)

@njit
def nir(x, aN, sigmaN, muN, freqN, phiN):
    return aN*np.exp(-(x-muN)**2/sigmaN**2/2)*np.cos(2*np.pi*freqN*(x-muN)+phiN )

def gaussSpec(x, A, sigma, mu):
    return A*np.exp(-(2*np.pi*(x-mu)/sigma)**2/2)


