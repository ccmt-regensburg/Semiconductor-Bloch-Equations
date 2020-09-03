import numpy as np
from numba import njit
from scipy import optimize

import params
import matplotlib.pyplot as pl

def opt_pulses():
    #Prepare pyplot axes
    fs_conv         = params.fs_conv
    E_conv          = params.E_conv
    THz_conv        = params.THz_conv
    vel_light       = params.c
    transient_number    = params.transient_number

    #Load THz Pulse data
    thzPulse        = np.loadtxt("fitting_data/Transient_" + str(transient_number) + ".dat")
    if transient_number == 0:
        factor_field = 1e-8
        factor_time  = 1e15
    else:
        factor_field = 1e-3
        factor_time  = 1

    thzPulse[:,0]   *= factor_time*fs_conv                             #Recalculation of s into a.u.
    thzPulse[:,1]   *= factor_field*E_conv                                     #Recalculation of V/m into MV/cm
    initThz         = [1e-4, 1000*fs_conv, 0, 1*THz_conv, 0]
    tOpt, tCov      = optimize.curve_fit(transient, thzPulse[:,0], thzPulse[:,1], p0=initThz)
    print(tOpt[0]/E_conv)
    print(tOpt[-2]/THz_conv)

    #Load experimental frequency spectrum
    pulseFreq       = np.loadtxt("fitting_data/NIR_spectrum.dat")
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
    print(nOpt[3]/THz_conv)
    nOpt.append(0)
    np.savetxt("fitting_data/transient_" + str(transient_number) + "_parameters.txt", tOpt, header="amplitude, sigma, mu, frequency, chirp" )
    np.savetxt("fitting_data/nir_parameters.txt", nOpt, header="amplitude, sigma, mu, frequency, phi" )
    pl.plot(thzPulse[:,0], thzPulse[:,1], linestyle="dashed", label="Data of the Transient " + str(transient_number))
    pl.plot(thzPulse[:,0], transient(thzPulse[:,0], *tOpt), label="Fit of the data")
    pl.legend()
    pl.xlabel("Time in at.u.")
    pl.ylabel("Electrical field in at.u.")
    pl.grid(True)
    pl.show()

    return [tOpt, nOpt]

@njit
def transient(x, aT, sigmaT, muT, freqT, chirpT):
    return aT*np.exp(-((x-muT)/sigmaT)**2/2)*np.cos(2*np.pi*(1+chirpT*x)*freqT*x)

@njit
def nir(x, aN, sigmaN, muN, freqN, phiN):
    return aN*np.exp(-(x-muN)**2/sigmaN**2/2)*np.cos(2*np.pi*freqN*(x-muN)+phiN )

def gaussSpec(x, A, sigma, mu):
    return A*np.exp(-(2*np.pi*(x-mu)/sigma)**2/2)


