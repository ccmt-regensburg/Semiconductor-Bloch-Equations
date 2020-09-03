from numba import njit
import numpy as np
import params 

import scipy.integrate as integrate
from scipy.special import erf

# Driving field parameters

w     = params.w*params.THz_conv                         # Driving pulse frequency
chirp = params.chirp*params.THz_conv                     # Pulse chirp frequency
alpha = params.alpha*params.fs_conv                      # Gaussian pulse width
phase = params.phase                              # Carrier-envelope phase

fitted_pulse        = params.fitted_pulse
transient_number    = params.transient_number
tOpt                = np.loadtxt("fitting_data/transient_" + str(transient_number) + "_parameters.txt") 
nOpt                = np.loadtxt("fitting_data/nir_parameters.txt")
nOpt[2]             = params.nir_mu*params.fs_conv


#nOpt            = tOpt
nOpt[0]         *= params.nir_fac
tOpt[0]         *= params.tra_fac
#nOpt[4]         = 0
#nOpt[3]         *= 4

a   = nOpt[0]
b   = nOpt[1]
c   = nOpt[2]
d   = nOpt[3]
e   = nOpt[4]

nir_E0      = nOpt[0]
nir_sigma   = nOpt[1]
nir_mu      = nOpt[2]
nir_w       = nOpt[3]
nir_phi     = nOpt[4]

aT      = tOpt[0]
sigmaT  = tOpt[1]
muT     = tOpt[2]
freqT   = tOpt[3]
chirpT  = tOpt[4]

with_transient  = params.with_transient
with_nir        = params.with_nir

#if fitted_pulse:
#    parameters = nir.opt_pulses()
#
#    print("Amplitude (without unit) =", parameters[0] )
#    print("Broadening Gauss [fs]    =", parameters[1]/params.fs_conv  )
#    print("Time shift [fs]          =", parameters[2]/params.fs_conv  )
#    print("Frequency [THz]          =", parameters[3]/params.THz_conv )
#    print("Chirp [THz]              =", parameters[4]/params.THz_conv )
#    print("Phase                    =", parameters[5] )

@njit
def driving_field(Amplitude, t):
    '''
    Returns the instantaneous driving pulse field
    '''
    # Non-pulse
    # return E0*np.sin(2.0*np.pi*w*t)
    # Chirped Gaussian pulse
    if fitted_pulse:
        if with_transient and with_nir:
            return transient(t) + nir(t)
        elif with_transient:
            return transient(t)
        elif with_nir:
            return nir(t)
        else:
            return 0

    else:
        return Amplitude*np.exp(-t**2.0/(2.0*alpha)**2)*np.sin(2.0*np.pi*w*t*(1 + chirp*t) + phase)

@njit
def transient(x):
    return aT*np.exp(-((x-muT)/sigmaT)**2/2)*np.cos(2*np.pi*(1+chirpT*x)*freqT*x)

@njit
def nir(x):
    return a*np.exp(-(x-c)**2/b**2/2)*np.cos(2*np.pi*d*(x-c)+e )

@njit
def A_nir(x):
    return -a/(2*np.pi*d)*np.exp(-(x-c)**2/b**2/2)*np.sin(2*np.pi*d*(x-c)+e )

@njit
def A_transient(x):
    return aT/(2*np.pi*freqT)*np.exp(-((x-muT)/sigmaT)**2/2)*np.cos(2*np.pi*(1+chirpT*x)*freqT*x)

def simple_transient(x):
    return 100*np.exp(-((x-tOpt[2])/tOpt[1])**2/2)*np.cos(2*np.pi*(1+tOpt[4]*x)*tOpt[3]*x)

def simple_A_field(t):
    return tOpt[0]/(2*np.pi*tOpt[3])*np.exp(-((t-tOpt[2])/tOpt[1])**2/2)*np.sin(2*np.pi*(1+tOpt[4]*t)*tOpt[3]*t)

    return np.real(2*(2+erf((t-1j*tOpt[1]**2*tOpt[3]*2*np.pi)/(np.sqrt(2)*tOpt[1]))+erf((t+1j*tOpt[1]**2*tOpt[3]*2*np.pi)/(np.sqrt(2)*tOpt[1])) ) )


