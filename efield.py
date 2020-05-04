from numba import njit
import numpy as np
import params 
import nir

# Driving field parameters

w     = params.w*params.THz_conv                         # Driving pulse frequency
chirp = params.chirp*params.THz_conv                     # Pulse chirp frequency
alpha = params.alpha*params.fs_conv                      # Gaussian pulse width
phase = params.phase                              # Carrier-envelope phase

parameters = nir.opt_pulses()

@njit
def driving_field(Amplitude, t):
    '''
    Returns the instantaneous driving pulse field
    '''
    # Non-pulse
    # return E0*np.sin(2.0*np.pi*w*t)
    # Chirped Gaussian pulse
    return Amplitude*nir.transient(t, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5])
    #return Amplitude*np.exp(-t**2.0/(2.0*alpha)**2)\
    #    * np.sin(2.0*np.pi*w*t*(1 + chirp*t) + phase)

