from numba import njit
import numpy as np
import params 

# Driving field parameters

w     = params.w*params.THz_conv                         # Driving pulse frequency
chirp = params.chirp*params.THz_conv                     # Pulse chirp frequency
alpha = params.alpha*params.fs_conv                      # Gaussian pulse width
phase = params.phase                              # Carrier-envelope phase

@njit
def driving_field(Amplitude, t):
    '''
    Returns the instantaneous driving pulse field
    '''
    # Non-pulse
    # return E0*np.sin(2.0*np.pi*w*t)
    # Chirped Gaussian pulse
    return Amplitude*np.exp(-t**2.0/(2.0*alpha)**2)\
        * np.sin(2.0*np.pi*w*t*(1 + chirp*t) + phase)

