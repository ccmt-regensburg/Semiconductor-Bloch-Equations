from numba import njit
import numpy as np
import params 
import nir

# Driving field parameters

w     = params.w*params.THz_conv                         # Driving pulse frequency
chirp = params.chirp*params.THz_conv                     # Pulse chirp frequency
alpha = params.alpha*params.fs_conv                      # Gaussian pulse width
phase = params.phase                              # Carrier-envelope phase

fitted_pulse   = params.fitted_pulse
parameters = nir.opt_pulses()

if fitted_pulse:
    print("Broadening Gauss [fs] =", parameters[0]/params.fs_conv  )
    print("Time shift [fs]       =", parameters[1]/params.fs_conv  )
    print("Frequency [THz]       =", parameters[2]/params.THz_conv )
    print("Chirp [THz]           =", parameters[3]/params.THz_conv )
    print("Phase                 =", parameters[4])

@njit
def driving_field(Amplitude, t):
    '''
    Returns the instantaneous driving pulse field
    '''
    # Non-pulse
    # return E0*np.sin(2.0*np.pi*w*t)
    # Chirped Gaussian pulse
    if fitted_pulse:
        return Amplitude*nir.transient(t, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5])

    else:
        return Amplitude*np.exp(-t**2.0/(2.0*alpha)**2)*np.sin(2.0*np.pi*w*t*(1 + chirp*t) + phase)

