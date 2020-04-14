#analytical solution for flat bands
import params
import numpy as np
import scipy as sp

delta_min           = params.delta_min*params.eV_conv
omega               = 2*np.pi*params.w*params.THz_conv
width_gaussian      = np.sqrt(2)*params.alpha*params.fs_conv
amplitude_elec      = params.E0*params.E_conv
damping_off         = 1/(params.T2*params.fs_conv)
phase               = params.phase - np.pi/2
delta_min           = params.delta_min*params.eV_conv                # Minimal energy gap in a.u.
delta_d             = params.delta_d*params.eV_conv                    # Difference min and max energy gap in a.u.



def emission():
    samplingPoints = 5000
    factor = int(delta_min/omega)+1
    factor *= 2
    maxOrder = int(factor/2)+1
    frequencies = np.linspace(0.2*omega, factor*omega, samplingPoints)
    erg = np.zeros(samplingPoints, dtype=complex)
    for h in range(maxOrder+1):
        erg += hilfsfct(h, frequencies)
    erg = erg*(0+1*(frequencies)**2)
    erg = np.real(erg*np.conj(erg) )
    erg = erg/max(erg)
    return np.transpose([frequencies/omega, erg])

def hilfsfct(h, w):
    summe = np.zeros(len(w), dtype=complex)
    erg = np.power(delta_min,2)/(np.power(delta_min,2)+np.power(damping_off+1j*w,2))*np.power(-1,h)*sp.special.factorial2(2*h-1)/sp.special.factorial2(2*h)*np.power(amplitude_elec/delta_min,2*h+1)*width_gaussian/np.sqrt(2*h+1)

    for k in range(0, h+1):
        summe += sp.special.binom(2*h+1, k)*np.exp(-np.power(width_gaussian*(w-(2*(h-k)+1)*omega),2)/(2*(2*h+1)))*np.exp(-1j*(2*(h-k)+1)*phase)

    return erg*summe



