import params
import numpy as np
import matplotlib.pyplot as pl

import data_directory
import os

from matplotlib import ticker, cm
from scipy.interpolate import interp1d


def main():
    fs_conv  = params.fs_conv
    E_conv   = params.E_conv
    B_conv   = params.B_conv
    THz_conv = params.THz_conv
    amp_conv = params.amp_conv
    eV_conv  = params.eV_conv

    # Set BZ type independent parameters
    # Hamiltonian parameters
    C0 = params.C0                                    # Dirac point position
    C2 = params.C2                                    # k^2 coefficient
    A = params.A                                      # Fermi velocity
    R = params.R                                      # k^3 coefficient
    k_cut = params.k_cut                              # Model hamiltonian cutoff parameter

    # System parameters
    a = params.a                                      # Lattice spacing
    e_fermi = params.e_fermi*eV_conv                  # Fermi energy for initial conditions
    temperature = params.temperature*eV_conv          # Temperature for initial conditions
    T       = 1*eV_conv
    
    kF      = np.arccos(1-e_fermi/(2*T))/a

    # Driving field parameters
    E0    = params.E0*E_conv                          # Driving pulse field amplitude
    B0    = params.B0*B_conv                          # Driving pulse magnetic field amplitude
    chirp = params.chirp*params.THz_conv                     # Pulse chirp frequency
    phase = params.phase                              # Carrier-envelope phase
    
    ####### changes on the fitted parameters
    w     = params.w*params.THz_conv                         # Driving pulse frequency
    alpha = params.alpha*params.fs_conv                      # Gaussian pulse width


    # Time scales T1 = params.T1*fs_conv                            # Occupation damping time
    T1 = params.T1*fs_conv                            # Polarization damping time
    T2 = params.T2*fs_conv                            # Polarization damping time
    gamma1 = 1/T1                                     # Occupation damping parameter
    gamma2 = 1/T2                                     # Polarization damping parameter
    t0 = int(params.t0*fs_conv)                       # Initial time condition
    tf = int(params.tf*fs_conv)                       # Final time
    phi = np.pi*0.5
    
    nir_values  = False
    if nir_values:
        nOpt        = np.loadtxt("fitting_data/nir_parameters.txt")
        E0      = nOpt[0]
        alpha   = nOpt[1]
        w       = nOpt[3]

    alpha   *= 1
    w       *= 1
    E0      *= 1
    kF      *= 1
    c       = 0.0e-1*w
    gamma   = gamma2

    print(w*alpha)
   
    real_fig, (axB,axA,axC,axI) = pl.subplots(4,1,figsize=(10,10))

    kx_array = np.linspace(-np.pi/a, np.pi/a, 1000)
    axB.plot(kx_array, conduction_band(kx_array, T, a) )
    np.savetxt("/home/nim60855/Documents/masterthesis/thesis/bericht/document/chapters/data/cosine_conduction.dat", np.transpose([kx_array/(np.pi/a), conduction_band(kx_array, T, a)/(T)] ) )
    #axB2    = axB.twinx()

    kx_array = np.linspace(-kF, kF, 1000)
    axB.plot(kx_array, np.ones(kx_array.size)*e_fermi)
    np.savetxt("/home/nim60855/Documents/masterthesis/thesis/bericht/document/chapters/data/cosine_fermi-contour.dat", np.transpose([kx_array/(np.pi/a), np.ones(kx_array.size)*e_fermi/(T)] ) )
    print(e_fermi/T)

    max_x   = 17
    dt = 1/(max_x*2*w)

    number      = 10
    t           = np.arange(-number*alpha, number*alpha, dt)
    #axB2.plot(A_field(t, E0, w, alpha, phi, c), t)

    current     = current_analyt_A(t, T, kF, a, E0, w, alpha, phi, c)

    axA.plot(t, A_field(t, E0, w, alpha, phi, c), label="analytical A-field" )
    axA.plot(t, A_field_exact(t, E0, w, alpha, phi, c), label="numerical A-field" )
    axC.plot(t/fs_conv, current, label="current analytical A-field")
    axC.plot(t/fs_conv, current_exact_A(t, T, kF, a, E0, w, alpha, phi, c), label="current numerical A-field")

    dt_out      = t[1]-t[0]
    freq        = np.fft.fftfreq(np.size(t), d=dt_out)
    Int_E_dir   = np.abs(freq*np.fft.fft(current ) )**2
    Int_E_dir_ex   = np.abs(freq*np.fft.fft(current_exact_A(t, T, kF, a, E0, w, alpha, phi, c) ) )**2

    freq_indices        = np.where(freq/w > 0)[0]
    freq        = freq[freq_indices]
    Int_E_dir   = Int_E_dir[freq_indices]
    Int_E_dir_ex   = Int_E_dir_ex[freq_indices]
    x           = freq/w

    freq_indices_near_base_freq = np.argwhere(np.logical_and(freq/w > 0.9, freq/w < 1.1))
    freq_index_base_freq    = int((freq_indices_near_base_freq[0] + freq_indices_near_base_freq[-1])/2)
    normalisation           = Int_E_dir[freq_index_base_freq]
    axI.plot(freq/w, Int_E_dir/normalisation, label="analytical A-field")
    axI.plot(freq/w, Int_E_dir_ex/normalisation, label="numerical A-field")

    pl.legend()
    axA.legend(loc="upper right")
    axC.legend(loc="upper right")
    pl.xticks(np.arange(1, max_x, 2) )
    pl.grid(True)
    pl.xlim((.1, max_x) )
    #pl.ylim((1e-12, 1e-6) )
    axI.set_yscale("log")
    pl.show()
    pl.clf()
    pl.close()
    return

    fig, ((axA,axN), (axW,axP)) = pl.subplots(2,2,figsize=(10,10))

    angles              = np.linspace(0, np.pi, 1*t.size)
    emission_cep_n      = np.array([np.abs(freq*np.fft.fft(A_field_exact(t, E0, w, alpha, phi, c)**13 )[freq_indices] )**2 for phi in angles])
    emission_cep_n      = np.array([np.abs(freq*np.fft.fft(current_exact_A(t, T, kF, a, E0, w, alpha, phi, c) )[freq_indices] )**2 for phi in angles])

    X, Y = np.meshgrid(freq/w, angles)
    imN = axN.contourf(X, Y, np.log10(emission_cep_n), levels=50)

    paths   = trace_cep(freq, emission_cep_n, angles, w)
    plot_winding(freq/w, paths, angles, axN, axW, label="Numerical CEP-tracing", color="b")

    paths   = trace_max(freq, emission_cep_n, angles, w)
    occurences  = np.unique(paths[:,-1], return_counts=True, return_index=True)
    #paths       = paths[occurences[1][occurences[2]==1] ]
    plot_winding(freq/w, paths, angles, axN, axW, label="Numerical Maxima-tracing", color="gray")

    axW.plot(x, 2*x*np.pi*(1+gamma**2)/(4*gamma*(alpha*2*np.pi*w)**2), color="k")

    for l in range(2, max_x+1, 1):
        peaks           = (l-1)-2*l*angles*(1+gamma**2)/(4*gamma*(alpha*2*np.pi*w)**2)
        axN.plot(peaks, angles, color="k")

    fig.colorbar(imN, ax=axN)

    axN.set_xlabel(r'$\frac{w}{w_0}$')
    axN.set_ylabel(r'$\phi$')
    axW.set_xlabel(r'$\frac{w}{w_0}$')
    axW.set_ylabel("Winding number")
    axP.set_xlabel(r'$\frac{w}{w_0}$')
    axP.set_ylabel('c')

    axA.set_xticks(np.arange(1, freq[-1]/w, 1))
    axN.set_xticks(np.arange(1, freq[-1]/w, 1))
    axW.set_xticks(np.arange(1, freq[-1]/w, 1))
    axP.set_xticks(np.arange(1, freq[-1]/w, 1))

    axA.grid(True)
    axW.grid(True)
    axW.grid(True)

    axA.legend()
    axN.legend()
    axW.legend()

    axA.set_xticks(np.arange(1, freq[-1]/w, 1))
    axN.set_xticks(np.arange(1, freq[-1]/w, 1))
    axW.set_xticks(np.arange(1, freq[-1]/w, 1))
    axP.set_xticks(np.arange(1, freq[-1]/w, 1))

    pl.tight_layout()
    pl.show()
    return



def conduction_band(kx, T, a):
    return -2*T*(np.cos(a*kx) -1)

def current_analyt_A(t, T, kF, a, E0, w, alpha, phi, c):
    return -2*T/np.pi*np.sin(a*kF)*np.sin(a*A_field(t, E0, w, alpha, phi, c) )

def current_exact_A(t, T, kF, a, E0, w, alpha, phi, c):
    return -2*T/np.pi*np.sin(a*kF)*np.sin(a*A_field_exact(t, E0, w, alpha, phi, c) )

def A_field(t, E0, w, alpha, phi, c):
    return E0/(2*np.pi*w)*np.exp(-(t/(2*alpha) )**2)*np.sin(2*np.pi*(1+c*t)*w*t + phi)

def A_field_exact(t, E0, w, alpha, phi, c):
    result  = np.cumsum(E_field(t, E0, w, alpha, phi, c))*(t[1]-t[0])
    return result*(0+1*np.exp(-(t/(4*alpha))**2) )

def E_field(t, E0, w, alpha, phi, c):
    return E0*np.exp(-(t/(2*alpha) )**2)*np.cos(2*np.pi*(1+c*t)*w*t + phi)

def trace_max(freq, Int, angles, w):
    dx          = (freq[1]-freq[0])/w
    epsilon     = dx/2
    lattice     = freq/w
    neighbors       = np.array([-1, 0, 1])                     # Take the three neigbors which are the nearest to the previous point

    paths       = []
    for k in range(3, Int[0].size-3, 1):
        if np.argmax(Int[0, k+neighbors] ) != 1:
            continue
        
        path        = [lattice[k] ]
        for i, phi in enumerate(angles):
            if i == len(angles)-1:
                paths.append(np.array(path) )
                continue
            
            k               += np.argmax(Int[i+1, k+neighbors] )-1
            if (k < 0) or (k > lattice.size-2):
                break
            path.append(lattice[k] )

    paths       = np.array(paths)
    return paths

def trace_cep(freq, Int, angles, w):
    dx          = (freq[1]-freq[0])/w
    epsilon     = dx/2
    lattice     = freq/w

    paths       = []

    for k0 in range(1, Int[0].size):
        x           = freq[k0]/w
        path        = [x]
        emis_x      = Int[0, k0]
        for i, phi in enumerate(angles):
            if i == len(angles)-1:
                paths.append(np.array(path) )
                continue
            
            neighbors       = np.where(np.abs(lattice -x) < 1.5*dx)[0]          # Take the three neigbors which are the nearest do the previous point
            nearest         = neighbors[1]

            to_close        = np.abs(lattice[nearest] - x) < epsilon            # Check if the middle one is to close to the previos point
            if to_close:
                neighbors   = [neighbors[0], neighbors[-1]]                     # If he is to close discard it
            else:
                neighbors  = np.where(np.abs(lattice -x) < dx)[0]               # If not, discard the point, which is the farest of the three neigbors

            if np.all(Int[i+1, neighbors] < emis_x) or np.all(Int[i+1, neighbors] > emis_x):
                break                                                               #If this is true then the path is broken

            m       = (Int[i+1, neighbors[0] ] - Int[i+1, neighbors[1] ])/(lattice[neighbors[0] ] - lattice[neighbors[1] ] )
            x       = (emis_x - Int[i+1, neighbors[0] ] )/m + lattice[neighbors[0] ]
            path.append(x)

    paths       = np.array(paths)
    return paths

def plot_winding(x, paths, angles, axA, axW, label="", color="k"):
    for i, path in enumerate(paths):
        if i == 0:
            axA.plot(path, angles, color=color, label=label)
        else:
            axA.plot(path, angles, color=color)

    winding = winding_number(x, paths)
    axW.plot(x, winding, label=label )
    return

def winding_number(x, paths):
    winding = paths[:,0] - paths[:,-1]
    winding_tilde   = np.insert(winding, 0, 0)
    path_tilde      = np.insert(paths[:,0], 0, 0)

    last_point      = x[-1]
    path_tilde      = np.append(path_tilde, last_point)
    winding_tilde   = np.append(winding_tilde, winding_tilde[-1])

    f       = interp1d(path_tilde, winding_tilde, kind="linear" )
    winding = f(x)
    return winding



if __name__ == "__main__":
    main()
