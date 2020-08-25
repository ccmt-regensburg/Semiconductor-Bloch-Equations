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

    # Driving field parameters
    E0    = params.E0*E_conv                          # Driving pulse field amplitude
    B0    = params.B0*B_conv                          # Driving pulse magnetic field amplitude
    chirp = params.chirp*params.THz_conv                     # Pulse chirp frequency
    phase = params.phase                              # Carrier-envelope phase
    
    ####### changes on the fitted parameters
    w     = params.w*params.THz_conv                         # Driving pulse frequency
    alpha = params.alpha*params.fs_conv                      # Gaussian pulse width

    kF  = e_fermi/A

    # Time scales T1 = params.T1*fs_conv                            # Occupation damping time
    T1 = params.T1*fs_conv                            # Polarization damping time
    T2 = params.T2*fs_conv                            # Polarization damping time
    gamma1 = 1/T1                                     # Occupation damping parameter
    gamma2 = 1/T2                                     # Polarization damping parameter
    t0 = int(params.t0*fs_conv)                       # Initial time condition
    tf = int(params.tf*fs_conv)                       # Final time
    phi = np.pi*0.0
    
    ky      = .0*kF

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
    c       = 1.0e-1*w

   
    real_fig, (axB,axA,axC,axI) = pl.subplots(4,1,figsize=(10,10))

    kx_array = np.linspace(-np.pi/a, np.pi/a, 1000)
    axB.plot(kx_array, conduction_band(kx_array, ky, A) )

    kx_array = np.linspace(-kF, kF, 1000)
    axB.plot(kx_array, np.ones(kx_array.size)*e_fermi)

    max_x   = 25
    dt = 1/(max_x*2*w)

    number      = 15
    t           = np.arange(-number*alpha, number*alpha, dt)
    int_current = integrated_current_exact_A(t, A, kF, E0, w, alpha, phi, c)

#    t_num   = 4096
#    k_num   = 800
#    t_lim   = 200*fs_conv
#    t           = np.linspace(-t_lim, t_lim, t_num)
#    c       = -0.920*THz_conv
#
#    A_t     = A_field_exact(t, E0, w, alpha, phi, c)
#    kMax    = 3/2*np.pi/a
#    kPoints = np.linspace(-kMax, kMax, k_num)
#    
#    density = np.zeros((t_num, k_num) )
#
#    for k in range(t_num):
#        ind_t   = np.where( np.abs(kPoints + A_t[k] ) < (0.991446)*kF)
#        density[k, ind_t ]  = 1
#    print(np.sum(density, axis=1) )
#    print(max(np.sum(density, axis=1) ) )
#    print(min(np.sum(density, axis=1) ) )
#
#    print(current_exact(t, ky, A, kF, E0, w, alpha, phi, c).shape)
#    print(density.shape)
#    np.save("current_analytical.npy", current_exact(t, ky, A, kF, E0, w, alpha, phi, c) )
#    np.save("density_analytical.npy", density )

    axA.plot(t, A_field(t, E0, w, alpha, phi, c), label="Simple A-field" )
    axA.plot(t, A_field_exact(t, E0, w, alpha, phi, c), label="Numerical exact A-field" )
    #axA.plot(t, A_field_tau(t, -2/(w), E0, w, alpha, phi, c), label=r"Simple $A_{\frac{1}{2w}}$-field" )
    axC.plot(t/fs_conv, int_current, label="Numerically integrated current")
    axC.plot(t/fs_conv, appr_int_current(t, A, kF, E0, w, alpha, phi, c), label="Numerically integrated analytical current")
    #axC.plot(t, signum_steps(t, E0, w, alpha, ky, kF, A, phi, c) )

    dt_out      = t[1]-t[0]
    freq        = np.fft.fftshift(np.fft.fftfreq(np.size(t), d=dt_out))

    Int_E_dir   = np.abs(freq*np.fft.fftshift(np.fft.fft(current(t, ky, A, kF, E0, w, alpha, phi, c) ) ) )**2
    Int_E_dir   = np.abs(freq*np.fft.fftshift(np.fft.fft(int_current ) ) )**2

    freq_indices_near_base_freq = np.argwhere(np.logical_and(freq/w > 0.9, freq/w < 1.1))
    freq_index_base_freq    = int((freq_indices_near_base_freq[0] + freq_indices_near_base_freq[-1])/2)
    normalisation           = Int_E_dir[freq_index_base_freq]
    axI.plot(freq/w, Int_E_dir/normalisation, label="numerical result of the emission")

    freq        = freq[int(len(freq)/2 )+1:]
    x           = freq/w

    freq_indices_near_base_freq = np.argwhere(np.logical_and(freq/w > 0.9, freq/w < 1.1))
    freq_index_base_freq    = int((freq_indices_near_base_freq[0] + freq_indices_near_base_freq[-1])/2)
    analyt_Int_E_dir        = np.abs(freq*analytical_fourier(freq, E0, w, alpha, ky, kF, A, phi, c) )**2
    normalisation_an        = analyt_Int_E_dir[freq_index_base_freq]
    axI.plot(freq/w, analyt_Int_E_dir/normalisation_an, label="Analytical fourier transformation")

    decay       = (w/freq)**2
    axI.plot(freq/w, decay, label="Assumed decay of the emission")

    sbe_numerics    = False
    if sbe_numerics:
        directory           = data_directory.change_to_data_directory()
        if os.path.exists(directory):
            os.chdir(directory)
            freq, Int_exact_E_dir, Int_exact_ortho, Int_exact_diag_E_dir, Int_exact_diag_ortho, Int_exact_offd_E_dir, Int_exact_offd_ortho  = np.transpose(np.loadtxt('frequency.txt') )
            conv_factor         = max(analyt_Int_E_dir)/max(Int_exact_E_dir)
            axI.plot(freq, Int_exact_E_dir*conv_factor, label="Result SBE-numerics")
        else:
            print("There is no data from SBE simulations available.")

    pl.legend()
    axA.legend(loc="upper right")
    axC.legend(loc="upper right")
    pl.xticks(np.arange(1, max_x, 2) )
    pl.grid(True)
    pl.xlim((.1, max_x) )
    #pl.ylim((1e-12, 1e-6) )
    pl.yscale("log")
    pl.show()
    pl.clf()
    pl.close()

    freq_indices        = np.where(np.abs(freq/w ) > 0)[0]
    freq                = freq[freq_indices]
    angles              = np.linspace(0, np.pi, 1*t.size)

    fig, ((axA,axN), (axW,axP)) = pl.subplots(2,2,figsize=(10,10))

    emission_cep_a      = np.array([np.abs(freq*analytical_fourier(freq, E0, w, alpha, ky, kF, A, phi, c) )**2 for phi in angles])
    emission_cep_n      = np.array([np.abs(freq*np.fft.fft(integrated_current_exact_A(t, A, kF, E0, w, alpha, phi, c) )[:freq.size] )**2 for phi in angles])
    emission_cep_a      = emission_cep_a[:,freq_indices]
    emission_cep_n      = emission_cep_n[:,freq_indices]

    X, Y = np.meshgrid(freq/w, angles)

    imA = axA.contourf(X, Y, np.log10(emission_cep_a), levels=50)
    imN = axN.contourf(X, Y, np.log10(emission_cep_n), levels=50)

    axW.plot(x, 2*c/(2*np.pi*w)*x*np.pi, color="k")
    paths   = trace_cep(freq, emission_cep_a, angles, w)
    plot_winding(freq/w, paths, angles, axA, axW, label="Analytical CEP-tracing", color="b")

    paths   = trace_max(freq, emission_cep_a, angles, w)
    occurences  = np.unique(paths[:,-1], return_counts=True, return_index=True)
    paths       = paths[occurences[1][occurences[2]==1] ]
    plot_winding(freq/w, paths, angles, axA, axW, label="Analytical Maxima-tracing", color="r")

    paths   = trace_cep(freq, emission_cep_n, angles, w)
    plot_winding(freq/w, paths, angles, axN, axW, label="Numerical CEP-tracing", color="b")

    paths   = trace_max(freq, emission_cep_n, angles, w)
    occurences  = np.unique(paths[:,-1], return_counts=True, return_index=True)
    paths       = paths[occurences[1][occurences[2]==1] ]
    plot_winding(freq/w, paths, angles, axN, axW, label="Numerical Maxima-tracing", color="gray")

    for l in range(1, max_x+1):
        peaks           = l*(1-c/w*angles/np.pi)
        axA.plot(peaks, angles, color="k")
        axN.plot(peaks, angles, color="k")

    fig.colorbar(imA, ax=axA)
    fig.colorbar(imN, ax=axN)

    axA.set_xlabel(r'$\frac{w}{w_0}$')
    axA.set_ylabel(r'$\phi$')
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

    paths   = trace_cep(freq, emission_cep_n, angles, w)

    axA.set_xticks(np.arange(1, freq[-1]/w, 1))
    axN.set_xticks(np.arange(1, freq[-1]/w, 1))
    axW.set_xticks(np.arange(1, freq[-1]/w, 1))
    axP.set_xticks(np.arange(1, freq[-1]/w, 1))

    pl.tight_layout()
    pl.show()
    return

    chirps, phase_data  = phase_plot_data(t, freq/w, A, kF, E0, w, alpha, phi, c, angles, c_number=10)
    X, C                = np.meshgrid(freq/w, chirps)
    imP                 = axP.contourf(X, C, phase_data, levels=50)
    fig.colorbar(imP, ax=axP)
    #pl.show()

    return
 
def phase_plot_data(t, x, A, kF, E0, w, alpha, phi, c_max, angles, c_number=10):
    chirps  = np.linspace(c_max/3, c_max, c_number)
    result  = []
    for k,c in enumerate(chirps):
        print(k)
        #emission_cep    = np.array([np.abs(w*x*analytical_fourier(w*x, E0, w, alpha, ky, kF, A, phi, c) )**2 for phi in angles])
        emission_cep    = np.array([np.abs(x*w*np.fft.fft(integrated_current_exact_A(t, A, kF, E0, w, alpha, phi, c) )[:x.size] )**2 for phi in angles])
        paths           = trace_cep(x, emission_cep, angles, 1)
        winding         = winding_number(x, paths)
        result.append(winding)

    return chirps/w, result

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

def appr_int_current(t, A, kF, E0, w, alpha, phi, c):
    paths               = np.linspace(-kF, kF, 100)
    dky                 = paths[1] - paths[0]
    integrated_current  = np.array([current(t, 0, A, np.sqrt(kF**2 - ky**2), E0, w, alpha, phi, c) for ky in paths] )
    integrated_current  = np.sum(integrated_current, axis=0)*dky/(2*np.pi)
    return integrated_current

def integrated_current(t, A, kF, E0, w, alpha, phi, c):
    paths               = np.linspace(-kF, kF, 100)
    dky                 = paths[1] - paths[0]
    integrated_current  = np.array([current(t, ky, A, kF, E0, w, alpha, phi, c) for ky in paths] )
    integrated_current  = np.sum(integrated_current, axis=0)*dky/(2*np.pi)
    return integrated_current

def integrated_current_exact_A(t, A, kF, E0, w, alpha, phi, c):
    paths               = np.linspace(-kF, kF, 100)
    dky                 = paths[1] - paths[0]
    integrated_current  = np.array([current_exact(t, ky, A, kF, E0, w, alpha, phi, c) for ky in paths] )
    integrated_current  = np.sum(integrated_current, axis=0)*dky/(2*np.pi)
    return integrated_current

def current(t, ky, A, kF, E0, w, alpha, phi, c):
    return -A/(2*np.pi) * (np.sqrt(-ky**2 + (kxMax(ky, kF) - A_field(t, E0, w, alpha, phi, c))**2 ) - np.sqrt(-ky**2 + (kxMax(ky, kF) + A_field(t, E0, w, alpha, phi, c))**2 ) )

def current_exact(t, ky, A, kF, E0, w, alpha, phi, c):
    return -A/(2*np.pi) * (np.sqrt(ky**2 + (kxMax(ky, kF) - A_field_exact(t, E0, w, alpha, phi, c))**2 ) - np.sqrt(ky**2 + (kxMax(ky, kF) + A_field_exact(t, E0, w, alpha, phi, c))**2 ) )

def A_field_exact(t, E0, w, alpha, phi, c):
    result  = np.cumsum(E_field(t, E0, w, alpha, phi, c))*(t[1]-t[0])
    return result*(0+1*np.exp(-(t/(4*alpha))**2) )

def A_field(t, E0, w, alpha, phi, c):
    return E0/(2*np.pi*w)*np.exp(-(t/(2*alpha) )**2)*np.sin(2*np.pi*(1+c*t)*w*t + phi)

def A_field_tau(t, tau, E0, w, alpha, phi, c):
    return E0/(2*np.pi*w*(1+2*c*tau))*np.exp(-(t/(2*alpha) )**2)*np.sin(2*np.pi*(1+c*t)*w*t + phi)

def E_field(t, E0, w, alpha, phi, c):
    return E0*np.exp(-(t/(2*alpha) )**2)*np.cos(2*np.pi*(1+c*t)*w*t + phi)

def kxMax(ky, kF):
    return np.sqrt(kF**2 - ky**2)

def conduction_band(kx, ky, A):
    return A*np.sqrt(kx**2+ky**2)

def signum_steps(t, E0, w, alpha, ky, kF, A, phi, c):
    kMax        = kxMax(ky, kF)
    Amp         = E0/(2*np.pi*w)
    t_critical  = alpha*np.sqrt(2*np.log(Amp/kMax) )
    t_range     = (np.abs(t) < t_critical).astype(int)
    return -2*A/(2*np.pi) * np.sign(A_field(t, E0, w, alpha, phi, c) )*kMax*t_range

def analytical_fourier(freq, E0, w, alpha, ky, kF, A, phi, c):
    nat_w       = freq/w
    kMax        = kxMax(ky, kF)
    Amp         = E0/(2*np.pi*w)
    t_critical  = alpha*np.sqrt(2*np.log(Amp/kMax) )
    max_ind     = int(2*t_critical*w + phi/np.pi )-0
    min_ind     = int(-2*t_critical*w + phi/np.pi )+0
    better_result   = 1j*np.zeros(freq.size)
    for k in range(min_ind, max_ind+1):
        better_result   += newHilfsfct(freq, E0, w, alpha, kMax, A, phi, c, k) 

    return np.sqrt(2/np.pi)*A/np.pi*E0*better_result/freq*w*4*np.pi
    return np.sqrt(2/np.pi)*kMax*A/np.pi*E0*better_result/freq**2

def hilfsfct(freq, E0, w, alpha, kMax, A, phi, k):
    tk          = (k*np.pi - phi)/(2*np.pi*w)
    deltaT      = 1/(2*np.pi*w)*(2*np.pi*w)*kMax/E0*np.exp(tk**2/(2*alpha**2) )
    return (-1)**k*np.exp(-tk**2/(2*alpha**2) )*np.exp(-1j*2*np.pi*freq*tk)*np.sin(freq*2*np.pi*deltaT)

def newHilfsfct(freq, E0, w, alpha, kMax, A, phi, c, k):
    tkOld       = (k*np.pi - phi)/(2*np.pi*w)
    tk          = (-1+np.sqrt(1+4*c*tkOld) )/(2*c)
    deltaT      = kMax/E0*np.exp(tk**2/(2*alpha**2) )
    wk          = w*(1+c*tk)
    return (+1)**k*np.exp(-tk**2/(2*alpha**2)*1 )*np.exp(1j*2*np.pi*freq*tk)*(np.exp(-1j*(tk*wk*2*np.pi+phi) )*np.sin((freq-wk)*2*np.pi*deltaT)/(freq-(1+1e-9j)*wk) + np.exp(1j*(tk*wk*2*np.pi+phi) )*np.sin((freq+wk)*2*np.pi*deltaT)/(freq+(1+1e-9j)*wk))/(2*np.pi)

def cep_plot():
    return 0

if __name__ == "__main__":
    main()
