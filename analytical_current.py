import params
import numpy as np
import matplotlib.pyplot as pl

import data_directory
import os

from matplotlib import ticker, cm


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

    A   *= 1
    kF  = e_fermi/A

    # Time scales T1 = params.T1*fs_conv                            # Occupation damping time
    T1 = params.T1*fs_conv                            # Polarization damping time
    T2 = params.T2*fs_conv                            # Polarization damping time
    gamma1 = 1/T1                                     # Occupation damping parameter
    gamma2 = 1/T2                                     # Polarization damping parameter
    t0 = int(params.t0*fs_conv)                       # Initial time condition
    tf = int(params.tf*fs_conv)                       # Final time
    phi = np.pi*0.5
    
    number  = 1000
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
    c       = 1.0e-9*w
    c       = -0.920*THz_conv

   
    real_fig, (axB,axA,axC,axI) = pl.subplots(4,1,figsize=(10,10))

    kx_array = np.linspace(-np.pi/a, np.pi/a, 1000)
    axB.plot(kx_array, conduction_band(kx_array, ky, A) )

    kx_array = np.linspace(-kF, kF, 1000)
    axB.plot(kx_array, np.ones(kx_array.size)*e_fermi)

    dt = 1/(60*2*w)

    number  = 40
    t_num   = 4096
    k_num   = 800
    
    t       = np.arange(-number*alpha, number*alpha, dt)
    t       = np.linspace(-200*fs_conv, 200*fs_conv, t_num)
    int_current = integrated_current(t, ky, A, kF, E0, w, alpha, phi, c)

    A_t     = A_field(t, E0, w, alpha, phi, c)
    kMax    = 3/2*np.pi/a
    kPoints = np.linspace(-kMax, kMax, k_num)
    
    density = np.zeros((t_num, k_num) )

    for t_k in range(t_num):
        ind_t   = np.where( np.abs(kPoints + A_field(t[t_k], E0, w, alpha, phi, c) ) < 0.991446*kF)
        density[t_k, ind_t ]  = 1
    print(np.sum(density, axis=1) )
    print(max(np.sum(density, axis=1) ) )
    print(min(np.sum(density, axis=1) ) )

    print(current(t, ky, A, kF, E0, w, alpha, phi, c).shape)
    print(density.shape)
    np.save("current_analytical.npy", current(t, ky, A, kF, E0, w, alpha, phi, c) )
    np.save("density_analytical.npy", density )


    
 
    axA.plot(t, A_field(t, E0, w, alpha, phi, c) )
    axC.plot(t/fs_conv, current(t, ky, A, kF, E0, w, alpha, phi, c))
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
    pl.xticks(np.arange(1, 30, 2) )
    pl.grid(True)
    pl.xlim((.1, 25) )
    #pl.ylim((1e-12, 1e-6) )
    pl.yscale("log")
    pl.show()
    pl.clf()
    pl.close()
    return

    angles  = np.linspace(0, np.pi, 10)
    emission_cep_a    = np.array([np.abs(freq*analytical_fourier(freq, E0, w, alpha, ky, kF, A, phi, c) )**2 for phi in angles])
    emission_cep_n    = np.array([np.abs(freq*np.fft.fftshift(np.fft.fft(integrated_current(t, ky, A, kF, E0, w, alpha, phi, c) ) )[len(freq)+1:] )**2 for phi in angles])

    trace_cep(freq, emission_cep_n)

    freq_indices    = np.where(np.abs(freq/w - 30) < 30)[0]
    freq                = freq[freq_indices]
    emission_cep_a      = emission_cep_a[:,freq_indices]
    emission_cep_n      = emission_cep_n[:,freq_indices]
    X, Y = np.meshgrid(freq/w, angles)

    fig, ((axA,axN), (ax2A,ax2N)) = pl.subplots(2,2,figsize=(10,10))
    imA = axA.contourf(X, Y, np.log10(emission_cep_a), levels=50)

    for l in range(1, 0, 1):
        omegas   = (1*l+0) * np.array([1-1*phi*c/(2*np.pi*w) for phi in angles])
        axA.plot(omegas, angles)
        ax2A.plot(omegas, angles)

    imA = axA.contourf(X, Y, np.log10(emission_cep_a), levels=50)
    imN = axN.contourf(X, Y, np.log10(emission_cep_n))
    im2A = ax2A.contourf(X, Y, np.log10(emission_cep_a), levels=50)
    im2N = ax2N.contourf(X, Y, np.log10(emission_cep_n))

    fig.colorbar(imN, ax=axN)
    fig.colorbar(imA, ax=axA)
    fig.colorbar(im2N, ax=ax2N)
    fig.colorbar(im2A, ax=ax2A)

    axA.set_xlabel(r'$\frac{w}{w_0}$')
    axA.set_ylabel(r'$\phi$')
    axN.set_xlabel(r'$\frac{w}{w_0}$')
    axN.set_xlabel(r'$\frac{w}{w_0}$')
    ax2A.set_xlabel(r'$\frac{w}{w_0}$')
    ax2A.set_ylabel(r'$\phi$')
    ax2N.set_ylabel(r'$\phi$')
    ax2N.set_ylabel(r'$\phi$')

    axA.set_xticks(np.arange(1, freq[-1]/w, 1))
    axN.set_xticks(np.arange(1, freq[-1]/w, 1))
    ax2A.set_xticks(np.arange(1, freq[-1]/w, 1))
    ax2N.set_xticks(np.arange(1, freq[-1]/w, 1))

    axA.set_xlim(1, 16)
    axN.set_xlim(1, 16)
    ax2A.set_xlim(16, 28)
    ax2N.set_xlim(16, 28)
    axA.grid(True)
    ax2A.grid(True)

    pl.ylabel(r'$\phi$')
    pl.tight_layout()
    pl.show()
 
def trace_cep(freq, Int):
    i_0         = 10
    w_t         
    print(freq.shape)
    print(Int.shape)
    return

def current(t, ky, A, kF, E0, w, alpha, phi, c):
    return -A/(2*np.pi) * (np.sqrt(ky**2 + (kxMax(ky, kF) - A_field(t, E0, w, alpha, phi, c))**2 ) - np.sqrt(ky**2 + (kxMax(ky, kF) + A_field(t, E0, w, alpha, phi, c))**2 ) )

def integrated_current(t, ky, A, kF, E0, w, alpha, phi, c):
    paths               = np.linspace(-kF, kF, 100)
    dky                 = paths[1] - paths[0]
    integrated_current  = np.array([current(t, ky, A, kF, E0, w, alpha, phi, c) for ky in paths] )
    integrated_current  = np.sum(integrated_current, axis=0)*dky
    return integrated_current

def A_field(t, E0, w, alpha, phi, c):
    return E0/(2*np.pi*w)*np.exp(-(t/(2*alpha) )**2)*np.sin(2*np.pi*(1+c*t)*w*t + phi)

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
