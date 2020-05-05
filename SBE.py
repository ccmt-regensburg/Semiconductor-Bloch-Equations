import numpy as np
from numba import njit
import matplotlib.pyplot as pl
from matplotlib import patches
from scipy.integrate import ode
from scipy.special import erf
from sys import exit

from hfsbe.utility import evaluate_njit_matrix as ev_mat

import params
import systems as sys
from efield import driving_field
'''
TO DO:
UPDATE MATRIX METHOD. NOT COMPATIBLE WITH CODE AS OF NOW. MAGNETIC FIELD.
'''
def main():
    # RETRIEVE PARAMETERS
    ###############################################################################################
    # Unit converstion factors
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
    w     = params.w*params.THz_conv                         # Driving pulse frequency
    chirp = params.chirp*params.THz_conv                     # Pulse chirp frequency
    alpha = params.alpha*params.fs_conv                      # Gaussian pulse width
    phase = params.phase                              # Carrier-envelope phase

    # Dipole scaling to obtain semiclassical motion
    scale_dipole_eq_mot = params.scale_dipole_eq_mot
    scale_dipole_emiss  = params.scale_dipole_emiss

    # Time scales
    T1 = params.T1*fs_conv                            # Occupation damping time
    T2 = params.T2*fs_conv                            # Polarization damping time
    gamma1 = 1/T1                                     # Occupation damping parameter
    gamma2 = 1/T2                                     # Polarization damping parameter
    t0 = int(params.t0*fs_conv)                       # Initial time condition
    tf = int(params.tf*fs_conv)                       # Final time
    dt = params.dt*fs_conv                            # Integration time step
    dt_out = 1/(2*params.dt)                          # Solution output time step

    # Brillouin zone type
    BZ_type = params.BZ_type                          # Type of Brillouin zone to construct

    # Brillouin zone type
    if BZ_type == 'full':
        Nk1   = params.Nk1                                # Number of kpoints in b1 direction
        Nk2   = params.Nk2                                # Number of kpoints in b2 direction
        Nk    = Nk1*Nk2                                   # Total number of kpoints
        align = params.align                              # E-field alignment
    elif BZ_type == 'full_for_velocity':
        Nk1   = params.Nk1_vel                            # Number of kpoints in b1 direction
        Nk2   = params.Nk2_vel                            # Number of kpoints in b2 direction
        Nk    = Nk1*Nk2                                   # Total number of kpoints
        angle_inc_E_field = params.angle_inc_E_field      # Angle of driving electric field
    elif BZ_type == '2line':
        Nk_in_path = params.Nk_in_path                    # Number of kpoints in each of the two paths
        Nk = 2*Nk_in_path                                 # Total number of k points, we have 2 paths
        rel_dist_to_Gamma = params.rel_dist_to_Gamma      # relative distance (in units of 2pi/a) of both paths to Gamma
        length_path_in_BZ = params.length_path_in_BZ      # Length of a single path in the BZ
        angle_inc_E_field = params.angle_inc_E_field      # Angle of driving electric field
        Nk1   = params.Nk_in_path                         # for printing file names, we use Nk1 and ...
        Nk2   = 2                                         # ... and Nk2 = 2

    # Gauge: length versus velocity gauge
    gauge = params.gauge

    b1 = params.b1                                        # Reciprocal lattice vectors
    b2 = params.b2

    user_out            = params.user_out
    print_J_P_I_files   = params.print_J_P_I_files
    energy_plots        = params.energy_plots
    dipole_plots        = params.dipole_plots
    test                = params.test                       # Testing flag for Travis
    do_emission_wavep   = params.emission_wavep
    store_all_timesteps = params.store_all_timesteps
    Bcurv_in_B_dynamics = params.Bcurv_in_B_dynamics

    # USER OUTPUT
    ###############################################################################################
    if user_out:
        print("Solving for...")
        print("Brillouin zone: " + BZ_type)
        print("Number of k-points              = " + str(Nk))
        if BZ_type == 'full':
            print("Driving field alignment         = " + align)
        elif BZ_type == '2line' or BZ_type == 'full_for_velocity':
            print("Driving field direction         = " + str(angle_inc_E_field))
        print("Driving amplitude (MV/cm)[a.u.] = " + "(" + '%.6f'%(E0/E_conv) + ")" + "[" + '%.6f'%(E0) + "]")
        print("Pulse Frequency (THz)[a.u.]     = " + "(" + '%.6f'%(w/THz_conv) + ")" + "[" + '%.6f'%(w) + "]")
        print("Pulse Width (fs)[a.u.]          = " + "(" + '%.6f'%(alpha/fs_conv) + ")" + "[" + '%.6f'%(alpha) + "]")
        print("Chirp rate (THz)[a.u.]          = " + "(" + '%.6f'%(chirp/THz_conv) + ")" + "[" + '%.6f'%(chirp) + "]")
        print("Damping time (fs)[a.u.]         = " + "(" + '%.6f'%(T2/fs_conv) + ")" + "[" + '%.6f'%(T2) + "]")
        print("Total time (fs)[a.u.]           = " + "(" + '%.6f'%((tf-t0)/fs_conv) + ")" + "[" + '%.5i'%(tf-t0) + "]")
        print("Time step (fs)[a.u.]            = " + "(" + '%.6f'%(dt/fs_conv) + ")" + "[" + '%.6f'%(dt) + "]")

    # INITIALIZATIONS
    ###########################################################################
    # Form the E-field direction

    # Form the Brillouin zone in consideration
    if BZ_type == 'full':
        kpnts, paths = hex_mesh(Nk1, Nk2, a, b1, b2, align)
        dk = 1/Nk1
        if align == 'K':
            E_dir = np.array([1, 0])
        elif align == 'M':
            E_dir = np.array([np.cos(np.radians(-30)),
                             np.sin(np.radians(-30))])
    elif BZ_type == 'full_for_velocity':
        E_dir = np.array([np.cos(np.radians(angle_inc_E_field)),
                         np.sin(np.radians(angle_inc_E_field))])
        kpnts, paths = hex_mesh(Nk1, Nk2, a, b1, b2, 'M')
        # dummy
        dk = 1
    elif BZ_type == '2line':
        E_dir = np.array([np.cos(np.radians(angle_inc_E_field)),
                         np.sin(np.radians(angle_inc_E_field))])
        dk, kpnts, paths = mesh(params, E_dir)

    if energy_plots:
        sys.system.evaluate_energy(kpnts[:, 0], kpnts[:, 1])
        sys.system.plot_bands_3d(kpnts[:, 0], kpnts[:, 1])
        sys.system.plot_bands_contour(kpnts[:, 0], kpnts[:, 1])
    if dipole_plots:
        Ax, Ay = sys.dipole.evaluate(kpnts[:, 0], kpnts[:, 1])
        sys.dipole.plot_dipoles(Ax, Ay)

    if store_all_timesteps:
        dt_out = 1

    if B0 > 1e-15:
        do_B_field = True
    else: 
        do_B_field = False

    # here,the time evolution of the density matrix is done
    solution, t, A_field, fermi_function = time_evolution(t0, tf, dt, paths, user_out, E_dir, scale_dipole_eq_mot, e_fermi, temperature, dk, 
                                                          gamma1, gamma2, E0, B0, w, chirp, alpha, phase, do_B_field, gauge, dt_out, BZ_type, Nk1, Nk_in_path, 
                                                          Bcurv_in_B_dynamics, 'density_matrix_dynamics')

    if do_emission_wavep:
       wf_solution, t_wf, A_field_wf, fermi_function = time_evolution(t0, tf, dt, paths, user_out, E_dir, scale_dipole_eq_mot, e_fermi, temperature, dk, 
                                                                      gamma1, gamma2, E0, B0, w, chirp, alpha, phase, do_B_field, gauge, dt_out, BZ_type, Nk1, Nk_in_path, 
                                                                      Bcurv_in_B_dynamics, 'wavefunction_dynamics')
    n_time_steps = np.size(solution[0,0,:,0])

    # COMPUTE OBSERVABLES
    ###########################################################################
    # Calculate parallel and orthogonal components of observables
    # Polarization (interband)
    P_E_dir, P_ortho = polarization(paths, solution[:, :, :, 1], E_dir, scale_dipole_emiss)
    # Current (intraband)
    J_E_dir, J_ortho = current( paths, solution[:, :, :, 0], solution[:, :, :, 3], t, alpha, E_dir)
    # Emission in time
    I_E_dir, I_ortho = diff(t,P_E_dir)*Gaussian_envelope(t,alpha) + J_E_dir*Gaussian_envelope(t,alpha), \
                       diff(t,P_ortho)*Gaussian_envelope(t,alpha) + J_ortho*Gaussian_envelope(t,alpha)
    # emission with exact formula
    if do_B_field:
       I_exact_E_dir, I_exact_ortho = emission_semicl_B_field(paths, solution, E_dir) 
    else:
       I_exact_E_dir, I_exact_ortho = emission_exact(paths, solution, E_dir, A_field, gauge) 
    # emission with exact formula with semiclassical formula
    if do_emission_wavep:
       I_wavep_E_dir, I_wavep_ortho = emission_wavep(paths, solution, wf_solution, E_dir, A_field, fermi_function) 
       I_wavep_check_E_dir, I_wavep_check_ortho = check_emission_wavep(paths, solution, wf_solution, E_dir, A_field, fermi_function) 

    # Polar emission in time
    Ir = []
    angles = np.linspace(0,2.0*np.pi,360)
    for angle in angles:
        Ir.append((I_E_dir*np.cos(angle) + I_ortho*np.sin(-angle)))

    # Fourier transforms
    dt_out   = t[1]-t[0]
    freq     = np.fft.fftshift(np.fft.fftfreq(np.size(t), d=dt_out))
    Iw_E_dir = np.fft.fftshift(np.fft.fft(I_E_dir, norm='ortho'))
    Iw_ortho = np.fft.fftshift(np.fft.fft(I_ortho, norm='ortho'))
    Iw_r     = np.fft.fftshift(np.fft.fft(Ir, norm='ortho'))
    Pw_E_dir = np.fft.fftshift(np.fft.fft(diff(t,P_E_dir), norm='ortho'))
    Pw_ortho = np.fft.fftshift(np.fft.fft(diff(t,P_ortho), norm='ortho'))
    Jw_E_dir = np.fft.fftshift(np.fft.fft(J_E_dir*Gaussian_envelope(t,alpha), norm='ortho'))
    Jw_ortho = np.fft.fftshift(np.fft.fft(J_ortho*Gaussian_envelope(t,alpha), norm='ortho'))
    Iw_exact_E_dir = np.fft.fftshift(np.fft.fft(I_exact_E_dir*Gaussian_envelope(t,alpha), norm='ortho'))
    Iw_exact_ortho = np.fft.fftshift(np.fft.fft(I_exact_ortho*Gaussian_envelope(t,alpha), norm='ortho'))
    if do_emission_wavep:
       Iw_wavep_E_dir = np.fft.fftshift(np.fft.fft(I_wavep_E_dir*Gaussian_envelope(t,alpha), norm='ortho'))
       Iw_wavep_ortho = np.fft.fftshift(np.fft.fft(I_wavep_ortho*Gaussian_envelope(t,alpha), norm='ortho'))
       Iw_wavep_check_E_dir = np.fft.fftshift(np.fft.fft(I_wavep_check_E_dir*Gaussian_envelope(t,alpha), norm='ortho'))
       Iw_wavep_check_ortho = np.fft.fftshift(np.fft.fft(I_wavep_check_ortho*Gaussian_envelope(t,alpha), norm='ortho'))
    fw_0     = np.fft.fftshift(np.fft.fft(solution[:,0,:,0], norm='ortho'),axes=(1,))

    # Emission intensity (approximate formula)
    Int_E_dir = (freq**2)*np.abs(Pw_E_dir + Jw_E_dir)**2.0
    Int_ortho = (freq**2)*np.abs(Pw_ortho + Jw_ortho)**2.0

    # Emission intensity (exact formula)
    Int_exact_E_dir = np.abs((freq**2)*Iw_exact_E_dir**2.0)
    Int_exact_ortho = np.abs((freq**2)*Iw_exact_ortho**2.0)

    # Save observables to file
    if (BZ_type == '2line'):
        Nk1 = Nk_in_path
        Nk2 = 2

    if print_J_P_I_files:  
        J_filename = str('J_Nk1-{}_Nk2-{}_w{:4.2f}_E{:4.2f}_a{:4.2f}_ph{:3.2f}_T2-{:05.2f}').format(Nk1,Nk2,w/THz_conv,E0/E_conv,alpha/fs_conv,phase,T2/fs_conv)
        np.save(J_filename, [t/fs_conv, J_E_dir, J_ortho, freq/w, Jw_E_dir, Jw_ortho])
        P_filename = str('P_Nk1-{}_Nk2-{}_w{:4.2f}_E{:4.2f}_a{:4.2f}_ph{:3.2f}_T2-{:05.2f}').format(Nk1,Nk2,w/THz_conv,E0/E_conv,alpha/fs_conv,phase,T2/fs_conv)
        np.save(P_filename, [t/fs_conv, P_E_dir, P_ortho, freq/w, Pw_E_dir, Pw_ortho])
        I_filename = str('I_Nk1-{}_Nk2-{}_w{:4.2f}_E{:4.2f}_a{:4.2f}_ph{:3.2f}_T2-{:05.2f}').format(Nk1,Nk2,w/THz_conv,E0/E_conv,alpha/fs_conv,phase,T2/fs_conv)
        np.save(I_filename, [t/fs_conv, I_E_dir, I_ortho, freq/w, np.abs(Iw_E_dir), np.abs(Iw_ortho), Int_E_dir, Int_ortho])

    if (not test and user_out):
        real_fig, (axE,axA,axP,axPdot,axJ) = pl.subplots(5,1,figsize=(10,10))
        t_lims = (-10*alpha/fs_conv, 10*alpha/fs_conv)
        freq_lims = (0,25)
        log_limits = (1e-7,1e1)
        axE.set_xlim(t_lims)
        axE.plot(t/fs_conv, driving_field(E0, t)/E_conv)
        axE.set_xlabel(r'$t$ in fs')
        axE.set_ylabel(r'$E$-field in MV/cm')
        axA.set_xlim(t_lims)
        axA.plot(t/fs_conv,A_field/E_conv/fs_conv)
        axA.set_xlabel(r'$t$ in fs')
        axA.set_ylabel(r'$A$-field in MV/cm$\cdot$fs')
        axP.set_xlim(t_lims)
        axP.plot(t/fs_conv,P_E_dir)
        axP.plot(t/fs_conv,P_ortho)
        axP.set_xlabel(r'$t$ in fs')
        axP.set_ylabel(r'$P$ in atomic units $\parallel \mathbf{E}_{in}$ (blue), $\bot \mathbf{E}_{in}$ (orange)')
        axPdot.set_xlim(t_lims)
        axPdot.plot(t/fs_conv,diff(t,P_E_dir))
        axPdot.plot(t/fs_conv,diff(t,P_ortho))
        axPdot.set_xlabel(r'$t$ in fs')
        axPdot.set_ylabel(r'$\dot P$ in atomic units $\parallel \mathbf{E}_{in}$ (blue), $\bot \mathbf{E}_{in}$ (orange)')
        axJ.set_xlim(t_lims)
        axJ.plot(t/fs_conv,J_E_dir)
        axJ.plot(t/fs_conv,J_ortho)
        axJ.set_xlabel(r'$t$ in fs')
        axJ.set_ylabel(r'$J$ in atomic units $\parallel \mathbf{E}_{in}$ (blue), $\bot \mathbf{E}_{in}$ (orange)')

        freq_indices_near_base_freq = np.argwhere(np.logical_and(freq/w > 0.9, freq/w < 1.1))
        freq_index_base_freq = int((freq_indices_near_base_freq[0] + freq_indices_near_base_freq[-1])/2)
        Int_tot_base_freq = Int_exact_E_dir[freq_index_base_freq] + Int_exact_ortho[freq_index_base_freq]

##########################

        if do_B_field:
           label_emission_E_dir = '$I_{\parallel E}(t) = q\sum_{nn\'}\int d\mathbf{k}\;\langle n\overline{\mathbf{k}}_n(t)|\hat{e}_E\cdot \partial h/\partial \mathbf{k}|n\'\overline{\mathbf{k}}_{n\'}(t) \\rangle\\varrho_{nn\'}(\mathbf{k};t)$'
           label_emission_ortho = '$I_{\\bot E}(t) = q\sum_{nn\'}\int d\mathbf{k}\;\langle n\overline{\mathbf{k}}_n(t)|\hat{e}_{\\bot E}\cdot \partial h/\partial \mathbf{k}|n\'\overline{\mathbf{k}}_{n\'}(t) \\rangle\\varrho_{nn\'}(\mathbf{k};t)$'
        else:
           label_emission_E_dir = '$I_{\parallel E}(t) = q\sum_{nn\'}\int d\mathbf{k}\;\langle u_{n\mathbf{k}}|\hat{e}_E\cdot \partial h/\partial \mathbf{k}|_{\mathbf{k}-\mathbf{A}(t)}|u_{n\'\mathbf{k}} \\rangle\\rho_{nn\'}(\mathbf{k},t)$'
           label_emission_ortho = '$I_{\\bot E}(t) = q\sum_{nn\'}\int d\mathbf{k}\;\langle u_{n\mathbf{k}}|\hat{e}_{\\bot E}\cdot \partial h/\partial \mathbf{k}|_{\mathbf{k}-\mathbf{A}(t)}|u_{n\'\mathbf{k}} \\rangle\\rho_{nn\'}(\mathbf{k},t)$'

        if gauge == 'length':
           five_fig, ((ax_I_E_dir,ax_I_ortho,ax_I_total)) = pl.subplots(3,1,figsize=(10,10))
           ax_I_E_dir.grid(True,axis='x')
           ax_I_E_dir.set_xlim(freq_lims)
           ax_I_E_dir.set_ylim(log_limits)
           ax_I_E_dir.semilogy(freq/w,Int_exact_E_dir / Int_tot_base_freq, label=label_emission_E_dir)
           if not do_B_field:
              ax_I_E_dir.semilogy(freq/w, Int_E_dir / Int_tot_base_freq, 
                 label='$I_{\mathrm{i+i} \parallel E}(t) = I_{\mathrm{intra} \parallel E}(t) + I_{\mathrm{inter} \parallel E}(t)$')
              ax_I_E_dir.semilogy(freq/w,np.abs(freq**2*Jw_E_dir**2) / Int_tot_base_freq,  linestyle='dashed',
                 label='$I_{\mathrm{intra} \parallel E}(t) = q\sum_{n}\int d\mathbf{k}\; \hat{e}_E\cdot\partial \\epsilon_n/\partial\mathbf{k}\;\\rho_{nn(\mathbf{k},t)}$')
              ax_I_E_dir.semilogy(freq/w,np.abs(freq**2*Pw_E_dir**2) / Int_tot_base_freq, linestyle='dashed', 
                 label='$I_{\mathrm{inter} \parallel E}(t) = \sum_{n\\neq n\'}\int d\mathbf{k}\;\hat{e}_E\cdot \mathbf{d}_{nn\'}(\mathbf{k})\dot\\rho_{n\'n(\mathbf{k},t)}$')
           ax_I_E_dir.set_xlabel(r'Frequency $\omega/\omega_0$')
           ax_I_E_dir.set_ylabel(r'Emission $I_{\parallel E}(\omega)$ in E-field direction')
           ax_I_E_dir.legend(loc='upper right')
           ax_I_ortho.grid(True,axis='x')
           ax_I_ortho.set_xlim(freq_lims)
           ax_I_ortho.set_ylim(log_limits)
           ax_I_ortho.semilogy(freq/w,Int_exact_ortho / Int_tot_base_freq, label=label_emission_ortho)
           if not do_B_field:
              ax_I_ortho.semilogy(freq/w,Int_ortho / Int_tot_base_freq, 
                 label='$I_{\mathrm{i+i} \\bot E}(t) = I_{\mathrm{intra} \\bot E}(t) + I_{\mathrm{inter} \\bot E}(t)$')
              ax_I_ortho.semilogy(freq/w,np.abs(freq**2*Jw_ortho**2) / Int_tot_base_freq,  linestyle='dashed',
                 label='$I_{\mathrm{intra} \\bot E}(t) = q\sum_{n}\int d\mathbf{k}\; \hat{e}_{\\bot E}\cdot\partial \\epsilon_n/\partial\mathbf{k}\;\\rho_{nn(\mathbf{k},t)}$')
              ax_I_ortho.semilogy(freq/w,np.abs(freq**2*Pw_ortho**2) / Int_tot_base_freq, linestyle='dashed',
                 label='$I_{\mathrm{inter} \\bot E}(t) = \sum_{n\\neq n\'}\int d\mathbf{k}\;\hat{e}_{\\bot E}\cdot \mathbf{d}_{nn\'}(\mathbf{k})\dot\\rho_{n\'n(\mathbf{k},t)}$')
           ax_I_ortho.set_xlabel(r'Frequency $\omega/\omega_0$')
           ax_I_ortho.set_ylabel(r'Emission $I_{\bot E}(\omega)$ $\bot$ to E-field direction')
           ax_I_ortho.legend(loc='upper right')
           ax_I_total.grid(True,axis='x')
           ax_I_total.set_xlim(freq_lims)
           ax_I_total.set_ylim(log_limits)
           ax_I_total.semilogy(freq/w,(Int_exact_E_dir + Int_exact_ortho) / Int_tot_base_freq, 
              label='$I(\omega) = I_{\parallel E}(\omega) + I_{\\bot E}(\omega)$')
           if not do_B_field:
              ax_I_total.semilogy(freq/w,(Int_E_dir+Int_ortho) / Int_tot_base_freq, 
                 label='$I_{\mathrm{i+i}}(t) = I_{\mathrm{i+i} \parallel E}(t) + I_{\mathrm{i+i} \\bot E}(t)$')
           ax_I_total.set_xlabel(r'Frequency $\omega/\omega_0$')
           ax_I_total.set_ylabel(r'Total emission $I(\omega)$')
           ax_I_total.legend(loc='upper right')

        B_fig_all_in_one, ((B_1)) = pl.subplots(1,1,figsize=(10,4))
        B_1.semilogy(freq/w,Int_exact_E_dir / Int_tot_base_freq, label=label_emission_E_dir)
        B_1.semilogy(freq/w,Int_exact_ortho / Int_tot_base_freq, label=label_emission_ortho)
        B_1.semilogy(freq/w,(Int_exact_E_dir + Int_exact_ortho) / Int_tot_base_freq, 
            label='$I(\omega) = I_{\parallel E}(\omega) + I_{\\bot E}(\omega)$')
        B_1.set_xlabel(r'Frequency $\omega/\omega_0$')
        B_1.set_ylabel(r'Relative emission intensity $I(\omega)$')
        B_1.legend(loc='upper right')
        B_1.grid(True,axis='x')
        B_1.set_xlim(freq_lims)
        B_1.set_ylim(log_limits)

        if do_emission_wavep:

           six_fig, ((sc_I_E_dir,sc_I_ortho,sc_I_total)) = pl.subplots(3,1,figsize=(10,10))
           sc_I_E_dir.grid(True,axis='x')
           sc_I_E_dir.set_xlim(freq_lims)
           sc_I_E_dir.set_ylim(log_limits)
           sc_I_E_dir.semilogy(freq/w,np.abs(freq**2*Iw_exact_E_dir**2) / Int_tot_base_freq, 
            label='$I_{\parallel E}^\mathrm{full}(t) = q\sum_{nn\'}\int d\mathbf{k}\;\langle u_{n\mathbf{k}}|\hat{e}_E\cdot \partial h/\partial \mathbf{k}|_{\mathbf{k}-\mathbf{A}(t)}|u_{n\'\mathbf{k}} \\rangle\\rho_{nn\'}(\mathbf{k},t)$')
           sc_I_E_dir.semilogy(freq/w, np.abs(freq**2*Iw_wavep_check_E_dir**2) / Int_tot_base_freq, linestyle='dotted', 
             label='$I_{\parallel E}^\mathrm{wavep}(t) = q\sum_{nn\'}\int d\mathbf{k}\;\langle u_{n\mathbf{k}}|\hat{e}_E\cdot \partial h/\partial \mathbf{k}|_{\mathbf{k}-\mathbf{A}(t)}|u_{n\'\mathbf{k}} \\rangle\\tilde{\\rho}_{nn\'}(\mathbf{k},t)$ with $\\tilde{\\rho}_{nn\'}(\mathbf{k}(t),t)$ from wf.~dyn.')
           sc_I_E_dir.semilogy(freq/w, np.abs(freq**2*Iw_wavep_E_dir**2) / Int_tot_base_freq, linestyle='dotted',
              label='$I_{\parallel E}^\mathrm{wavep check}(t) = q\sum_{nn\'}\int d\mathbf{k}\;\langle n\mathbf{k}(t),t|\hat{e}_E\cdot \partial h/\partial \mathbf{k}|_{\mathbf{k}-\mathbf{A}(t)}|n\mathbf{k}(t),t \\rangle f_{n}(\mathbf{k}(t)) $')
           sc_I_E_dir.set_xlabel(r'Frequency $\omega/\omega_0$')
           sc_I_E_dir.set_ylabel(r'Emission $I_{\parallel E}(\omega)$ in E-field direction')
           sc_I_E_dir.legend(loc='lower right')
   
           sc_I_ortho.grid(True,axis='x')
           sc_I_ortho.set_xlim(freq_lims)
           sc_I_ortho.set_ylim(log_limits)
           sc_I_ortho.semilogy(freq/w,np.abs(freq**2*Iw_exact_ortho**2) / Int_tot_base_freq, 
            label='$I_{\\bot E}^\mathrm{full}(t) = q\sum_{nn\'}\int d\mathbf{k}\;\langle u_{n\mathbf{k}}|\hat{e}_{\\bot E}\cdot \partial h/\partial \mathbf{k}|_{\mathbf{k}-\mathbf{A}(t)}|u_{n\'\mathbf{k}} \\rangle\\rho_{nn\'(\mathbf{k},t)}$')
           sc_I_ortho.semilogy(freq/w, np.abs(freq**2*Iw_wavep_check_ortho**2) / Int_tot_base_freq, linestyle='dotted',
              label='$I_{\\bot E}^\mathrm{wavep}(t)$')
           sc_I_ortho.semilogy(freq/w, np.abs(freq**2*Iw_wavep_ortho**2) / Int_tot_base_freq, linestyle='dotted',
              label='$I_{\\bot E}^\mathrm{wavep check}(t)$')
           sc_I_ortho.set_xlabel(r'Frequency $\omega/\omega_0$')
           sc_I_ortho.set_ylabel(r'Emission $I_{\parallel E}(\omega)$ in E-field direction')
           sc_I_ortho.legend(loc='lower right')
   
           sc_I_total.grid(True,axis='x')
           sc_I_total.set_xlim(freq_lims)
           sc_I_total.set_ylim(log_limits)
           sc_I_total.semilogy(freq/w,np.abs(freq**2*(Iw_exact_E_dir**2 + Iw_exact_ortho**2)) / Int_tot_base_freq, 
            label='$I^\mathrm{full}(\omega) = I_{\parallel E}^\mathrm{full}(\omega) + I_{\\bot E}^\mathrm{full}(\omega)$')
           sc_I_total.semilogy(freq/w,np.abs(freq**2*(Iw_wavep_check_E_dir**2 + Iw_wavep_check_ortho**2)) / Int_tot_base_freq, linestyle='dotted',
            label='$I^\mathrm{wavep}(\omega) = I^\mathrm{wavep}_{\parallel E}(\omega) + I^\mathrm{wavep}_{\\bot E}(\omega)$')
           sc_I_total.semilogy(freq/w,np.abs(freq**2*(Iw_wavep_E_dir**2 + Iw_wavep_ortho**2)) / Int_tot_base_freq, linestyle='dotted',
            label='$I^\mathrm{wavep check}(\omega) = I^\mathrm{wavep check}_{\parallel E}(\omega) + I^\mathrm{wavep}_{\\bot E}(\omega)$')
           sc_I_total.set_xlabel(r'Frequency $\omega/\omega_0$')
           sc_I_total.set_ylabel(r'Total emission $I(\omega)$')
           sc_I_total.legend(loc='lower right')

######################äää

        kp_array = length_path_in_BZ*np.linspace(-0.5 + (1/(2*Nk_in_path)), 0.5 - (1/(2*Nk_in_path)), num = Nk_in_path)
        # Countour plots of occupations and gradients of occupations
        fig5 = pl.figure()
        X, Y = np.meshgrid(t/fs_conv,kp_array)
        pl.contourf(X, Y, np.real(solution[:,0,:,3]), 100)
        pl.colorbar().set_label(r'$f_e(k)$ in path 0')
        pl.xlim([-5*alpha/fs_conv,10*alpha/fs_conv])
        pl.xlabel(r'$t\;(fs)$')
        pl.ylabel(r'$k$')
        pl.tight_layout()

        # High-harmonic emission polar plots
        polar_fig = pl.figure(figsize=(10, 10))
        i_loop = 1
        i_max = 20
        while i_loop <= i_max:
            freq_indices = np.argwhere(np.logical_and(freq/w > float(i_loop)-0.1, freq/w < float(i_loop)+0.1))
            freq_index   = freq_indices[int(np.size(freq_indices)/2)]
            pax          = polar_fig.add_subplot(1,i_max,i_loop,projection='polar')
            pax.plot(angles,np.abs(Iw_r[:,freq_index]))
            rmax = pax.get_rmax()
            pax.set_rmax(1.1*rmax)
            pax.set_yticklabels([""])
            if i_loop == 1:
                pax.set_rgrids([0.25*rmax,0.5*rmax,0.75*rmax,1.0*rmax],labels=None, angle=None, fmt=None)
                pax.set_title('HH'+str(i_loop), va='top', pad=30)
                pax.set_xticks(np.arange(0,2.0*np.pi,np.pi/6.0))
            else:
                pax.set_rgrids([0.0],labels=None, angle=None, fmt=None)
                pax.set_xticks(np.arange(0,2.0*np.pi,np.pi/2.0))
                pax.set_xticklabels([""])
                pax.set_title('HH'+str(i_loop), va='top', pad=15)
            i_loop += 1

        # Plot Brilluoin zone with paths
        BZ_plot(kpnts,a,b1,b2,E_dir,paths)

        pl.show()

    # OUTPUT STANDARD TEST VALUES
    ##############################################################################################
    if test:
        t_zero = np.argwhere(t == 0)
        f5 = np.argwhere(np.logical_and(freq/w > 4.9, freq/w < 5.1))
        f125 = np.argwhere(np.logical_and(freq/w > 12.4, freq/w < 12.6))
        f15= np.argwhere(np.logical_and(freq/w > 14.9, freq/w < 15.1))
        f_5 = f5[int(np.size(f5)/2)]
        f_125 = f125[int(np.size(f125)/2)]
        f_15 = f15[int(np.size(f15)/2)]
        test_out = np.zeros(6, dtype=[('names','U16'),('values',float)])
        test_out['names'] = np.array(['P(t=0)','J(t=0)','N_gamma(t=tf)','Emis(w/w0=5)','Emis(w/w0=12.5)','Emis(w/w0=15)'])
        test_out['values'] = np.array([pol[t_zero],curr[t_zero],N_gamma[Nt-1],emis[f_5],emis[f_125],emis[f_15]])
        np.savetxt('test.dat',test_out, fmt='%16s %.16e')


def time_evolution(t0, tf, dt, paths, user_out, E_dir, scale_dipole_eq_mot, e_fermi, temperature, dk, gamma1, gamma2, 
                   E0, B0, w, chirp, alpha, phase, do_B_field, gauge, dt_out, BZ_type, Nk1, Nk_in_path, Bcurv_in_B_dynamics, 
                   dynamics_type):

    if dynamics_type == 'density_matrix_dynamics' and user_out:
       print("Enter density matrix dynamics.")
    elif dynamics_type == 'wavefunction_dynamics' and user_out:
       print("Enter wavefunction dynamics.")
       if gauge == 'length': 
           print("Wavefunction dynamics only implemented for velocity gauge. Script abords.")
           exit("")

    # Solution containers
    t = []
    solution = []
    fermi_function = []

    # Number of integration steps, time array construction flag
    Nt = int((tf-t0)/dt)
    t_constructed = False

    # Initialize the ode solver
    solver = ode(f, jac=None).set_integrator('zvode', method='bdf', max_step=dt)

    # SOLVING
    ###########################################################################
    # Iterate through each path in the Brillouin zone
    path_num = 1
    for path in paths:
        if user_out:
            print('path: ' + str(path_num))

        # Solution container for the current path
        path_solution = []
        path_fermi_function = []

        # Retrieve the set of k-points for the current path
        kx_in_path = path[:, 0]
        ky_in_path = path[:, 1]

        # Calculate the dipole components along the path
        di_x, di_y = sys.dipole.evaluate(kx_in_path, ky_in_path)

        # Calculate the dot products E_dir.d_nm(k).
        # To be multiplied by E-field magnitude later.
        # A[0,1,:] means 0-1 offdiagonal element
        dipole_in_path = scale_dipole_eq_mot*(E_dir[0]*di_x[0, 1, :] + E_dir[1]*di_y[0, 1, :])
        A_in_path = E_dir[0]*di_x[0, 0, :] + E_dir[1]*di_y[0, 0, :] \
            - (E_dir[0]*di_x[1, 1, :] + E_dir[1]*di_y[1, 1, :])
        Avv_in_path = E_dir[0]*di_x[0, 0, :] + E_dir[1]*di_y[0, 0, :]
        Acc_in_path = E_dir[0]*di_x[1, 1, :] + E_dir[1]*di_y[1, 1, :]

        # in bite.evaluate, there is also an interpolation done if b1, b2
        # are provided and a cutoff radius
        bandstruct = sys.system.evaluate_energy(kx_in_path, ky_in_path)
        ecv_in_path = bandstruct[1] - bandstruct[0]
        ev_in_path = -ecv_in_path/2
        ec_in_path = ecv_in_path/2

        ec = bandstruct[1]

        # Initialize the values of of each k point vector
        # (rho_nn(k), rho_nm(k), rho_mn(k), rho_mm(k))
        y0 = []
        for i_k, k in enumerate(path):
            initial_condition(y0,e_fermi,temperature,bandstruct[1],i_k, dynamics_type)

        # append the A-field
        y0.append(0.0)

        y0_np = np.array(y0)

        # Set the initual values and function parameters for the current kpath
        solver.set_initial_value(y0, t0).set_f_params(path, dk, gamma1, gamma2, E0, B0, w, chirp, alpha, phase, do_B_field, 
                                                      ecv_in_path, ev_in_path, ec_in_path, 
                                                      dipole_in_path, A_in_path, Avv_in_path, Acc_in_path, 
                                                      gauge, kx_in_path, ky_in_path, E_dir, y0_np, Bcurv_in_B_dynamics, 
                                                      dynamics_type)

        # Propagate through time
        ti = 0
        while solver.successful() and ti < Nt:

            # User output of integration progress
            if (ti % 1000 == 0 and user_out):
                print('{:5.2f}%'.format(ti/Nt*100))

            # Integrate one integration time step
            solver.integrate(solver.t + dt)

            # Save solution each output step
            if ti % dt_out == 0:
                path_solution.append(solver.y)
                if dynamics_type == 'wavefunction_dynamics':
                    path_fermi_function.append( 1/(np.exp((ec[:]-e_fermi)/temperature)+1) )

                # Construct time array only once
                if not t_constructed:
                    t.append(solver.t)

            # Increment time counter
            ti += 1

        # Flag that time array has been built up
        t_constructed = True
        path_num += 1

        # Append path solutions to the total solution arrays
        solution.append(np.array(path_solution)[:, 0:-1])
        if dynamics_type == 'wavefunction_dynamics':
           fermi_function.append(np.array(path_fermi_function)[:, :])
   
    # Convert solution and time array to numpy arrays
    t = np.array(t)
    solution       = np.array(solution)
    fermi_function = np.array(fermi_function)
    A_field        = np.array(path_solution)[:, -1]

    # Slice solution along each path for easier observable calculation
    if BZ_type == 'full' or BZ_type == 'full_for_velocity':
        solution = np.array_split(solution, Nk1, axis=2)
        if dynamics_type == 'wavefunction_dynamics':
           fermi_function = np.array_split(fermi_function, Nk1, axis=2)
    elif BZ_type == '2line':
        solution = np.array_split(solution, Nk_in_path, axis=2)
        if dynamics_type == 'wavefunction_dynamics':
           fermi_function = np.array_split(fermi_function, Nk_in_path, axis=2)

    # Convert lists into numpy arrays
    solution = np.array(solution)
    # Now the solution array is structred as:
    # first index is kx-index, second is ky-index,
    # third is timestep, fourth is f_h, p_he, p_eh, f_e

    if dynamics_type == 'wavefunction_dynamics':
        fermi_function = np.array(fermi_function)

    # In case of the velocity gauge, we need to shift the time-dependent
    # k(t)=k_0+e/hbar A(t) to k_0 = k(t) - e/hbar A(t)
    if gauge == 'velocity' and do_B_field == False and dynamics_type == 'wavefunction_dynamics':
        solution = shift_solution(solution, A_field, dk, dynamics_type)
        fermi_function = shift_solution(fermi_function, A_field, dk, dynamics_type)

    return solution, t, A_field, fermi_function

#################################################################################################
# FUNCTIONS
################################################################################################
def mesh(params, E_dir):
    Nk_in_path = params.Nk_in_path                    # Number of kpoints in each of the two paths
    rel_dist_to_Gamma = params.rel_dist_to_Gamma      # relative distance (in units of 2pi/a) of both paths to Gamma
    a = params.a                                      # Lattice spacing
    length_path_in_BZ = params.length_path_in_BZ      #

    alpha_array = np.linspace(-0.5 + (1/(2*Nk_in_path)), 0.5 - (1/(2*Nk_in_path)), num = Nk_in_path)
    vec_k_path = E_dir*length_path_in_BZ

    vec_k_ortho = 2.0*np.pi/a*rel_dist_to_Gamma*np.array([E_dir[1], -E_dir[0]])

    # Containers for the mesh, and BZ directional paths
    mesh = []
    paths = []

    # Create the kpoint mesh and the paths
    for path_index in [-1, 1]:
        # Container for a single path
        path = []
        for alpha in alpha_array:
            # Create a k-point
            kpoint = path_index*vec_k_ortho + alpha*vec_k_path

            mesh.append(kpoint)
            path.append(kpoint)

        # Append the a1'th path to the paths array
        paths.append(path)

    dk = 1.0/Nk_in_path*length_path_in_BZ

    return dk, np.array(mesh), np.array(paths)


def hex_mesh(Nk1, Nk2, a, b1, b2, align):
    alpha1 = np.linspace(-0.5 + (1/(2*Nk1)), 0.5 - (1/(2*Nk1)), num=Nk1)
    alpha2 = np.linspace(-0.5 + (1/(2*Nk2)), 0.5 - (1/(2*Nk2)), num=Nk2)

    def is_in_hex(p, a):
        # Returns true if the point is in the hexagonal BZ.
        # Checks if the absolute values of x and y components of p are within the first quadrant of the hexagon.
        x = np.abs(p[0])
        y = np.abs(p[1])
        return ((y <= 2.0*np.pi/(np.sqrt(3)*a)) and (np.sqrt(3.0)*x + y <= 4*np.pi/(np.sqrt(3)*a)))

    def reflect_point(p, a, b1, b2):
        x = p[0]
        y = p[1]
        if (y > 2*np.pi/(np.sqrt(3)*a)):                     # Crosses top
            p -= b2
        elif (y < -2*np.pi/(np.sqrt(3)*a)):                  # Crosses bottom
            p += b2
        elif (np.sqrt(3)*x + y > 4*np.pi/(np.sqrt(3)*a)):    # Crosses top-right
            p -= b1 + b2
        elif (-np.sqrt(3)*x + y < -4*np.pi/(np.sqrt(3)*a)):  # Crosses bot-right
            p -= b1
        elif (np.sqrt(3)*x + y < -4*np.pi/(np.sqrt(3)*a)):   # Crosses bot-left
            p += b1 + b2
        elif (-np.sqrt(3)*x + y > 4*np.pi/(np.sqrt(3)*a)):   # Crosses top-left
            p += b1
        return p

    # Containers for the mesh, and BZ directional paths
    mesh = []
    paths = []

    # Create the Monkhorst-Pack mesh
    if align == 'M':
        for a2 in alpha2:
            # Container for a single gamma-M path
            path_M = []
            for a1 in alpha1:
                # Create a k-point
                kpoint = a1*b1 + a2*b2
                # If the current point is in the BZ, append it to the mesh and path_M
                if (is_in_hex(kpoint,a)):
                    mesh.append(kpoint)
                    path_M.append(kpoint)
                # If the current point is NOT in the BZ, reflect is along the appropriate axis to get it in the BZ, then append.
                else:
                    while (is_in_hex(kpoint,a) != True):
                        kpoint = reflect_point(kpoint,a,b1,b2)
                    mesh.append(kpoint)
                    path_M.append(kpoint)
            # Append the a1'th path to the paths array
            paths.append(path_M)

    elif align == 'K':
        b_a1 = 8*np.pi/(a*3)*np.array([1,0])
        b_a2 = 4*np.pi/(a*3)*np.array([1,np.sqrt(3)])
        # Extend over half of the b2 direction and 1.5x the b1 direction (extending into the 2nd BZ to get correct boundary conditions)
        alpha1 = np.linspace(-0.5 + (1/(2*Nk1)), 1.0 - (1/(2*Nk1)), num = Nk1)
        alpha2 = np.linspace(0, 0.5 - (1/(2*Nk2)), num = Nk2)
        for a2 in alpha2:
            path_K = []
            for a1 in alpha1:
                kpoint = a1*b_a1 + a2*b_a2
                if is_in_hex(kpoint,a):
                    mesh.append(kpoint)
                    path_K.append(kpoint)
                else:
                    kpoint -= 2*np.pi/(a)*np.array([1,1/np.sqrt(3)])
                    mesh.append(kpoint)
                    path_K.append(kpoint)
            paths.append(path_K)

    return np.array(mesh), np.array(paths)

# @njit
# def driving_field(E0, w, t, chirp, alpha, phase):
#     '''
#     Returns the instantaneous driving pulse field
#     '''
#     # Non-pulse
#     # return E0*np.sin(2.0*np.pi*w*t)
#     # Chirped Gaussian pulse
#     return E0*np.exp(-t**2.0/(2.0*alpha)**2)*np.sin(2.0*np.pi*w*t*(1 + chirp*t) + phase)

@njit
def rabi(E0, t, dipole):
    '''
    Rabi frequency of the transition.
    Calculated from dipole element and driving field
    '''
    return dipole*driving_field(E0, t)


def diff(x, y):
    '''
    Takes the derivative of y w.r.t. x
    '''
    if (len(x) != len(y)):
        raise ValueError('Vectors have different lengths')
    elif len(y) == 1:
        return 0
    else:
        dx = np.gradient(x)
        dy = np.gradient(y)
        return dy/dx


def Gaussian_envelope(t, alpha):
    '''
    Function to multiply a Function f(t) before Fourier transform
    to ensure no step in time between t_final and t_final + delta
    '''
    return np.exp(-t**2.0/(2.0*1.0*alpha)**2)


def polarization(paths, pcv, E_dir, scale_dipole_emiss):
    '''
    Calculates the polarization as: P(t) = sum_n sum_m sum_k [d_nm(k)p_nm(k)]
    Dipole term currently a crude model to get a vector polarization
    '''
    E_ort = np.array([E_dir[1], -E_dir[0]])

    d_E_dir, d_ortho = [],[]
    for path in paths:

        kx_in_path = path[:, 0]
        ky_in_path = path[:, 1]

        # Evaluate the dipole moments in path
        di_x, di_y = sys.dipole.evaluate(kx_in_path, ky_in_path)

        # Append the dot product d.E
        d_E_dir.append(di_x[0, 1, :]*E_dir[0] + di_y[0, 1, :]*E_dir[1])
        d_ortho.append(di_x[0, 1, :]*E_ort[0] + di_y[0, 1, :]*E_ort[1])

    d_E_dir_swapped = np.swapaxes(d_E_dir, 0, 1)
    d_ortho_swapped = np.swapaxes(d_ortho, 0, 1)
    # d_E_dir = d_E_dir.T
    # d_ortho = d_ortho.T

    P_E_dir = 2*np.real(np.tensordot(d_E_dir_swapped, pcv, 2))*scale_dipole_emiss
    P_ortho = 2*np.real(np.tensordot(d_ortho_swapped, pcv, 2))*scale_dipole_emiss
    # P_E_dir = 2*np.real(np.tensordot(d_E_dir,pcv,2))
    # P_ortho = 2*np.real(np.tensordot(d_ortho,pcv,2))

    return P_E_dir, P_ortho


def current(paths, fv, fc, t, alpha, E_dir):
    '''
    Calculates the current as: J(t) = sum_k sum_n [j_n(k)f_n(k,t)]
    where j_n(k) != (d/dk) E_n(k)
    '''
    E_ort = np.array([E_dir[1], -E_dir[0]])
    
    # Calculate the gradient analytically at each k-point
    J_E_dir, J_ortho = [], []
    jc_E_dir, jc_ortho, jv_E_dir, jv_ortho = [], [], [], []
    for path in paths:
        path = np.array(path)
        kx_in_path = path[:, 0]
        ky_in_path = path[:, 1]
        
        evdx = sys.system.ederivfjit[0](kx=kx_in_path, ky=ky_in_path)
        evdy = sys.system.ederivfjit[1](kx=kx_in_path, ky=ky_in_path)
        ecdx = sys.system.ederivfjit[2](kx=kx_in_path, ky=ky_in_path)
        ecdy = sys.system.ederivfjit[3](kx=kx_in_path, ky=ky_in_path)
    
        # 0: v, x 1: v,y 2: c, x 3: c, y
        jc_E_dir.append(ecdx*E_dir[0] + ecdy*E_dir[1])
        jc_ortho.append(ecdx*E_ort[0] + ecdy*E_ort[1])
        jv_E_dir.append(evdx*E_dir[0] + evdy*E_dir[1])
        jv_ortho.append(evdx*E_ort[0] + evdy*E_ort[1])
    
    jc_E_dir = np.swapaxes(jc_E_dir, 0, 1)
    jc_ortho = np.swapaxes(jc_ortho, 0, 1)
    jv_E_dir = np.swapaxes(jv_E_dir, 0, 1)
    jv_ortho = np.swapaxes(jv_ortho, 0, 1)
    
    # tensordot for contracting the first two indices (2 kpoint directions)
    J_E_dir = np.tensordot(jc_E_dir, fc, 2) + np.tensordot(jv_E_dir, fv, 2)
    J_ortho = np.tensordot(jc_ortho, fc, 2) + np.tensordot(jv_ortho, fv, 2)
    
    # Return the real part of each component
    return np.real(J_E_dir), np.real(J_ortho)

def emission_exact(paths, solution, E_dir, A_field, gauge):

    E_ort = np.array([E_dir[1], -E_dir[0]])

    n_time_steps = np.size(solution[0,0,:,0])

    # I_E_dir is of size (number of time steps)
    I_E_dir = np.zeros(n_time_steps)
    I_ortho = np.zeros(n_time_steps)

    for i_time in range(n_time_steps):

        for i_path, path in enumerate(paths):
            path = np.array(path)
            kx_in_path = path[:, 0]
            ky_in_path = path[:, 1]

            if gauge == 'length':

               kx_in_path_for_h_deriv = kx_in_path - A_field[i_time]*E_dir[0]
               ky_in_path_for_h_deriv = ky_in_path - A_field[i_time]*E_dir[1]
   
               kx_in_path_for_U       = kx_in_path 
               ky_in_path_for_U       = ky_in_path 

            elif gauge == 'velocity':
       
               kx_in_path_for_h_deriv = kx_in_path - 2*A_field[i_time]*E_dir[0]
               ky_in_path_for_h_deriv = ky_in_path - 2*A_field[i_time]*E_dir[1]
   
               kx_in_path_for_U       = kx_in_path - A_field[i_time]*E_dir[0]
               ky_in_path_for_U       = ky_in_path - A_field[i_time]*E_dir[1]

            h_deriv_x = ev_mat(sys.h_deriv[0], kx=kx_in_path_for_h_deriv, ky=ky_in_path_for_h_deriv)
            h_deriv_y = ev_mat(sys.h_deriv[1], kx=kx_in_path_for_h_deriv, ky=ky_in_path_for_h_deriv)
 
            h_deriv_E_dir = h_deriv_x*E_dir[0] + h_deriv_y*E_dir[1]
            h_deriv_ortho = h_deriv_x*E_ort[0] + h_deriv_y*E_ort[1]

            U   = sys.wf  (kx=kx_in_path_for_U, ky=ky_in_path_for_U)
            U_h = sys.wf_h(kx=kx_in_path_for_U, ky=ky_in_path_for_U)
    
            for i_k in range(np.size(kx_in_path)):

                U_h_H_U_E_dir = np.matmul(U_h[:,:,i_k], np.matmul(h_deriv_E_dir[:,:,i_k], U[:,:,i_k]))
                U_h_H_U_ortho = np.matmul(U_h[:,:,i_k], np.matmul(h_deriv_ortho[:,:,i_k], U[:,:,i_k]))

                I_E_dir[i_time] += np.real(U_h_H_U_E_dir[0,0])*np.real(solution[i_k, i_path, i_time, 0])
                I_E_dir[i_time] += np.real(U_h_H_U_E_dir[1,1])*np.real(solution[i_k, i_path, i_time, 3])
                I_E_dir[i_time] += 2*np.real(U_h_H_U_E_dir[0,1]*solution[i_k, i_path, i_time, 2])

                I_ortho[i_time] += np.real(U_h_H_U_ortho[0,0])*np.real(solution[i_k, i_path, i_time, 0])
                I_ortho[i_time] += np.real(U_h_H_U_ortho[1,1])*np.real(solution[i_k, i_path, i_time, 3])
                I_ortho[i_time] += 2*np.real(U_h_H_U_ortho[0,1]*solution[i_k, i_path, i_time, 2])

    return I_E_dir, I_ortho


def emission_semicl_B_field(paths, solution, E_dir):

    E_ort = np.array([E_dir[1], -E_dir[0]])

    n_time_steps = np.size(solution[0,0,:,0])

    # I_E_dir is of size (number of time steps)
    I_E_dir = np.zeros(n_time_steps)
    I_ortho = np.zeros(n_time_steps)

    for i_time in range(n_time_steps):

        for i_path, path in enumerate(paths):
            path = np.array(path)
            kx_in_path = path[:, 0]
            ky_in_path = path[:, 1]

            h_deriv_x = ev_mat(sys.h_deriv[0], kx=kx_in_path, ky=ky_in_path)
            h_deriv_y = ev_mat(sys.h_deriv[1], kx=kx_in_path, ky=ky_in_path)
 
            h_deriv_E_dir = h_deriv_x*E_dir[0] + h_deriv_y*E_dir[1]
            h_deriv_ortho = h_deriv_x*E_ort[0] + h_deriv_y*E_ort[1]
    
            for i_k in range(np.size(kx_in_path)):

                kx_in_path_shifted_v = kx_in_path[i_k] + solution[i_k, i_path, i_time, 4]
                ky_in_path_shifted_v = ky_in_path[i_k] + solution[i_k, i_path, i_time, 5]
                kx_in_path_shifted_c = kx_in_path[i_k] + solution[i_k, i_path, i_time, 6]
                ky_in_path_shifted_c = ky_in_path[i_k] + solution[i_k, i_path, i_time, 7]

                U_shift_v   = sys.wf  (kx=kx_in_path_shifted_v, ky=ky_in_path_shifted_v)
                U_shift_v_h = sys.wf_h(kx=kx_in_path_shifted_v, ky=ky_in_path_shifted_v)
                U_shift_c   = sys.wf  (kx=kx_in_path_shifted_c, ky=ky_in_path_shifted_c)
                U_shift_c_h = sys.wf_h(kx=kx_in_path_shifted_c, ky=ky_in_path_shifted_c)

                U_shift = np.zeros((2,2),dtype='complex')
                U_shift[0,0] = U_shift_v[0,0]
                U_shift[1,0] = U_shift_v[1,0]
                U_shift[0,1] = U_shift_c[0,1]
                U_shift[1,1] = U_shift_c[1,1]

                U_shift_h = np.zeros((2,2),dtype='complex')
                U_shift_h[0,0] = U_shift_v_h[0,0]
                U_shift_h[0,1] = U_shift_v_h[0,1]
                U_shift_h[1,0] = U_shift_c_h[1,0]
                U_shift_h[1,1] = U_shift_c_h[1,1]

                U_h_H_U_E_dir = np.matmul(U_shift_h[:,:], np.matmul(h_deriv_E_dir[:,:,i_k], U_shift[:,:]))
                U_h_H_U_ortho = np.matmul(U_shift_h[:,:], np.matmul(h_deriv_ortho[:,:,i_k], U_shift[:,:]))

                I_E_dir[i_time] += np.real(U_h_H_U_E_dir[0,0])*np.real(solution[i_k, i_path, i_time, 0])
                I_E_dir[i_time] += np.real(U_h_H_U_E_dir[1,1])*np.real(solution[i_k, i_path, i_time, 3])
                I_E_dir[i_time] += 2*np.real(U_h_H_U_E_dir[0,1]*solution[i_k, i_path, i_time, 2])

                I_ortho[i_time] += np.real(U_h_H_U_ortho[0,0])*np.real(solution[i_k, i_path, i_time, 0])
                I_ortho[i_time] += np.real(U_h_H_U_ortho[1,1])*np.real(solution[i_k, i_path, i_time, 3])
                I_ortho[i_time] += 2*np.real(U_h_H_U_ortho[0,1]*solution[i_k, i_path, i_time, 2])

    return I_E_dir, I_ortho


def check_emission_wavep(paths, solution, wf_solution, E_dir, A_field, fermi_function):

    E_ort = np.array([E_dir[1], -E_dir[0]])

    n_time_steps = np.size(A_field)

    # I_E_dir is of size (number of time steps)
    I_E_dir = np.zeros(n_time_steps)
    I_ortho = np.zeros(n_time_steps)

    for i_time in range(n_time_steps):

        for i_path, path in enumerate(paths):
            path = np.array(path)
            kx_in_path = path[:, 0]
            ky_in_path = path[:, 1]
    
            kx_in_path_shifted = kx_in_path - A_field[i_time]*E_dir[0]
            ky_in_path_shifted = ky_in_path - A_field[i_time]*E_dir[1]

            h_deriv_x = ev_mat(sys.h_deriv[0], kx=kx_in_path_shifted, ky=ky_in_path_shifted)
            h_deriv_y = ev_mat(sys.h_deriv[1], kx=kx_in_path_shifted, ky=ky_in_path_shifted)
 
            h_deriv_E_dir = h_deriv_x*E_dir[0] + h_deriv_y*E_dir[1]
            h_deriv_ortho = h_deriv_x*E_ort[0] + h_deriv_y*E_ort[1]

            U = sys.wf(kx=kx_in_path, ky=ky_in_path)
            U_h = sys.wf_h(kx=kx_in_path, ky=ky_in_path)

            for i_k in range(np.size(kx_in_path)):

                U_wf_dynamics = np.matrix([[0+0*1j, 0+0*1j], [0+0*1j, 0+0*1j]])
                U_wf_dynamics[0,0] = wf_solution[i_k, i_path, i_time, 0]
                U_wf_dynamics[0,1] = wf_solution[i_k, i_path, i_time, 1]
                U_wf_dynamics[1,0] = wf_solution[i_k, i_path, i_time, 2]
                U_wf_dynamics[1,1] = wf_solution[i_k, i_path, i_time, 3]

                #HACK TO CHECK THE DENSITY MATRIX
                ff = fermi_function[i_k, i_path, i_time, 0]
                solution[i_k, i_path, i_time, 0] = np.abs(wf_solution[i_k, i_path, i_time, 0])**2 + ff*np.abs(wf_solution[i_k, i_path, i_time, 1])**2
                solution[i_k, i_path, i_time, 1] = ff*wf_solution[i_k, i_path, i_time, 3]*np.conj(wf_solution[i_k, i_path, i_time, 1]) + wf_solution[i_k, i_path, i_time, 2]*np.conj(wf_solution[i_k, i_path, i_time, 0])
                solution[i_k, i_path, i_time, 2] = ff*wf_solution[i_k, i_path, i_time, 1]*np.conj(wf_solution[i_k, i_path, i_time, 3]) + wf_solution[i_k, i_path, i_time, 0]*np.conj(wf_solution[i_k, i_path, i_time, 2])
                solution[i_k, i_path, i_time, 3] = np.abs(wf_solution[i_k, i_path, i_time, 2])**2 + ff*np.abs(wf_solution[i_k, i_path, i_time, 3])**2

                U_h_H_U_E_dir = np.matmul(U_h[:,:,i_k], np.matmul(h_deriv_E_dir[:,:,i_k], U[:,:,i_k]))
                U_h_H_U_ortho = np.matmul(U_h[:,:,i_k], np.matmul(h_deriv_ortho[:,:,i_k], U[:,:,i_k]))

                I_E_dir[i_time] += np.real(U_h_H_U_E_dir[0,0])*np.real(solution[i_k, i_path, i_time, 0])
                I_E_dir[i_time] += np.real(U_h_H_U_E_dir[1,1])*np.real(solution[i_k, i_path, i_time, 3])
                I_E_dir[i_time] += 2*np.real(U_h_H_U_E_dir[0,1]*solution[i_k, i_path, i_time, 2])

                I_ortho[i_time] += np.real(U_h_H_U_ortho[0,0])*np.real(solution[i_k, i_path, i_time, 0])
                I_ortho[i_time] += np.real(U_h_H_U_ortho[1,1])*np.real(solution[i_k, i_path, i_time, 3])
                I_ortho[i_time] += 2*np.real(U_h_H_U_ortho[0,1]*solution[i_k, i_path, i_time, 2])

    return I_E_dir, I_ortho


def emission_wavep(paths, solution, wf_solution, E_dir, A_field, fermi_function):

    E_ort = np.array([E_dir[1], -E_dir[0]])

    n_time_steps = np.size(A_field)

    # I_E_dir is of size (number of time steps)
    I_E_dir = np.zeros(n_time_steps)
    I_ortho = np.zeros(n_time_steps)

    for i_time in range(n_time_steps):

        for i_path, path in enumerate(paths):
            path = np.array(path)
            kx_in_path = path[:, 0]
            ky_in_path = path[:, 1]
    
            kx_in_path_shifted = kx_in_path - A_field[i_time]*E_dir[0]
            ky_in_path_shifted = ky_in_path - A_field[i_time]*E_dir[1]

            h_deriv_x = ev_mat(sys.h_deriv[0], kx=kx_in_path_shifted, ky=ky_in_path_shifted)
            h_deriv_y = ev_mat(sys.h_deriv[1], kx=kx_in_path_shifted, ky=ky_in_path_shifted)
 
            h_deriv_E_dir = h_deriv_x*E_dir[0] + h_deriv_y*E_dir[1]
            h_deriv_ortho = h_deriv_x*E_ort[0] + h_deriv_y*E_ort[1]

            U = sys.wf(kx=kx_in_path, ky=ky_in_path)
            U_h = sys.wf_h(kx=kx_in_path, ky=ky_in_path)

            for i_k in range(np.size(kx_in_path)):

                U_wf_dynamics = np.matrix([[0+0*1j, 0+0*1j], [0+0*1j, 0+0*1j]])
                U_wf_dynamics[0,0] = wf_solution[i_k, i_path, i_time, 0]
                U_wf_dynamics[0,1] = wf_solution[i_k, i_path, i_time, 1]
                U_wf_dynamics[1,0] = wf_solution[i_k, i_path, i_time, 2]
                U_wf_dynamics[1,1] = wf_solution[i_k, i_path, i_time, 3]

                U_h_H_U_E_dir = np.matmul(U_h[:,:,i_k], np.matmul(h_deriv_E_dir[:,:,i_k], U[:,:,i_k]))
                U_h_H_U_ortho = np.matmul(U_h[:,:,i_k], np.matmul(h_deriv_ortho[:,:,i_k], U[:,:,i_k]))

                U_h_wf_H_U_wf_E_dir = np.matmul(U_wf_dynamics.T[:,:], np.matmul(U_h_H_U_E_dir, np.conj(U_wf_dynamics[:,:])))
                U_h_wf_H_U_wf_ortho = np.matmul(U_wf_dynamics.T[:,:], np.matmul(U_h_H_U_ortho, np.conj(U_wf_dynamics[:,:])))

                I_E_dir[i_time] += np.real(U_h_wf_H_U_wf_E_dir[0,0])
                I_E_dir[i_time] += np.real(U_h_wf_H_U_wf_E_dir[1,1])*np.real(fermi_function[i_k, i_path, i_time, 0])

                I_ortho[i_time] += np.real(U_h_wf_H_U_wf_ortho[0,0])
                I_ortho[i_time] += np.real(U_h_wf_H_U_wf_ortho[1,1])*np.real(fermi_function[i_k, i_path, i_time, 0])

    return I_E_dir, I_ortho


def get_A_field(E0, w, t, alpha):
    '''
    Returns the analytical A-field as integration of the E-field
    '''
    w_eff = 4*np.pi*alpha*w
    return np.real(-alpha*E0*np.sqrt(np.pi)/2*np.exp(-w_eff**2/4)*(2+erf(t/2/alpha-1j*w_eff/2)-erf(-t/2/alpha-1j*w_eff/2)))


def f(t, y, kpath, dk, gamma1, gamma2, E0, B0, w, chirp, alpha, phase, do_B_field, 
      ecv_in_path, ev_in_path, ec_in_path, dipole_in_path, 
      A_in_path, Avv_in_path, Acc_in_path, gauge,
      kx_in_path, ky_in_path, E_dir, y0_np, Bcurv_in_B_dynamics, 
      dynamics_type):
    return fnumba(t, y, kpath, dk, gamma1, gamma2, E0, B0, w, chirp, alpha, phase, do_B_field, 
                  ecv_in_path,  ev_in_path, ec_in_path, dipole_in_path, 
                  A_in_path, Avv_in_path, Acc_in_path, gauge,
                  kx_in_path, ky_in_path, E_dir, y0_np, Bcurv_in_B_dynamics, 
                  dynamics_type)

@njit
def fnumba(t, y, kpath, dk, gamma1, gamma2, E0, B0, w, chirp, alpha, phase, do_B_field, 
           ecv_in_path, ev_in_path, ec_in_path, dipole_in_path, 
           A_in_path, Avv_in_path, Acc_in_path, gauge,
           kx_in_path, ky_in_path, E_dir, y0_np, Bcurv_in_B_dynamics, 
           dynamics_type):

    # x != y(t+dt)
    x = np.empty(np.shape(y), dtype=np.dtype('complex'))

    # Gradient term coefficient
    if gauge == 'length':
        D = driving_field(E0, t)/(2*dk)
    elif gauge == 'velocity':
        k_shift = (y[-1]).real
        kx_shift_path = kx_in_path+E_dir[0]*k_shift
        ky_shift_path = ky_in_path+E_dir[1]*k_shift
        ecv_in_path = sys.ecjit(kx=kx_shift_path, ky=ky_shift_path) \
            - sys.evjit(kx=kx_shift_path, ky=ky_shift_path)
        ev_in_path = sys.evjit(kx=kx_shift_path, ky=ky_shift_path)    
        ec_in_path = sys.ecjit(kx=kx_shift_path, ky=ky_shift_path)    

        di_00x = sys.di_00xjit(kx=kx_shift_path, ky=ky_shift_path)
        di_01x = sys.di_01xjit(kx=kx_shift_path, ky=ky_shift_path)
        di_11x = sys.di_11xjit(kx=kx_shift_path, ky=ky_shift_path)
        di_00y = sys.di_00yjit(kx=kx_shift_path, ky=ky_shift_path)
        di_01y = sys.di_01yjit(kx=kx_shift_path, ky=ky_shift_path)
        di_11y = sys.di_11yjit(kx=kx_shift_path, ky=ky_shift_path)
        # found that the dipole needs a complex conjugate
        dipole_in_path = E_dir[0]*di_01x + E_dir[1]*di_01y
        A_in_path = E_dir[0]*di_00x + E_dir[1]*di_00y \
            - (E_dir[0]*di_11x + E_dir[1]*di_11y)
        Avv_in_path = E_dir[0]*di_00x + E_dir[1]*di_00y
        Acc_in_path = E_dir[0]*di_11x + E_dir[1]*di_11y
        D = 0

    # Update the solution vector
    Nk_path = kpath.shape[0]
    for k in range(Nk_path):

        num_time_functions = 8

        i = num_time_functions*k
        if k == 0:
            m = num_time_functions*(k+1)
            n = num_time_functions*(Nk_path-1)
        elif k == Nk_path-1:
            m = 0
            n = num_time_functions*(k-1)
        else:
            m = num_time_functions*(k+1)
            n = num_time_functions*(k-1)

        # Energy term eband(i,k) the energy of band i at point k
        ecv = ecv_in_path[k]
        ev = -ecv_in_path[k]/2
        ec =  ecv_in_path[k]/2

        # Rabi frequency: w_R = d_12(k).E(t)
        dipole = dipole_in_path[k]
        wr = rabi(E0, t, dipole)
        wr_c = wr.conjugate()

        # Rabi frequency: w_R = (d_11(k) - d_22(k))*E(t)
        Berry_con_diff = A_in_path[k]
        wr_d_diag      = rabi(E0, t, Berry_con_diff)
        Berry_con_v    = Avv_in_path[k]
        wr_d_vv        = rabi(E0, t, Berry_con_v)
        Berry_con_c    = Acc_in_path[k]
        wr_d_cc        = rabi(E0, t, Berry_con_c)

        if dynamics_type == 'density_matrix_dynamics':

           # Update each component of the solution vector
           # i = f_v, i+1 = p_vc, i+2 = p_cv, i+3 = f_c
           # wr -> d_vc, wr_c -> d_vc
           x[i] = 2*(wr*y[i+1]).imag + D*(y[m] - y[n]) - gamma1*(y[i]-y0_np[i])
           x[i+1] = (1j*ecv - gamma2 + 1j*wr_d_diag)*y[i+1] - 1j*wr_c*(y[i]-y[i+3]) + D*(y[m+1] - y[n+1])
           x[i+2] = x[i+1].conjugate()
           x[i+3] = -2*(wr*y[i+1]).imag + D*(y[m+3] - y[n+3]) - gamma1*(y[i+3]-y0_np[i+3])
           x[i+4] = 0
           x[i+5] = 0
           x[i+6] = 0
           x[i+7] = 0

           if do_B_field:

               kx_shifted_path_v = kx_in_path[k] + np.real(y[i+4])
               ky_shifted_path_v = ky_in_path[k] + np.real(y[i+5])
               kx_shifted_path_c = kx_in_path[k] + np.real(y[i+6])
               ky_shifted_path_c = ky_in_path[k] + np.real(y[i+7])
        
               ev_dx = sys.ev_dx(kx=kx_shifted_path_v, ky=ky_shifted_path_v)
               ev_dy = sys.ev_dy(kx=kx_shifted_path_v, ky=ky_shifted_path_v)
               ec_dx = sys.ec_dx(kx=kx_shifted_path_c, ky=ky_shifted_path_c)
               ec_dy = sys.ec_dy(kx=kx_shifted_path_c, ky=ky_shifted_path_c)

               di_00x_B_field = sys.di_00xjit     (kx=kx_shifted_path_v, ky=ky_shifted_path_v)
               di_01x_B_field = sys.di_01xjit_offk(kx=kx_shifted_path_v, ky=ky_shifted_path_v, kxp=kx_shifted_path_c, kyp=ky_shifted_path_c)
               di_11x_B_field = sys.di_11xjit     (kx=kx_shifted_path_c, ky=ky_shifted_path_c)
               di_00y_B_field = sys.di_00yjit     (kx=kx_shifted_path_v, ky=ky_shifted_path_v)
               di_01y_B_field = sys.di_01yjit_offk(kx=kx_shifted_path_v, ky=ky_shifted_path_v, kxp=kx_shifted_path_c, kyp=ky_shifted_path_c)
               di_11y_B_field = sys.di_11yjit     (kx=kx_shifted_path_c, ky=ky_shifted_path_c)

               dipole_in_path_B = E_dir[0]*di_01x_B_field + E_dir[1]*di_01y_B_field
               A_in_path_B      = E_dir[0]*di_00x_B_field + E_dir[1]*di_00y_B_field - (E_dir[0]*di_11x_B_field + E_dir[1]*di_11y_B_field)
               wr_B             = rabi(E0, t, dipole_in_path_B)
               wr_B_c           = wr_B.conjugate()
               wr_d_diag_B      = rabi(E0, t, A_in_path_B)
               ecv_in_path_B    = sys.ecjit   (kx=kx_shifted_path_c, ky=ky_shifted_path_c) \
                                - sys.evjit   (kx=kx_shifted_path_v, ky=ky_shifted_path_v)
#               if Bcurv_in_B_dynamics: 
#                  Bcurv_v = sys.cu_00jit(kx=kx_shifted_path_v, ky=ky_shifted_path_v)
#                  Bcurv_c = sys.cu_11jit(kx=kx_shifted_path_c, ky=ky_shifted_path_c)
#               else:
               Bcurv_v = 0
               Bcurv_c = 0

               # use the unnecessary entry i+2 to compute the k-point shift 
               B_z = driving_field(B0, t)
               E_x = driving_field(E0, t) * E_dir[0]
               E_y = driving_field(E0, t) * E_dir[1]
               x[i]   = 2*(wr_B*y[i+1]).imag - gamma1*(y[i]-y0_np[i])
               x[i+1] = (1j*ecv_in_path_B - gamma2 + 1j*wr_d_diag_B)*y[i+1] - 1j*wr_B_c*(y[i]-y[i+3]) 
               x[i+2] = x[i+1].conjugate()
               x[i+3] = -2*(wr_B*y[i+1]).imag - gamma1*(y[i+3]-y0_np[i+3])
               # k_v_x
               x[i+4] = - driving_field(E0, t)*E_dir[0] - B_z*(ev_dy + Bcurv_v*E_x) / (1 - Bcurv_v*B_z)
               # k_v_y
               x[i+5] = - driving_field(E0, t)*E_dir[1] + B_z*(ev_dx - Bcurv_v*E_y) / (1 - Bcurv_v*B_z)
               # k_c_x
               x[i+6] = - driving_field(E0, t)*E_dir[0] - B_z*(ec_dy + Bcurv_c*E_x) / (1 - Bcurv_c*B_z)
               # k_v_y
               x[i+7] = - driving_field(E0, t)*E_dir[1] + B_z*(ec_dx - Bcurv_c*E_y) / (1 - Bcurv_c*B_z)

        elif dynamics_type == 'wavefunction_dynamics':

           # Update each component of the solution vector
           # i = U_vv, i+1 = U_vc, i+2 = U_cv, i+3 = U_cc
           x[i]   = (1j*ev - 1j*wr_d_vv)*y[i]   - 1j*wr  *y[i+2] 
           x[i+1] = (1j*ev - 1j*wr_d_vv)*y[i+1] - 1j*wr  *y[i+3] 
           x[i+2] = (1j*ec - 1j*wr_d_cc)*y[i+2] - 1j*wr_c*y[i]   
           x[i+3] = (1j*ec - 1j*wr_d_cc)*y[i+3] - 1j*wr_c*y[i+1] 
           x[i+4] = 0
           x[i+5] = 0
           x[i+6] = 0
           x[i+7] = 0

    # last component of x is the E-field to obtain the vector potential A(t)
    x[-1] = -driving_field(E0, t)

    return x

'''
OUT OF DATE/NOT FUNCTIONAL! FOR FUTURE WORK ON MAGNETIC FIELD IMPLEMENTATION.
'''
def f_matrix(t, y, kgrid, Nk, dk, gamma2, E0, w, alpha):
    '''
    Function driving the dynamics of the system.
    This is required as input parameter to the ode solver
    '''
    # Constant vector container
    b = []

    # Create propogation matrix for this time step
    for k1 in range(Nk): # Iterate down all the rows

        # Construct each block of the matrix
        '''
        Energy term eband(i,k) the energy of band i at point k
        '''
        ecv = eband(2, kgrid[k1]) - eband(1, kgrid[k1])

        '''
        Rabi frequency: w_R = w_R(i,j,k,t) = d_ij(k).E(t)
        Rabi frequency conjugate
        '''
        wr = rabi(1, 2, kgrid[k1], E0, w, t, alpha)
        wr_c = np.conjugate(wr)

        '''
        Brillouin zone drift term coefficient: E(t)*grad_k
        Coefficient for finite difference derivative.
        '''
        drift_coef = driving_field(E0, t)/(2.0*dk)

        '''
        Diagonal block of the propagation matrix M. Contains all terms not related to drift term. Case for electron-hole picture.
        '''
        diag_block = 1j*np.array([[0.0,wr,-wr_c,0.0],\
                                  [wr_c,-(ecv-1j*gamma2),0.0,wr_c],\
                                  [-wr,0.0,(ecv+1j*gamma2),-wr],\
                                  [0.0,wr,-wr_c,0.0]])

        '''
        Blocks for the forward and backwards portion of the finite difference derivative
        '''
        for_deriv = np.array([[drift_coef,0.0,0.0,0.0],\
                              [0.0,drift_coef,0.0,0.0],\
                              [0.0,0.0,drift_coef,0.0],\
                              [0.0,0.0,0.0,drift_coef]])
        back_deriv = np.array([[-drift_coef,0.0,0.0,0.0],\
                               [0.0,-drift_coef,0.0,0.0],\
                               [0.0,0.0,-drift_coef,0.0],\
                               [0.0,0.0,0.0,-drift_coef]])

        '''
        4x4 block of zeros. M is a very sparse matrix
        '''
        zero_block = np.zeros((4,4),dtype='float')

        '''
        Constructs the matrix M one block at a time. See notes for details
        '''
        # Put each block in their proper columns
        if (k1 == 0): # Construction of the first row
            M = np.concatenate((diag_block,for_deriv),axis=1) # Create first two columns
            for k2 in range(2,Nk-1): # From k3 to Nk-1
                M = np.concatenate((M,zero_block),axis=1) # Concatenate zero blocks
            M = np.concatenate((M,back_deriv),axis=1) # Concatenate final column
        elif (k1 == Nk-1): # Construction of the last row
            row = for_deriv # Create first column
            for k2 in range(1,Nk-2): # From k2 to Nk-2
                row = np.concatenate((row,zero_block),axis=1) # Concatenate zero blocks
            row = np.concatenate((row,back_deriv,diag_block),axis=1) # Concatenate final two columns
            M = np.concatenate((M,row),axis=0) # Concatenate this row to the matrix
        else: # Construction of all other rows
            # Initiate row variable
            if k1 == 1:
                row = back_deriv
            else:
                row = zero_block
            for k2 in range(1,Nk): # Scan across each column skipping the first
                if k2 == k1: # On the diagonal
                    row = np.concatenate((row,diag_block),axis=1) # Concatenate diagonal block
                elif k2 == k1-1: # If one behind the diagonal
                    row = np.concatenate((row,back_deriv),axis=1) # Concatenate back_deriv
                elif k2 == k1+1: # If one in front of diagonal
                    row = np.concatenate((row,for_deriv),axis=1)  # Concatenate for_deriv
                else: # If anywhere else
                    row = np.concatenate((row,zero_block),axis=1) # Concatenate zero_block
            M = np.concatenate((M,row),axis=0)
        '''
        'Constant' vector with leftover terms
        '''
        b.extend([0.0,-wr_c,wr,0.0])

    # Convert to numpy array
    b = 1j*np.array(b)

    # Calculate the timestep
    svec = np.dot(M, y) + b
    return svec


def shift_solution(solution, A_field, dk, dynamics_type):

    for i_time in range(np.size(A_field)):
        # shift of k index in the direction of the E-field 
        # (direction is already included in the paths)
        k_shift = (A_field[i_time]/dk).real
        k_index_shift_1 = int(int(np.abs(k_shift))*np.sign(k_shift))
        if(k_shift < 0): 
            k_index_shift_1 = k_index_shift_1 - 1
        k_index_shift_2 = k_index_shift_1 + 1
        weight_1      = k_index_shift_2 - k_shift
        weight_2      = 1-weight_1

        if dynamics_type == 'wavefunction_dynamics':
            if weight_1 > weight_2:
                weight_1 = 1
                weight_2 = 0
            else:
                weight_2 = 1
                weight_1 = 0

        n_kpoints = np.size(solution[:,0,0,0])

        # transfer to polar coordinates
        r   = np.abs(solution[:,:,i_time,:])
        phi = np.arctan2(np.imag(solution[:,:,i_time,:]), np.real(solution[:,:,i_time,:]))

        r   = weight_1*np.roll(r,   k_index_shift_1, axis=0) + weight_2*np.roll(r,   k_index_shift_2, axis=0)
        phi = weight_1*np.roll(phi, k_index_shift_1, axis=0) + weight_2*np.roll(phi, k_index_shift_2, axis=0)

#        solution[:, :, i_time, :] = weight_1*np.roll(solution[:,:,i_time,:], k_index_shift_1, axis=0) + \
#                                    weight_2*np.roll(solution[:,:,i_time,:], k_index_shift_2, axis=0)

        solution[:, :, i_time, :] = r*np.cos(phi) + 1j*r*np.sin(phi)

    return solution


def initial_condition(y0,e_fermi,temperature,e_c,i_k,dynamics_type):
    if dynamics_type == 'density_matrix_dynamics':
        if (temperature > 1e-5):
            fermi_function = 1/(np.exp((e_c[i_k]-e_fermi)/temperature)+1)
            y0.extend([1.0,0.0,0.0,fermi_function,0.0,0.0,0.0,0.0])
        else:
            y0.extend([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    elif dynamics_type == 'wavefunction_dynamics':
        y0.extend([1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0])


def BZ_plot(kpnts,a,b1,b2,E_dir,paths):

    R = 4.0*np.pi/(3*a)
    r = 2.0*np.pi/(np.sqrt(3)*a)

    BZ_fig = pl.figure(figsize=(10,10))
    ax = BZ_fig.add_subplot(111,aspect='equal')

    ax.add_patch(patches.RegularPolygon((0,0),6,radius=R,orientation=np.pi/6,fill=False))
    ax.add_patch(patches.RegularPolygon(b1,6,radius=R,orientation=np.pi/6,fill=False))
    ax.add_patch(patches.RegularPolygon(-b1,6,radius=R,orientation=np.pi/6,fill=False))
    ax.add_patch(patches.RegularPolygon(b2,6,radius=R,orientation=np.pi/6,fill=False))
    ax.add_patch(patches.RegularPolygon(-b2,6,radius=R,orientation=np.pi/6,fill=False))
    ax.add_patch(patches.RegularPolygon(b1+b2,6,radius=R,orientation=np.pi/6,fill=False))
    ax.add_patch(patches.RegularPolygon(-b1-b2,6,radius=R,orientation=np.pi/6,fill=False))

    ax.arrow(-0.5*E_dir[0],-0.5*E_dir[1],E_dir[0],E_dir[1],width=0.005,alpha=0.5,label='E-field')

    pl.scatter(0,0,s=15,c='black')
    pl.text(0.01,0.01,r'$\Gamma$')
    pl.scatter(r*np.cos(-np.pi/6),r*np.sin(-np.pi/6),s=15,c='black')
    pl.text(r*np.cos(-np.pi/6)+0.01,r*np.sin(-np.pi/6)-0.05,r'$M$')
    pl.scatter(R,0,s=15,c='black')
    pl.text(R,0.02,r'$K$')
    pl.scatter(kpnts[:,0],kpnts[:,1], s=15)
    pl.xlim(-5.0/a,5.0/a)
    pl.ylim(-5.0/a,5.0/a)
    pl.xlabel(r'$k_x$ ($1/a_0$)')
    pl.ylabel(r'$k_y$ ($1/a_0$)')

    for path in paths:
        path = np.array(path)
        pl.plot(path[:,0],path[:,1])

    return

if __name__ == "__main__":
    main()
