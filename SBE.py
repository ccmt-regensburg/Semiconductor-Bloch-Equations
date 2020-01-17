import params
import numpy as np
from numba import njit
import matplotlib.pyplot as pl
from matplotlib import patches
from scipy.integrate import ode

import hfsbe.dipole
import hfsbe.example
import hfsbe.utility

'''
TO DO ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
- legacy matrix method not compatible with two-dimensional case.
- current function not compatible with K-paths.
- arbitrary (circular) polarization (big task).
- plots for two-dimensional case
- emission spectrum for arbitrary direction
- change testing outputs for 1d case 
- generalize energy band gradient (adaptable for 2d case)
'''

def main():
    
    # RETRIEVE PARAMETERS
    ###############################################################################################
    # Unit converstion factors
    fs_conv = params.fs_conv
    E_conv = params.E_conv 
    THz_conv = params.THz_conv
    amp_conv = params.amp_conv
    eV_conv = params.eV_conv
    
    # Set parameters
    Nk_in_path = params.Nk_in_path                    # Number of kpoints in each of the two paths
    rel_dist_to_Gamma = params.rel_dist_to_Gamma      # relative distance (in units of 2pi/a) of both paths to Gamma
    a = params.a                                      # Lattice spacing
    length_path_in_BZ = params.length_path_in_BZ      # 
    angle_inc_E_field = params.angle_inc_E_field
    e_fermi = params.e_fermi*eV_conv
    temperature = params.temperature*eV_conv
    k_cut = params.k_cut
    Nk = 2*Nk_in_path                                 # Total number of k points, we have 2 paths
    E0 = params.E0*E_conv                             # Driving field amplitude
    w = params.w*THz_conv                             # Driving frequency
    alpha = params.alpha*fs_conv                      # Gaussian pulse width
    T2 = params.T2*fs_conv                            # Damping time
    gamma2 = 1/T2                                     # Gamma parameter
    t0 = int(params.t0*fs_conv)                       # Initial time condition
    tf = int(params.tf*fs_conv)                       # Final time
    dt = params.dt*fs_conv                            # Integration time step
    dt_out = 1/(2*params.dt)                          # Solution output time step
    test = params.test                                # Testing flag for Travis

    # Hamiltonian parameters
    C0 = 0
    C2 = 0
    A  = 0.1974
    R  = 11.06

    # USER OUTPUT
    ###############################################################################################
    print("Solving for...")
    if Nk < 20:
        print("***WARNING***: Convergence issues may result from Nk < 20")
    if params.dt > 1.0:
        print("***WARNING***: Time-step may be insufficiently small. Use dt < 1.0fs")
    print("Number of k-points              = " + str(Nk))
    print("Driving amplitude (MV/cm)[a.u.] = " + "(" + '%.6f'%(E0/E_conv) + ")" + "[" + '%.6f'%(E0) + "]")
    print("Pulse Frequency (THz)[a.u.]     = " + "(" + '%.6f'%(w/THz_conv) + ")" + "[" + '%.6f'%(w) + "]")
    print("Pulse Width (fs)[a.u.]          = " + "(" + '%.6f'%(alpha/fs_conv) + ")" + "[" + '%.6f'%(alpha) + "]")
    print("Damping time (fs)[a.u.]         = " + "(" + '%.6f'%(T2/fs_conv) + ")" + "[" + '%.6f'%(T2) + "]")
    print("Total time (fs)[a.u.]           = " + "(" + '%.6f'%((tf-t0)/fs_conv) + ")" + "[" + '%.5i'%(tf-t0) + "]")
    print("Time step (fs)[a.u.]            = " + "(" + '%.6f'%(dt/fs_conv) + ")" + "[" + '%.6f'%(dt) + "]")
    
    # INITIALIZATIONS
    ###############################################################################################
    # Form the E-field direction
    E_dir = np.array([np.cos(angle_inc_E_field/360*2*np.pi),np.sin(angle_inc_E_field/360*2*np.pi)])

    # Form the Brillouin zone in consideration
    dk, kpnts, paths = mesh(params, E_dir)

    # Number of time steps, time vector
    Nt = int((tf-t0)/dt)
    t_constructed = False
    #t = np.linspace(t0,tf,Nt)
    
    # Solution containers
    t                           = []
    solution                    = []    
    dipole_E_dir_for_print      = []
    dipole_diag_E_dir_for_print = []
    dipole_x_for_print          = []
    dipole_y_for_print          = []
    val_band_for_print          = []
    cond_band_for_print         = []
    phase_1                     = []
    phase_2                     = []

    # Initialize the ode solver
    solver = ode(f, jac=None).set_integrator('zvode', method='bdf', max_step= dt)

    # Get initialize sympy bandstructure, energies/derivatives, dipoles
    # Topological cone call
    bite = hfsbe.example.BiTe(C0=C0,C2=C2,A=A,R=R,kcut=k_cut)
    # Trivial cone call
    #bite = hfsbe.example.BiTeTrivial(C0=C0,C2=C2,R=R,vf=A,kcut=k_cut)
    h, ef, wf, ediff = bite.eigensystem(gidx=1)
    dipole = hfsbe.dipole.SymbolicDipole(h, ef, wf)

    # SOLVING 
    ###############################################################################################
    # Iterate through each path in the Brillouin zone
    for path in paths:

        # This step is needed for the gamma-K paths, as they are not uniform in length, thus not suitable to be stored as numpy array initially.
        path = np.array(path)

        # Solution container for the current path
        path_solution = []

        # Retrieve the set of k-points for the current path
        kx_in_path = path[:,0]
        ky_in_path = path[:,1]

        # Calculate the dipole components along the path
        di_x,di_y = dipole.evaluate(kx_in_path, ky_in_path)

        # Calculate the dot products E_dir.d_nm(k). To be multiplied by E-field magnitude later. 
        # A[0,1,:] means 0-1 offdiagonal element
        dipole_in_path = E_dir[0]*di_x[0,1,:] + E_dir[1]*di_y[0,1,:]
        A_in_path      = E_dir[0]*di_x[0,0,:] + E_dir[1]*di_y[0,0,:] - (E_dir[0]*di_x[1,1,:] + E_dir[1]*di_y[1,1,:])

        # in bite.evaluate, there is also an interpolation done if b1, b2 are provided and a cutoff radius
        bandstruct  = bite.evaluate_energy(kx_in_path, ky_in_path)
        ecv_in_path = bandstruct[1] - bandstruct[0]

        # Initialize the values of of each k point vector (rho_nn(k), rho_nm(k), rho_mn(k), rho_mm(k))
        y0 = []
        for i_k, k in enumerate(path):
            initial_condition(y0,e_fermi,temperature,bandstruct[1],i_k)

        # Set the initual values and function parameters for the current kpath
        solver.set_initial_value(y0,t0).set_f_params(path,dk,gamma2,E0,w,alpha,ecv_in_path, \
               dipole_in_path,A_in_path)

        # Propagate through time
        ti = 0
        while solver.successful() and ti < Nt:
            # User output of integration progress
            if (ti%1000) == 0:
                print('{:5.2f}%'.format(ti/Nt*100))

            # Integrate one integration time step
            solver.integrate(solver.t + dt)

            # Save solution each output step 
            if ti%dt_out == 0:
                path_solution.append(solver.y)
                # Construct time array only once
                if not t_constructed:
                    t.append(solver.t)

            # Increment time counter
            ti += 1

        # Flag that time array has been built up
        t_constructed = True

        # Append path solutions to the total solution arrays
        solution.append(path_solution)
        dipole_E_dir_for_print.append(dipole_in_path)
        dipole_diag_E_dir_for_print.append(A_in_path)
        dipole_x_for_print.append(di_x[0,1,:])
        dipole_y_for_print.append(di_y[0,1,:])
        val_band_for_print.append(bandstruct[0])
        cond_band_for_print.append(bandstruct[1])
#        phase_2.append((ky_in_path+1j*kx_in_path)/(kx_in_path[:]**2+ky_in_path[:]**2)**0.5)
#        phase_1.append((ky_in_path-1j*kx_in_path)/(kx_in_path[:]**2+ky_in_path[:]**2)**0.5)

    # Convert solution and time array to numpy arrays
    t        = np.array(t)
    solution = np.array(solution)
    # Slice solution along each path for easier observable calculation
    solution = np.array_split(solution,Nk_in_path,axis=2)
    solution = np.array(solution)

    print(np.shape(t))
    print(np.shape(solution))

    # Now the solution array is structred as: first index is kx-index, second is ky-index, third is timestep, fourth is f_h, p_he, p_eh, f_e
    
    # COMPUTE OBSERVABLES
    ###############################################################################################
    bandstruct_deriv_for_print = []
    dipole_ortho_for_print    = []

    # Calculate the parallel and orthogonal components 
    J_E_dir, J_ortho = current(paths, solution[:,:,:,0], solution[:,:,:,3], bite, path, t, alpha, E_dir, bandstruct_deriv_for_print)
    P_E_dir, P_ortho = polarization(paths, solution[:,:,:,1], dipole, E_dir, dipole_ortho_for_print)

    I_E_dir, I_ortho = (diff(t,P_E_dir) + J_E_dir)*Gaussian_envelope(t,alpha), \
                       (diff(t,P_ortho) + J_ortho)*Gaussian_envelope(t,alpha)

    Ir = []
    angles = np.linspace(0,2.0*np.pi,360)
    for angle in angles:
        Ir.append((I_E_dir*np.cos(angle) + I_ortho*np.sin(-angle)))

    dt_out   = t[1]-t[0]
    freq     = np.fft.fftshift(np.fft.fftfreq(np.size(t),d=dt_out))
    Iw_E_dir = np.fft.fftshift(np.fft.fft(I_E_dir, norm='ortho'))
    Iw_ortho = np.fft.fftshift(np.fft.fft(I_ortho, norm='ortho'))
    Iw_r     = np.fft.fftshift(np.fft.fft(Ir, norm='ortho'))
    Pw_E_dir = np.fft.fftshift(np.fft.fft(diff(t,P_E_dir), norm='ortho'))
    Pw_ortho = np.fft.fftshift(np.fft.fft(diff(t,P_ortho), norm='ortho'))
    Jw_E_dir = np.absolute(np.fft.fftshift(np.fft.fft(J_E_dir*Gaussian_envelope(t,alpha), norm='ortho')))
    Jw_ortho = np.absolute(np.fft.fftshift(np.fft.fft(J_ortho*Gaussian_envelope(t,alpha), norm='ortho')))
    fw_0     = np.fft.fftshift(np.fft.fft(solution[:,0,:,0], norm='ortho'),axes=(1,))

    if not test:
        fig1, (axE,ax1,ax1a,ax2) = pl.subplots(1,4)
        t_lims = (-10*alpha/fs_conv, 10*alpha/fs_conv)
        freq_lims = (0,25)
        log_limits = (10e-15,100)
        axE.set_xlim(t_lims)
        axE.plot(t/fs_conv,driving_field(E0,w,t,alpha)/E_conv)
        axE.set_xlabel(r'$t$ in fs')
        axE.set_ylabel(r'$E$-field in MV/cm')
        ax1.set_xlim(t_lims)
        ax1.plot(t/fs_conv,P_E_dir)
        ax1.plot(t/fs_conv,P_ortho)
        ax1.set_xlabel(r'$t$ in fs')
        ax1.set_ylabel(r'$P$ in atomic units $\parallel \mathbf{E}_{in}$ (blue), $\bot \mathbf{E}_{in}$ (orange)')
        ax1a.set_xlim(t_lims)
        ax1a.plot(t/fs_conv,diff(t,P_E_dir))
        ax1a.plot(t/fs_conv,diff(t,P_ortho))
        ax1a.set_xlabel(r'$t$ in fs')
        ax1a.set_ylabel(r'$\dot P$ in atomic units $\parallel \mathbf{E}_{in}$ (blue), $\bot \mathbf{E}_{in}$ (orange)')
        ax2.set_xlim(t_lims)
        ax2.plot(t/fs_conv,J_E_dir)
        ax2.plot(t/fs_conv,J_ortho)
        ax2.set_xlabel(r'$t$ in fs')
        ax2.set_ylabel(r'$J$ in atomic units $\parallel \mathbf{E}_{in}$ (blue), $\bot \mathbf{E}_{in}$ (orange)')

        fig1a, (ax3a,ax3b,ax3) = pl.subplots(1,3)
        ax3a.set_xlim(freq_lims)
        ax3a.set_ylim(log_limits)
        ax3a.semilogy(freq/w,np.abs(Pw_E_dir))
        ax3a.semilogy(freq/w,np.abs(Pw_ortho))
        ax3a.set_xlabel(r'Frequency $\omega/\omega_0$')
        ax3a.set_ylabel(r'$[\dot P](\omega)$ (interband) in a.u. $\parallel \mathbf{E}_{in}$ (blue), $\bot \mathbf{E}_{in}$ (orange)')
        ax3b.set_xlim(freq_lims)
        ax3b.set_ylim(log_limits)
        ax3b.semilogy(freq/w,np.abs(Jw_E_dir))
        ax3b.semilogy(freq/w,np.abs(Jw_ortho))
        ax3b.set_xlabel(r'Frequency $\omega/\omega_0$')
        ax3b.set_ylabel(r'$[\dot P](\omega)$ (intraband) in a.u. $\parallel \mathbf{E}_{in}$ (blue), $\bot \mathbf{E}_{in}$ (orange)')
        ax3.set_xlim(freq_lims)
        ax3.set_ylim(log_limits)
        ax3.semilogy(freq/w,np.abs(Iw_E_dir))
        ax3.semilogy(freq/w,np.abs(Iw_ortho))
        ax3.set_xlabel(r'Frequency $\omega/\omega_0$')
        ax3.set_ylabel(r'$[\dot P](\omega)$ (total = emitted E-field) in a.u.')

        # High-harmonic emission polar plots
        fig2a = pl.figure()
        i_loop = 1
        i_max  = 20
        while i_loop <= i_max:
            freq_indices = np.argwhere(np.logical_and(freq/w > float(i_loop)-0.1, freq/w < float(i_loop)+0.1))
            freq_index   = freq_indices[int(np.size(freq_indices)/2)]
            pax          = fig2a.add_subplot(1,i_max,i_loop,projection='polar')
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

        fig3, (ax3_0,ax3_0a,ax3_1,ax3_2,ax3_3,ax3_4,ax3_5) = pl.subplots(1,7)
        kp_array = length_path_in_BZ*np.linspace(-0.5 + (1/(2*Nk_in_path)), 0.5 - (1/(2*Nk_in_path)), num = Nk_in_path)
        ax3_0.plot(kp_array,np.real(dipole_E_dir_for_print[0]))
        ax3_0.plot(kp_array,np.real(dipole_E_dir_for_print[1]), linestyle='dashed')
        ax3_0.set_xlabel(r'$k$-point in path ($1/a_0$)')
        ax3_0.set_ylabel(r'Dipole real part Re $(\vec{d}(k)\cdot\vec{e}_E)$ (a.u.) in path 0/1')
        ax3_0a.plot(kp_array,np.imag(dipole_E_dir_for_print[0]))
        ax3_0a.plot(kp_array,np.imag(dipole_E_dir_for_print[1]), linestyle='dashed')
        ax3_0a.set_xlabel(r'$k$-point in path ($1/a_0$)')
        ax3_0a.set_ylabel(r'Dipole im. part Im$(\vec{d}(k)\cdot\vec{e}_E)$ (a.u.) in path 0/1')
        # we have a strange additional first index 0 here due to an append
        ax3_1.plot(kp_array,dipole_ortho_for_print[0][0])
        ax3_1.plot(kp_array,dipole_ortho_for_print[0][1])
        ax3_1.set_xlabel(r'$k$-point in path ($1/a_0$)')
        ax3_1.set_ylabel(r'Dipole real part Re $\vec{d}(k)\cdot\vec{e}_{ortho}$ (a.u.) in path 0/1')
        ax3_2.plot(kp_array,np.imag(dipole_ortho_for_print[0][0]))
        ax3_2.plot(kp_array,np.imag(dipole_ortho_for_print[0][1]))
        ax3_2.set_xlabel(r'$k$-point in path ($1/a_0$)')
        ax3_2.set_ylabel(r'Dipole imag part Im $\vec{d}(k)\cdot\vec{e}_{ortho}$ (a.u.) in path 0/1')
        ax3_3.plot(kp_array,dipole_x_for_print[0])
        ax3_3.plot(kp_array,dipole_x_for_print[1])
        ax3_3.set_ylabel(r'Dipole $d_x(k)$ (a.u.) in path 0/1')
        ax3_4.plot(kp_array,dipole_y_for_print[0])
        ax3_4.plot(kp_array,dipole_y_for_print[1])
        ax3_4.set_ylabel(r'Dipole $d_y(k)$ (a.u.) in path 0/1')
        ax3_5.plot(kp_array,np.real(dipole_diag_E_dir_for_print[0]))
        ax3_5.plot(kp_array,np.real(dipole_diag_E_dir_for_print[1]), linestyle='dashed')
        ax3_5.set_xlabel(r'$k$-point in path ($1/a_0$)')
        ax3_5.set_ylabel(r'Real part Re $([\vec{d}_{vv}(k)-\vec{d}_{cc}]\cdot\vec{e}_E)$ (a.u.) in path 0/1')

#        fig3a, (ax3a_0,ax3a_0a,ax3a_1,ax3a_2) = pl.subplots(1,4)
#        kp_array = length_path_in_BZ*np.linspace(-0.5 + (1/(2*Nk_in_path)), 0.5 - (1/(2*Nk_in_path)), num = Nk_in_path)
#        ax3a_0.plot(kp_array,np.real(dipole_E_dir_for_print[0]*phase_1[0]))
#        ax3a_0.plot(kp_array,np.real(dipole_E_dir_for_print[1]*phase_1[1]), linestyle='dashed')
#        ax3a_0.set_xlabel(r'$k$-point in path ($1/a_0$)')
#        ax3a_0.set_ylabel(r'WITH PHASE 1 Dipole real part Re $(\vec{d}(k)\cdot\vec{e}_E)$ (a.u.) in path 0/1')
#        ax3a_0a.plot(kp_array,np.imag(dipole_E_dir_for_print[0]*phase_1[0]))
#        ax3a_0a.plot(kp_array,np.imag(dipole_E_dir_for_print[1]*phase_1[1]), linestyle='dashed')
#        ax3a_0a.set_xlabel(r'$k$-point in path ($1/a_0$)')
#        ax3a_0a.set_ylabel(r'WITH PHASE 1 Dipole im. part Im$(\vec{d}(k)\cdot\vec{e}_E)$ (a.u.) in path 0/1')
#        ax3a_1.plot(kp_array,dipole_ortho_for_print[0][0]*phase_1[0])
#        ax3a_1.plot(kp_array,dipole_ortho_for_print[0][1]*phase_1[1])
#        ax3a_1.set_xlabel(r'$k$-point in path ($1/a_0$)')
#        ax3a_1.set_ylabel(r'WITH PHASE 1 Dipole real part Re $\vec{d}(k)\cdot\vec{e}_{ortho}$ (a.u.) in path 0/1')
#        ax3a_2.plot(kp_array,np.imag(dipole_ortho_for_print[0][0]*phase_1[0]))
#        ax3a_2.plot(kp_array,np.imag(dipole_ortho_for_print[0][1]*phase_1[1]))
#        ax3a_2.set_xlabel(r'$k$-point in path ($1/a_0$)')
#        ax3a_2.set_ylabel(r'WITH PHASE 1 Dipole imag part Im $\vec{d}(k)\cdot\vec{e}_{ortho}$ (a.u.) in path 0/1')

        E_ort = np.array([E_dir[1], -E_dir[0]])

        fig4, (ax4_1,ax4_2,ax4_3,ax4_4,ax4_5,ax4_6) = pl.subplots(1,6)
        ax4_1.plot(kp_array,1.0/eV_conv*val_band_for_print[0])
        ax4_1.plot(kp_array,1.0/eV_conv*cond_band_for_print[0])
        ax4_2.plot(kp_array,1.0/eV_conv*val_band_for_print[1])
        ax4_2.plot(kp_array,1.0/eV_conv*cond_band_for_print[1])
        ax4_1.set_xlabel(r'$k$-point in path 0 ($1/a_0$)')
        ax4_1.set_ylabel(r'Bandstruc. $\varepsilon(k)$ (eV)')
        ax4_2.set_xlabel(r'$k$-point in path 1 ($1/a_0$)')
        ax4_2.set_ylabel(r'Bandstruc. $\varepsilon(k)$ (eV)')
        ax4_3.plot(kp_array,1.0/eV_conv*(bandstruct_deriv_for_print[0][0]*E_dir[0] + bandstruct_deriv_for_print[0][1]*E_dir[1]))
        ax4_3.plot(kp_array,1.0/eV_conv*(bandstruct_deriv_for_print[0][2]*E_dir[0] + bandstruct_deriv_for_print[0][3]*E_dir[1]))
        ax4_3.set_xlabel(r'$k$-point in path 0 ($1/a_0$)')
        ax4_3.set_ylabel(r'$\partial \varepsilon_{v/c}(k)/\partial k_{\parallel \mathbf{E}}$ (eV*$a_0$) in path 0 (blue: v, orange: c)')
        ax4_4.plot(kp_array,1.0/eV_conv*(bandstruct_deriv_for_print[1][0]*E_dir[0] + bandstruct_deriv_for_print[1][1]*E_dir[1]))
        ax4_4.plot(kp_array,1.0/eV_conv*(bandstruct_deriv_for_print[1][2]*E_dir[0] + bandstruct_deriv_for_print[1][3]*E_dir[1]))
        ax4_4.set_xlabel(r'$k$-point in path 0 ($1/a_0$)')
        ax4_4.set_ylabel(r'$\partial \varepsilon_{v/c}(k)/\partial k_{\parallel \mathbf{E}}$ (eV*$a_0$) in path 1')
        ax4_5.plot(kp_array,1.0/eV_conv*(bandstruct_deriv_for_print[0][0]*E_ort[0] + bandstruct_deriv_for_print[0][1]*E_ort[1]))
        ax4_5.plot(kp_array,1.0/eV_conv*(bandstruct_deriv_for_print[0][2]*E_ort[0] + bandstruct_deriv_for_print[0][3]*E_ort[1]))
        ax4_5.set_xlabel(r'$k$-point in path 0 ($1/a_0$)')
        ax4_5.set_ylabel(r'$\partial \varepsilon_{v/c}(k)/\partial k_{\bot \mathbf{E}}$ (eV*$a_0$) in path 0 (blue: v, orange: c)')
        ax4_6.plot(kp_array,1.0/eV_conv*(bandstruct_deriv_for_print[1][0]*E_ort[0] + bandstruct_deriv_for_print[1][1]*E_ort[1]))
        ax4_6.plot(kp_array,1.0/eV_conv*(bandstruct_deriv_for_print[1][2]*E_ort[0] + bandstruct_deriv_for_print[1][3]*E_ort[1]))
        ax4_6.set_xlabel(r'$k$-point in path 0 ($1/a_0$)')
        ax4_6.set_ylabel(r'$\partial \varepsilon_{v/c}(k)/\partial k_{\bot \mathbf{E}}$ (eV*$a_0$) in path 1 (blue: v, orange: c)')

        # Countour plots of occupations and gradients of occupations
        fig5 = pl.figure()
        X, Y = np.meshgrid(t/fs_conv,kp_array)
        pl.contourf(X, Y, np.real(solution[:,0,:,3]), 100)
        pl.colorbar().set_label(r'$f_e(k)$ in path 0')
        pl.xlim([-5*alpha/fs_conv,10*alpha/fs_conv])
        pl.xlabel(r'$t\;(fs)$')
        pl.ylabel(r'$k$')
        pl.tight_layout()
#
#        fig6 = pl.figure()
#        X, Y = np.meshgrid(t/fs_conv,kp_array)
#        pl.contourf(X, Y, np.real(solution[:,0,:,3]-solution[:,1,:,3]), 100)
#        pl.colorbar().set_label(r'$f_{e,path 0}(k) - f_{e,path 1}(k)$')
#        pl.xlim([-5*alpha/fs_conv,10*alpha/fs_conv])
#        pl.xlabel(r'$t\;(fs)$')
#        pl.ylabel(r'$k$')
#        pl.tight_layout()
#
        fig6 = pl.figure()
        X, Y = np.meshgrid(t/fs_conv,kp_array)
        pl.contourf(X, Y, np.real(solution[:,0,:,0]), 100)
        pl.colorbar().set_label(r'$f_h(k)$ in path 0')
        pl.xlim([-5*alpha/fs_conv,10*alpha/fs_conv])
        pl.xlabel(r'$t\;(fs)$')
        pl.ylabel(r'$k$')
        pl.tight_layout()
#
#        fig7 = pl.figure()
#        X, Y = np.meshgrid(t/fs_conv,kp_array)
#        pl.contourf(X, Y, np.real(solution[:,1,:,3]), 100)
#        pl.colorbar().set_label(r'$f_e(k)$ in path 1')
#        pl.xlim([-5*alpha/fs_conv,10*alpha/fs_conv])
#        pl.xlabel(r'$t\;(fs)$')
#        pl.ylabel(r'$k$')
#        pl.tight_layout()
#
#        fig8 = pl.figure()
#        X, Y = np.meshgrid(t/fs_conv,kp_array)
#        pl.contourf(X, Y, np.real(solution[:,1,:,0]), 100)
#        pl.colorbar().set_label(r'$f_h(k)$ in path 1')
#        pl.xlim([-5*alpha/fs_conv,10*alpha/fs_conv])
#        pl.xlabel(r'$t\;(fs)$')
#        pl.ylabel(r'$k$')
#        pl.tight_layout()
#
#        fig9 = pl.figure()
#        X, Y = np.meshgrid(freq/w,kp_array)
#        pl.contourf(X, Y, fw_0, 100)
#        pl.colorbar().set_label(r'log $f_h(k)$ in path 0')
#        pl.xlim(freq_lims)
#        pl.xlabel(r'$\omega/\omega_0$')
#        pl.ylabel(r'$k$')
#        pl.tight_layout()
#
#        print("freq =", freq)
#        print("size freq =", np.size(freq))
#
#        print("omega(100,1000,10000,100000) =", freq[100], freq[1000], freq[10000], freq[100000])
#        print("omega 1 2 3 =", freq[75100]/w, freq[75200]/w, freq[75300]/w, freq[75400]/w)
#
#        fig10, (ax10_0) = pl.subplots(1,1)
#        ax10_0.plot(kp_array,fw_0[:,75100])
#        ax10_0.plot(kp_array,fw_0[:,75200])
#        ax10_0.plot(kp_array,fw_0[:,75300])
#        ax10_0.plot(kp_array,fw_0[:,75400])
#        ax10_0.set_xlabel(r'$k$-point in path ($1/a_0$)')
#        ax10_0.set_ylabel(r'$f_h(k,\omega)$ in path 0 at $\omega = $')
#
#        for i_phase, phase in enumerate(phase_2[0]):
#            solution[i_phase,0,:,1] = solution[i_phase,0,:,1] * phase
#        for i_phase, phase in enumerate(phase_2[1]):
#            solution[i_phase,1,:,1] = solution[i_phase,1,:,1] * phase
#
#        fig11 = pl.figure()
#        X, Y = np.meshgrid(t/fs_conv,kp_array)
#        pl.contourf(X, Y, np.real(solution[:,0,:,1]), 100)
#        pl.colorbar().set_label(r'$Re(p_{cv}(k))$ in path 0')
#        pl.xlim([-5*alpha/fs_conv,10*alpha/fs_conv])
#        pl.xlabel(r'$t\;(fs)$')
#        pl.ylabel(r'$k$')
#        pl.tight_layout()
#
#        fig12 = pl.figure()
#        X, Y = np.meshgrid(t/fs_conv,kp_array)
#        pl.contourf(X, Y, np.imag(solution[:,0,:,1]), 100)
#        pl.colorbar().set_label(r'$Im(p_{cv}(k))$ in path 0')
#        pl.xlim([-5*alpha/fs_conv,10*alpha/fs_conv])
#        pl.xlabel(r'$t\;(fs)$')
#        pl.ylabel(r'$k$')
#        pl.tight_layout()
#
#        fig12a = pl.figure()
#        X, Y = np.meshgrid(t/fs_conv,kp_array)
#        pl.contourf(X, Y, np.abs(solution[:,0,:,1]), 100)
#        pl.colorbar().set_label(r'$|p_{cv}(k)|$ in path 0')
#        pl.xlim([-5*alpha/fs_conv,10*alpha/fs_conv])
#        pl.xlabel(r'$t\;(fs)$')
#        pl.ylabel(r'$k$')
#        pl.tight_layout()
#
#        fig13 = pl.figure()
#        X, Y = np.meshgrid(t/fs_conv,kp_array)
#        pl.contourf(X, Y, np.real(solution[:,1,:,1]), 100)
#        pl.colorbar().set_label(r'$Re(p_{cv}(k))$ in path 1')
#        pl.xlim([-5*alpha/fs_conv,10*alpha/fs_conv])
#        pl.xlabel(r'$t\;(fs)$')
#        pl.ylabel(r'$k$')
#        pl.tight_layout()
#
#        fig14 = pl.figure()
#        X, Y = np.meshgrid(t/fs_conv,kp_array)
#        pl.contourf(X, Y, np.imag(solution[:,1,:,1]), 100)
#        pl.colorbar().set_label(r'$Im(p_{cv}(k))$ in path 1')
#        pl.xlim([-5*alpha/fs_conv,10*alpha/fs_conv])
#        pl.xlabel(r'$t\;(fs)$')
#        pl.ylabel(r'$k$')
#        pl.tight_layout()

        BZ_plot(kpnts,a)
        path_plot(paths)

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
    print ("vec_k_path =", vec_k_path)

    vec_k_ortho = 2.0*np.pi/a*rel_dist_to_Gamma*np.array([E_dir[1],-E_dir[0]])
    print ("vec_k_ortho =", vec_k_ortho)

    # Containers for the mesh, and BZ directional paths
    mesh = []
    paths = []

    # Create the kpoint mesh and the paths
    for path_index in [-1,1]:
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

    return dk, np.array(mesh), paths

def hex_mesh(Nk1, Nk2, a, b1, b2, test):
    # Calculate the alpha values needed based on the size of the Brillouin zone
    if Nk2 == 1: # 1d case set by Nk2 value
        # alpha1 zero to ensure 1d lane running through gamma-point
        alpha1 = [0.0]
        alpha2 = np.linspace(-0.5 + (1/(2*Nk1)), 0.5 - (1/(2*Nk1)), num = Nk1)
        # Rescale reciprocal lattice vector to original 1d case
        #b2 = np.array([0,1]) 
    else: 
        alpha1 = np.linspace(-0.5 + (1/(2*Nk1)), 0.5 - (1/(2*Nk1)), num = Nk1)
        alpha2 = np.linspace(-0.5 + (1/(2*Nk2)), 0.5 - (1/(2*Nk2)), num = Nk2)
    
    def is_in_hex(p,a):
        # Returns true if the point is in the hexagonal BZ.
        # Checks if the absolute values of x and y components of p are within the first quadrant of the hexagon.
        x = np.abs(p[0])
        y = np.abs(p[1])
        return ((y <= 2.0*np.pi/(np.sqrt(3)*a)) and (np.sqrt(3.0)*x + y <= 4*np.pi/(np.sqrt(3)*a)))

    # Containers for the mesh, and BZ directional paths
    mesh = []
    M_paths = []
    K_paths = []

    # Create the Monkhorst-Pack mesh
    for a1 in alpha1:
        # Container for a single gamma-M path
        path_M = []
        for a2 in alpha2:
            # Create a k-point
            kpoint = a1*b1 + a2*b2
            # If the current point is in the BZ, append it to the mesh and path_M
            if (is_in_hex(kpoint,a)):
                mesh.append(kpoint)
                path_M.append(kpoint)
            # If the current point is NOT in the BZ, reflect is along the appropriate axis to get it in the BZ, then append.
            else:
                while (is_in_hex(kpoint,a) != True):
                    if (kpoint[1] < -2*np.pi/(np.sqrt(3)*a)):
                        kpoint += b1
                    elif (kpoint[1] > 2*np.pi/(np.sqrt(3)*a)):
                        kpoint -= b1
                    elif (np.sqrt(3)*kpoint[0] + kpoint[1] > 4*np.pi/(np.sqrt(3)*a)): #Crosses top-right
                        kpoint -= b1 + b2
                    elif (-np.sqrt(3)*kpoint[0] + kpoint[1] < -4*np.pi/(np.sqrt(3)*a)): #Crosses bot-right
                        kpoint -= b2
                    elif (np.sqrt(3)*kpoint[0] + kpoint[1] < -4*np.pi/(np.sqrt(3)*a)): #Crosses bot-left
                        kpoint += b1 + b2
                    elif (-np.sqrt(3)*kpoint[0] + kpoint[1] > 4*np.pi/(np.sqrt(3)*a)): #Crosses top-left
                        kpoint += b2
                mesh.append(kpoint)
                path_M.append(kpoint) 

        # Append the a1'th path to the paths array
        M_paths.append(path_M)

    # Temp mesh array that gets progressively deleted
    path_K_mesh = np.array(mesh)

    while np.size(path_K_mesh) != 0:
        path_K = []
        # Determine the top horizontal points in the current path_K_mesh (a gamma-K path) and their arguments.
        y_top = np.amax(path_K_mesh[:,1])
        y_top_args = np.argwhere(np.logical_and(path_K_mesh[:,1] <= y_top+1/(4*Nk1), path_K_mesh[:,1] >= y_top-1/(4*Nk1)))

        # Iterate through the arguments of the top most path, adding each point to the path. Delete these points as they are added.
        for arg in y_top_args:
            path_K.append(path_K_mesh[arg,:])

        path_K_mesh = np.delete(path_K_mesh, y_top_args, 0)

        K_paths.append(path_K)
    
    return np.array(mesh), M_paths, K_paths

@njit
def driving_field(E0, w, t, alpha):
    '''
    Returns the instantaneous driving pulse field
    '''
    #return E0*np.sin(2.0*np.pi*w*t)
    return E0*np.exp(-t**2.0/(2.0*alpha)**2)*np.sin(2.0*np.pi*w*t)

@njit
def rabi(k,E0,w,t,alpha,dipole_in_path):
    '''
    Rabi frequency of the transition. Calculated from dipole element and driving field
    '''
    return dipole_in_path[k]*driving_field(E0, w, t, alpha)

def diff(x,y):
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

def Gaussian_envelope(t,alpha):
    '''
    Function to multiply a Function f(t) before Fourier transform 
    to ensure no step in time between t_final and t_final + delta
    '''
    return np.exp(-t**2.0/(2.0*1.0*alpha)**2)  

def polarization(paths,pcv,dipole,E_dir,dipole_ortho_for_print):
    '''
    Calculates the polarization as: P(t) = sum_n sum_m sum_k [d_nm(k)p_nm(k)]
    Dipole term currently a crude model to get a vector polarization
    '''
    E_ort = np.array([E_dir[1], -E_dir[0]])

    d_E_dir, d_ortho = [],[]
    for path in paths:

        path = np.array(path)

        kx_in_path = path[:,0]
        ky_in_path = path[:,1]

        dipole_in_path, dipole_in_path = dipole.evaluate(kx_in_path, ky_in_path)

        d_E_dir.append(dipole_in_path[0,1,:]*E_dir[0] + dipole_in_path[0,1,:]*E_dir[1])
        d_ortho.append(dipole_in_path[0,1,:]*E_ort[0] + dipole_in_path[0,1,:]*E_ort[1])

    dipole_ortho_for_print.append(d_ortho)

    d_E_dir_swapped = np.swapaxes(d_E_dir,0,1)
    d_ortho_swapped = np.swapaxes(d_ortho,0,1)

    P_E_dir = 2*np.real(np.tensordot(d_E_dir_swapped,pcv,2))
    P_ortho = 2*np.real(np.tensordot(d_ortho_swapped,pcv,2))

    return P_E_dir, P_ortho


def current(paths,fv,fc,bite,path,t,alpha,E_dir,bandstruct_deriv_for_print):
    '''
    Calculates the current as: J(t) = sum_k sum_n [j_n(k)f_n(k,t)]
    where j_n(k) != (d/dk) E_n(k)
    '''

    E_ort = np.array([E_dir[1], -E_dir[0]])

    # Calculate the gradient analytically at each k-point
    J_E_dir, J_ortho = [], []
    je_E_dir,je_ortho,jh_E_dir,jh_ortho = [],[],[],[]
    for path in paths:
        path = np.array(path)
        kx_in_path = path[:,0]
        ky_in_path = path[:,1]
        bandstruct_deriv = bite.evaluate_ederivative(kx_in_path, ky_in_path)
        bandstruct_deriv_for_print.append(bandstruct_deriv)
        #0: v, x   1: v,y   2: c, x  3: c, y
        je_E_dir.append(bandstruct_deriv[2]*E_dir[0] + bandstruct_deriv[3]*E_dir[1])
        je_ortho.append(bandstruct_deriv[2]*E_ort[0] + bandstruct_deriv[3]*E_ort[1])
        jh_E_dir.append(bandstruct_deriv[0]*E_dir[0] + bandstruct_deriv[1]*E_dir[1])
        jh_ortho.append(bandstruct_deriv[0]*E_ort[0] + bandstruct_deriv[1]*E_ort[1])

    je_E_dir_swapped = np.swapaxes(je_E_dir,0,1)
    je_ortho_swapped = np.swapaxes(je_ortho,0,1)
    jh_E_dir_swapped = np.swapaxes(jh_E_dir,0,1)
    jh_ortho_swapped = np.swapaxes(jh_ortho,0,1)

    # we need tensordot for contracting the first two indices (2 kpoint directions)
    J_E_dir = np.tensordot(je_E_dir_swapped,fc,2) + np.tensordot(jh_E_dir_swapped,fv,2)
    J_ortho = np.tensordot(je_ortho_swapped,fc,2) + np.tensordot(jh_ortho_swapped,fv,2)

    # Return the real part of each component
    return np.real(J_E_dir), np.real(J_ortho)


def f(t, y, kpath, dk, gamma2, E0, w, alpha, ecv_in_path, dipole_in_path, A_in_path):
    return fnumba(t, y, kpath, dk, gamma2, E0, w, alpha, ecv_in_path, dipole_in_path, A_in_path)


@njit
def fnumba(t, y, kpath, dk, gamma2, E0, w, alpha, ecv_in_path, dipole_in_path, A_in_path):

    # x != y(t+dt)
    x = np.empty(np.shape(y), dtype=np.dtype('complex'))
    
    # Gradient term coefficient
    D = driving_field(E0, w, t, alpha)/(2*dk)

    # Update the solution vector
    Nk_path = kpath.shape[0]
    for k in range(Nk_path):

        i = 4*k
        if k == 0:
            m = 4*(k+1)
            n = 4*(Nk_path-1)
        elif k == Nk_path-1:
            m = 0
            n = 4*(k-1)
        else:
            m = 4*(k+1)
            n = 4*(k-1)

        #Energy term eband(i,k) the energy of band i at point k
        ecv = ecv_in_path[k]

        # Rabi frequency: w_R = d_12(k).E(t)
        # Rabi frequency conjugate
        wr          = dipole_in_path[k]*D
        wr_c        = np.conjugate(wr)

        # Rabi frequency: w_R = (d_11(k) - d_22(k))*E(t)
        wr_d_diag   = A_in_path[k]*D

        # Update each component of the solution vector
        x[i]   = 2*np.imag(wr*y[i+1]) + D*(y[m] - y[n])
        x[i+1] = ( -1j*ecv - gamma2 + 1j*wr_d_diag)*y[i+1] - 1j*wr_c*(y[i]-y[i+3]) + D*(y[m+1] - y[n+1])
        x[i+2] = np.conjugate(x[i+1])
        x[i+3] = -2*np.imag(wr*y[i+1]) + D*(y[m+3] - y[n+3])

    return x

def initial_condition(y0,e_fermi,temperature,e_c,i_k):

    if (temperature > 1e-5):
      y0.extend([1.0,0.0,0.0,1/(np.exp((e_c[i_k]-e_fermi)/temperature)+1)])
    else:
      y0.extend([1.0,0.0,0.0,0.0])

def BZ_plot(kpnts,a):
    
    R = 4.0*np.pi/(3*a)
    r = 2.0*np.pi/(np.sqrt(3)*a)

    BZ_fig = pl.figure()
    ax = BZ_fig.add_subplot(111,aspect='equal')
    
#    ax.add_patch(patches.RegularPolygon((0,0),6,radius=R,orientation=np.pi/6,fill=False))

    pl.scatter(0,0,s=15,c='black')
    pl.text(0.05,0.05,r'$\Gamma$')
#    pl.scatter(R,0,s=15,c='black')
#    pl.text(R,0.05,r'$K$')
#    pl.scatter(r*np.cos(np.pi/6),-r*np.sin(np.pi/6),s=15,c='black')
#    pl.text(r*np.cos(np.pi/6),-r*np.sin(np.pi/6)-0.2,r'$M$')
    pl.scatter(kpnts[:,0],kpnts[:,1], s=15)
    pl.xlim(-8.0/a,8.0/a)
    pl.ylim(-8.0/a,8.0/a)
    pl.xlabel(r'$k_x$ ($1/a_0$)')
    pl.ylabel(r'$k_y$ ($1/a_0$)')
    
    return

def path_plot(paths):

    for path in paths:
        path = np.array(path)
        pl.plot(path[:,0], path[:,1])

    return

if __name__ == "__main__":
    main()

