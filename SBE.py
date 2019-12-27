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
    E_dir = params.E_dir                              # Reciprocal lattice vector
    scale_dipole = params.scale_dipole                # phenomenological rescaling of the dipole moments to match the experiments
    Nk = 2*Nk_in_path                                 # Total number of k points, we have 2 paths
    E0 = params.E0*E_conv                             # Driving field amplitude
    w = params.w*THz_conv                             # Driving frequency
    alpha = params.alpha*fs_conv                      # Gaussian pulse width
    align = params.align                              # Pulse polarization direction
    T2 = params.T2*fs_conv                            # Damping time
    gamma2 = 1/T2                                     # Gamma parameter
    t0 = int(params.t0*fs_conv)                       # Initial time condition
    tf = int(params.tf*fs_conv)                       # Final time
    dt = params.dt*fs_conv                            # Integration time step
    test = params.test                                # Testing flag for Travis
    matrix_method = params.matrix_method              # 'Vector' or 'matrix' updates in f(t,y)

    # USER OUTPUT
    ###############################################################################################
    print("Solving for...")
    if Nk < 20:
        print("***WARNING***: Convergence issues may result from Nk < 20")
    if params.dt > 1.0:
        print("***WARNING***: Time-step may be insufficiently small. Use dt < 1.0fs")
    if matrix_method:
        print("*** USING MATRIX METHOD SOLVER ***")
    print("Number of k-points              = " + str(Nk))
    print("Driving amplitude (MV/cm)[a.u.] = " + "(" + '%.6f'%(E0/E_conv) + ")" + "[" + '%.6f'%(E0) + "]")
    print("Pulse Frequency (THz)[a.u.]     = " + "(" + '%.6f'%(w/THz_conv) + ")" + "[" + '%.6f'%(w) + "]")
    print("Pulse Width (fs)[a.u.]          = " + "(" + '%.6f'%(alpha/fs_conv) + ")" + "[" + '%.6f'%(alpha) + "]")
    print("Damping time (fs)[a.u.]         = " + "(" + '%.6f'%(T2/fs_conv) + ")" + "[" + '%.6f'%(T2) + "]")
    print("Total time (fs)[a.u.]           = " + "(" + '%.6f'%((tf-t0)/fs_conv) + ")" + "[" + '%.5i'%(tf-t0) + "]")
    print("Time step (fs)[a.u.]            = " + "(" + '%.6f'%(dt/fs_conv) + ")" + "[" + '%.6f'%(dt) + "]")
    print("Driving field polarization      = " + "Gamma-" + str(align))
    
    # INITIALIZATIONS
    ###############################################################################################
    # Form the Brillouin zone in consideration
    dk, kpnts, paths = mesh(params)

    print("dk =", dk)

    # Number of time steps, time vector
    Nt = int((tf-t0)/dt)
    t = np.linspace(t0,tf,Nt)

    # containers
    solution            = []    
    dip_dot_E_for_print = []
    dipole_x_for_print  = []
    dipole_y_for_print  = []
    val_band_for_print  = []
    cond_band_for_print = []

    # Initialize ode solver according to chosen method
    if matrix_method:
        solver = ode(f_matrix, jac=None).set_integrator('zvode', method='bdf', max_step= dt)
    else:
        solver = ode(f, jac=None).set_integrator('zvode', method='bdf', max_step= dt)

    # Get band structure, its derivative and the dipole
#    bite = hfsbe.example.BiTe(b1=b1, b2=b2, default_params=True)
#    bite = hfsbe.example.BiTe(default_params=True)
#    bite = hfsbe.example.BiTe(C0=0.0,C2=0.0,R=0,A=0.1974,default_params=True)
    bite = hfsbe.example.BiTe(C0=0.0,C2=0.0,R=0,A=0.1974)


    h, ef, wf, ediff = bite.eigensystem()
    dipole = hfsbe.dipole.SymbolicDipole(h, ef, wf)

    # cutoff for k for setting dipole to zero if |k| exceeds k_cut (in paper: 0.04 A^-1 = 0.02 a.u.^-1)
    k_cut = 2.0

    # SOLVING 
    ###############################################################################################
    # Iterate through each path in the Brillouin zone
    for path in paths:

        # This step is needed for the gamma-K paths, as they are not uniform in length, thus not suitable to be stored as numpy array initially.
        path = np.array(path)

        # Solution container for the current path
        path_solution = []

        # Initialize the values of of each k point vector (rho_nn(k), rho_nm(k), rho_mn(k), rho_mm(k))
        y0 = []
        for k in path:
            y0.extend([0.0,0.0,0.0,0.0])

        kx_in_path = path[:,0]
        ky_in_path = path[:,1]

        Ax,Ay             = dipole.evaluate(kx_in_path, ky_in_path)
        # A[0,1,:] means 0-1 offdiagonal element
        dipole_in_path    = E_dir[0]*Ax[0,1,:] + E_dir[1]*Ay[0,1,:]

        # in bite.evaluate, there is also an interpolation done if b1, b2 are provided and a cutoff radius
        bandstruc         = bite.evaluate_energy(kx_in_path, ky_in_path)
        bandstruc_in_path = bandstruc[1] - bandstruc[0]

        # Set the initual values and function parameters for the current kpath
        solver.set_initial_value(y0,t0).set_f_params(path,dk,gamma2,E0,w,alpha,bandstruc_in_path,dipole_in_path,k_cut,scale_dipole)

        # Propagate through time
        ti = 0
        while solver.successful() and ti < Nt:
            if (ti%10000) == 0:
                print('{:5.2f}%'.format(ti/Nt*100))
            solver.integrate(solver.t + dt)
            path_solution.append(solver.y)
            ti += 1

        solution.append(path_solution)
        dip_dot_E_for_print.append(dipole_in_path)
        dipole_x_for_print.append(Ax[0,1,:])
        dipole_y_for_print.append(Ay[0,1,:])
        val_band_for_print.append(bandstruc[0])
        cond_band_for_print.append(bandstruc[1])

    # Slice solution along each path for easier observable calculation
    solution = np.array(solution)

    print("shape of solution before reshaping =", np.shape(solution))

    solution = np.array_split(solution,Nk_in_path,axis=2)
    solution = np.array(solution)

    print("shape of solution =", np.shape(solution))

    # Now the solution array is structred as: first index is kx-index, second is ky-index, third is timestep, fourth is f_h, p_he, p_eh, f_e
    
    # COMPUTE OBSERVABLES
    ###############################################################################################
    # Electrons occupations

    bandstruc_deriv_for_print = []

    Jx, Jy = current(paths, solution[:,:,:,0], solution[:,:,:,3], bite, path, t, alpha, bandstruc_deriv_for_print)
    Px, Py = polarization(paths, solution[:,:,:,1], solution[:,:,:,2], dipole)
    Ix, Iy = (diff(t,Px) + Jx)*Gaussian_envelope(t,alpha), (diff(t,Py) + Jy)*Gaussian_envelope(t,alpha)
#    Ix, Iy = diff(t,Px) + Jx, diff(t,Py) + Jy

    Ir = []
    angles = np.linspace(0,2.0*np.pi,72)
    for angle in angles:
        Ir.append((Ix*np.cos(angle))**2.0 + (Iy*np.sin(angle))**2.0)
        
    freq = np.fft.fftshift(np.fft.fftfreq(Nt,d=dt))
    Iw_x = np.fft.fftshift(np.fft.fft(Ix, norm='ortho'))
    Iw_y = np.fft.fftshift(np.fft.fft(Iy, norm='ortho'))
    Iw_r = np.fft.fftshift(np.fft.fft(Ir, norm='ortho'))
    Pw_x = np.fft.fftshift(np.fft.fft(diff(t,Px)*Gaussian_envelope(t,alpha), norm='ortho'))
    Pw_y = np.fft.fftshift(np.fft.fft(diff(t,Py)*Gaussian_envelope(t,alpha), norm='ortho'))
    Jw_x = np.absolute(np.fft.fftshift(np.fft.fft(Jx*Gaussian_envelope(t,alpha), norm='ortho')))
    Jw_y = np.absolute(np.fft.fftshift(np.fft.fft(Jy*Gaussian_envelope(t,alpha), norm='ortho')))
    fw_0 = np.fft.fftshift(np.fft.fft(solution[:,0,:,0], norm='ortho'),axes=(1,))

    print("shape bs_deriv =", np.shape(bandstruc_deriv_for_print))
    print ("eV_conv =", 1.0/eV_conv)

    if not test:
        fig1, (axE,ax1,ax2,ax3a,ax3b,ax3) = pl.subplots(1,6)
        t_lims = (-10*alpha/fs_conv, 10*alpha/fs_conv)
        freq_lims = (0,30)
        axE.set_xlim(t_lims)
        axE.plot(t/fs_conv,driving_field(E0,w,t,alpha)/E_conv)
        axE.set_xlabel(r'$t$ in fs')
        axE.set_ylabel(r'$E$-field in MV/cm')
        ax1.set_xlim(t_lims)
        ax1.plot(t/fs_conv,Px)
        ax1.plot(t/fs_conv,Py)
        ax2.set_xlim(t_lims)
        ax2.plot(t/fs_conv,Jx/amp_conv)
        ax2.plot(t/fs_conv,Jy/amp_conv)
        ax3a.set_xlim(freq_lims)
        ax3a.semilogy(freq/w,np.abs(Pw_x))
        ax3a.semilogy(freq/w,np.abs(Pw_y))
        ax3a.set_xlabel(r'Frequency $\omega/\omega_0$')
        ax3a.set_ylabel(r'$[\dot P](\omega)$ x (blue), y (yellow)')
        ax3b.set_xlim(freq_lims)
        ax3b.semilogy(freq/w,np.abs(Jw_x))
        ax3b.semilogy(freq/w,np.abs(Jw_y))
        ax3b.set_xlabel(r'Frequency $\omega/\omega_0$')
        ax3b.set_ylabel(r'$J(\omega)$ x (blue), y (yellow)')
        ax3.set_xlim(freq_lims)
        ax3.semilogy(freq/w,np.abs(Iw_x))
        ax3.semilogy(freq/w,np.abs(Iw_y))
        ax3.set_xlabel(r'Frequency $\omega/\omega_0$')
        ax3.set_ylabel(r'Normalized emission spectrum')


        f5 = np.argwhere(np.logical_and(freq/w > 9.9, freq/w < 10.1))
        f125 = np.argwhere(np.logical_and(freq/w > 13.9, freq/w < 14.1))
        f15= np.argwhere(np.logical_and(freq/w > 17.9, freq/w < 18.1))
        f_5 = f5[int(np.size(f5)/2)]
        f_125 = f125[int(np.size(f125)/2)]
        f_15 = f15[int(np.size(f15)/2)]

        fig2 = pl.figure()
        pax0 = fig2.add_subplot(131,projection='polar')
        pax0.plot(angles,Iw_r[:,f_5])
        pax1 = fig2.add_subplot(132,projection='polar')
        pax1.plot(angles,Iw_r[:,f_125])
        pax2 = fig2.add_subplot(133,projection='polar')
        pax2.plot(angles,Iw_r[:,f_15])

        fig3, (ax3_0,ax3_3,ax3_4) = pl.subplots(1,3)
        kp_array = length_path_in_BZ*np.linspace(-0.5 + (1/(2*Nk_in_path)), 0.5 - (1/(2*Nk_in_path)), num = Nk_in_path)
        ax3_0.plot(kp_array,scale_dipole*dip_dot_E_for_print[0])
        ax3_0.plot(kp_array,scale_dipole*dip_dot_E_for_print[1])
        ax3_0.set_xlabel(r'$k$-point in path ($1/a_0$)')
        ax3_0.set_ylabel(r'Scaled dipole $\vec{d}(k)\cdot\vec{e}_E$ (a.u.) in path 0/1')
        ax3_3.plot(kp_array,scale_dipole*dipole_x_for_print[0])
        ax3_3.plot(kp_array,scale_dipole*dipole_x_for_print[1])
        ax3_3.set_ylabel(r'Scaled dipole $d_x(k)$ (a.u.) in path 0/1')
        ax3_4.plot(kp_array,scale_dipole*dipole_y_for_print[0])
        ax3_4.plot(kp_array,scale_dipole*dipole_y_for_print[1])
        ax3_4.set_ylabel(r'Scaled dipole $d_y(k)$ (a.u.) in path 0/1')

        print("shape bs_deriv =", np.shape(bandstruc_deriv_for_print))

        fig4, (ax4_1,ax4_2,ax4_3,ax4_4,ax4_5,ax4_6) = pl.subplots(1,6)
        ax4_1.plot(kp_array,1.0/eV_conv*val_band_for_print[0])
        ax4_1.plot(kp_array,1.0/eV_conv*cond_band_for_print[0])
        ax4_2.plot(kp_array,1.0/eV_conv*val_band_for_print[1])
        ax4_2.plot(kp_array,1.0/eV_conv*cond_band_for_print[1])
        ax4_1.set_xlabel(r'$k$-point in path 0 ($1/a_0$)')
        ax4_1.set_ylabel(r'Bandstruc. $\varepsilon(k)$ (eV)')
        ax4_2.set_xlabel(r'$k$-point in path 1 ($1/a_0$)')
        ax4_2.set_ylabel(r'Bandstruc. $\varepsilon(k)$ (eV)')
        ax4_3.plot(kp_array,1.0/eV_conv*bandstruc_deriv_for_print[0][0])
        ax4_3.plot(kp_array,1.0/eV_conv*bandstruc_deriv_for_print[0][2])
        ax4_3.set_xlabel(r'$k$-point in path 0 ($1/a_0$)')
        ax4_3.set_ylabel(r'$\partial \varepsilon_{v/c}(k)/\partial k_x$ (eV*$a_0$) in path 0')
        ax4_4.plot(kp_array,1.0/eV_conv*bandstruc_deriv_for_print[1][0])
        ax4_4.plot(kp_array,1.0/eV_conv*bandstruc_deriv_for_print[1][2])
        ax4_4.set_xlabel(r'$k$-point in path 0 ($1/a_0$)')
        ax4_4.set_ylabel(r'$\partial \varepsilon_{v/c}(k)/\partial k_x$ (eV*$a_0$) in path 1')
        ax4_5.plot(kp_array,1.0/eV_conv*bandstruc_deriv_for_print[0][1])
        ax4_5.plot(kp_array,1.0/eV_conv*bandstruc_deriv_for_print[0][3])
        ax4_5.set_xlabel(r'$k$-point in path 0 ($1/a_0$)')
        ax4_5.set_ylabel(r'$\partial \varepsilon_{v/c}(k)/\partial k_y$ (eV*$a_0$) in path 0')
        ax4_6.plot(kp_array,1.0/eV_conv*bandstruc_deriv_for_print[1][1])
        ax4_6.plot(kp_array,1.0/eV_conv*bandstruc_deriv_for_print[1][3])
        ax4_6.set_xlabel(r'$k$-point in path 0 ($1/a_0$)')
        ax4_6.set_ylabel(r'$\partial \varepsilon_{v/c}(k)/\partial k_y$ (eV*$a_0$) in path 1')

        # Countour plots of occupations and gradients of occupations
        fig5 = pl.figure()
        X, Y = np.meshgrid(t/fs_conv,kp_array)
        pl.contourf(X, Y, np.real(solution[:,0,:,3]), 100)
        pl.colorbar().set_label(r'$f_e(k)$ in path 0')
        pl.xlim([-5*alpha/fs_conv,10*alpha/fs_conv])
        pl.xlabel(r'$t\;(fs)$')
        pl.ylabel(r'$k$')
        pl.tight_layout()

#        fig6 = pl.figure()
#        X, Y = np.meshgrid(t/fs_conv,kp_array)
#        pl.contourf(X, Y, np.real(solution[:,0,:,0]), 100)
#        pl.colorbar().set_label(r'$f_h(k)$ in path 0')
#        pl.xlim([-5*alpha/fs_conv,10*alpha/fs_conv])
#        pl.xlabel(r'$t\;(fs)$')
#        pl.ylabel(r'$k$')
#        pl.tight_layout()
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

#        fig9 = pl.figure()
#        X, Y = np.meshgrid(freq/w,kp_array)
#        pl.contourf(X, Y, fw_0, 100)
#        pl.colorbar().set_label(r'log $f_h(k)$ in path 0')
#        pl.xlim(freq_lims)
#        pl.xlabel(r'$\omega/\omega_0$')
#        pl.ylabel(r'$k$')
#        pl.tight_layout()

#        print("freq =", freq)
#        print("size freq =", np.size(freq))
#
#        print("omega(100,1000,10000,100000) =", freq[100], freq[1000], freq[10000], freq[100000])
#        print("omega 1 2 3 =", freq[75100]/w, freq[75200]/w, freq[75300]/w, freq[75400]/w)
#
#
#        fig10, (ax10_0) = pl.subplots(1,1)
#        ax10_0.plot(kp_array,fw_0[:,75100])
#        ax10_0.plot(kp_array,fw_0[:,75200])
#        ax10_0.plot(kp_array,fw_0[:,75300])
#        ax10_0.plot(kp_array,fw_0[:,75400])
#        ax10_0.set_xlabel(r'$k$-point in path ($1/a_0$)')
#        ax10_0.set_ylabel(r'$f_h(k,\omega)$ in path 0 at $\omega = $')

        fig11 = pl.figure()
        X, Y = np.meshgrid(t/fs_conv,kp_array)
        pl.contourf(X, Y, np.real(solution[:,0,:,1]), 100)
        pl.colorbar().set_label(r'$Re(p_cv(k))$ in path 0')
        pl.xlim([-5*alpha/fs_conv,10*alpha/fs_conv])
        pl.xlabel(r'$t\;(fs)$')
        pl.ylabel(r'$k$')
        pl.tight_layout()

        fig12 = pl.figure()
        X, Y = np.meshgrid(t/fs_conv,kp_array)
        pl.contourf(X, Y, np.real(solution[:,0,:,2]), 100)
        pl.colorbar().set_label(r'$Re(p_vc(k))$ in path 0')
        pl.xlim([-5*alpha/fs_conv,10*alpha/fs_conv])
        pl.xlabel(r'$t\;(fs)$')
        pl.ylabel(r'$k$')
        pl.tight_layout()

        fig13 = pl.figure()
        X, Y = np.meshgrid(t/fs_conv,kp_array)
        pl.contourf(X, Y, np.real(solution[:,1,:,1]), 100)
        pl.colorbar().set_label(r'$Re(p_cv(k))$ in path 1')
        pl.xlim([-5*alpha/fs_conv,10*alpha/fs_conv])
        pl.xlabel(r'$t\;(fs)$')
        pl.ylabel(r'$k$')
        pl.tight_layout()

        fig12 = pl.figure()
        X, Y = np.meshgrid(t/fs_conv,kp_array)
        pl.contourf(X, Y, np.real(solution[:,1,:,2]), 100)
        pl.colorbar().set_label(r'$Re(p_vc(k))$ in path 1')
        pl.xlim([-5*alpha/fs_conv,10*alpha/fs_conv])
        pl.xlabel(r'$t\;(fs)$')
        pl.ylabel(r'$k$')
        pl.tight_layout()

        BZ_plot(kpnts,a)
        path_plot(paths)

        pl.show()

#    occu_filename = str('occu_Nk1{}_Nk2{}_w{:4.2f}_E{:4.2f}_a{:4.2f}_dt{:3.2f}.dat').format(Nk1,Nk2,w/THz_conv,E0/E_conv,alpha/fs_conv,dt)
#    np.save(occu_filename, N_elec)
#    curr_filename = str('curr_Nk1{}_Nk2{}_w{:4.2f}_E{:4.2f}_a{:4.2f}_dt{:3.2f}.dat').format(Nk1,Nk2,w/THz_conv,E0/E_conv,alpha/fs_conv,dt)
#    np.save(curr_filename, [t/fs_conv, Jx, Jy])
#    pol_filename = str('pol_Nk1{}_Nk2{}_w{:4.2f}_E{:4.2f}_a{:4.2f}_dt{:3.2f}.dat').format(Nk1,Nk2,w/THz_conv,E0/E_conv,alpha/fs_conv,dt)
#    np.save(pol_filename, [t/fs_conv,Px,Py])
#    emis_filename = str('emis_Nk1{}_Nk2{}_w{:4.2f}_E{:4.2f}_a{:4.2f}_dt{:3.2f}.dat').format(Nk1,Nk2,w/THz_conv,E0/E_conv,alpha/fs_conv,dt)
#    np.save(emis_filename, [freq/w, Iw_x, Iw_y])
    
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
def mesh(params):
    Nk_in_path = params.Nk_in_path                    # Number of kpoints in each of the two paths
    rel_dist_to_Gamma = params.rel_dist_to_Gamma      # relative distance (in units of 2pi/a) of both paths to Gamma
    a = params.a                                      # Lattice spacing
    length_path_in_BZ = params.length_path_in_BZ      # 
    E_dir = params.E_dir                              # Reciprocal lattice vector

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

def eband(n, kx, ky):
    '''
    Returns the energy of a band n, from the k-point.
    Band structure modeled as (e.g.)...
    E1(k) = (-1eV) + (1eV)exp(-10*k^2)*(4k^2-1)^2(4k^2+1)^2
    '''
    envelope = ((2.0*kx**2 + 2.0*ky**2 - 1.0)**2.0)*((2.0*kx**2 +  2.0*ky**2 + 1.0)**2.0)
    if (n==1):   # Valence band
        #return np.zeros(np.shape(k)) # Flat structure
        #return (-1.0/27.211)+(1.0/27.211)*np.exp(-10.0*kx**2 - 10.0*ky**2)*envelope
        return (-1.0/27.211)+(1.0/27.211)*np.exp(-0.4*kx**2 - 0.4*ky**2)#*envelope

    elif (n==2): # Conduction band
        #return (2.0/27.211)*np.ones(np.shape(k)) # Flat structure
        #return (3.0/27.211)-(1.0/27.211)*np.exp(-5.0*kx**2 - 5.0*ky**2)*envelope
        return (3.0/27.211)-(1.0/27.211)*np.exp(-0.2*kx**2 - 0.2*ky**2)#*envelope

def dipole(kx, ky):
    '''
    Returns the dipole matrix element for making the transition from band n to band m at k = kx*e_kx + ky*e_ky
    '''
    return 1.0 #Eventually inputting some data from other calculations

@njit
def driving_field(E0, w, t, alpha):
    '''
    Returns the instantaneous driving pulse field
    '''
    #return E0*np.sin(2.0*np.pi*w*t)
    return E0*np.exp(-t**2.0/(2.0*alpha)**2)*np.sin(2.0*np.pi*w*t)

@njit
def rabi(n,m,kx,ky,k,E0,w,t,alpha,dipole_in_path,k_cut,scale_dipole):
    '''
    Rabi frequency of the transition. Calculated from dipole element and driving field
    '''
#    return dipole(kx,ky)*driving_field(E0, w, t, alpha)
#   Jan: Hack, we set all dipole elements to zero if they exceed the cutoff region
#    print ("kx, ky, dipole =", kx, ky, dipole_in_path[1,0,k])
    if(kx**2+ky**2 < k_cut**2):
#      return dipole_in_path[1,0,k]*driving_field(E0, w, t, alpha)
#      return np.real(dipole_in_path[1,0,k]*driving_field(E0, w, t, alpha))
      return np.real(dipole_in_path[k])*scale_dipole*driving_field(E0, w, t, alpha)
    else:
      return 0.0

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
    return np.exp(-t**2.0/(2.0*3.0*alpha)**2)  

def polarization(paths,pvc,pcv,dipole):
    '''
    Calculates the polarization as: P(t) = sum_n sum_m sum_k [d_nm(k)p_nm(k)]
    Dipole term currently a crude model to get a vector polarization
    '''
    # Determine number of k-points in b2 direction
    Nk1 = np.size(pvc, axis=0)
    Nk2 = np.size(pvc, axis=1)

    # Create dipole matrix elements (as a crude model)
    d_x, d_y = [],[]
    for path in paths:

        path = np.array(path)

        kx_in_path = path[:,0]
        ky_in_path = path[:,1]

        Ax_in_path, Ay_in_path = dipole.evaluate(kx_in_path, ky_in_path)

        d_x.append(Ax_in_path[1,0,:])
        d_y.append(Ay_in_path[1,0,:])

#        for i_k, k in enumerate(path):
#            d_x.append(np.maximum(np.minimum(np.real(Ax_in_path[1,0,i_k]),10.0),-10.0))
#            d_y.append(np.maximum(np.minimum(np.real(Ay_in_path[1,0,i_k]),10.0),-10.0))

    # Reshape for dot product
#    d_x = np.reshape(d_x, (Nk1,Nk2))
#    d_y = np.reshape(d_y, (Nk1,Nk2))
    d_x_swapped = np.swapaxes(d_x,0,1)
    d_y_swapped = np.swapaxes(d_y,0,1)

    Px = np.tensordot(d_x_swapped,pvc,2) + np.tensordot(d_x_swapped,pcv,2)
    Py = np.tensordot(d_y_swapped,pvc,2) + np.tensordot(d_y_swapped,pcv,2)

    # Return the real part of each component
    return np.real(Px), np.real(Py)


def current(paths,fv,fc,bite,path,t,alpha,bandstruc_deriv_for_print):
    '''
    Calculates the current as: J(t) = sum_k sum_n [j_n(k)f_n(k,t)]
    where j_n(k) != (d/dk) E_n(k)
    '''

    Nk1 = np.size(fc,axis=0)
    Nk2 = np.size(fc,axis=1)

    # Calculate the gradient analytically at each k-point
    Jx, Jy = [], []
    jex,jey,jhx,jhy = [],[],[],[]
    for path in paths:
        path = np.array(path)
        kx_in_path = path[:,0]
        ky_in_path = path[:,1]
        bandstruc_deriv = bite.evaluate_ederivative(kx_in_path, ky_in_path)
        bandstruc_deriv_for_print.append(bandstruc_deriv)
        #0: v, x   1: v,y   2: c, x  3: c, y
        jex.append(bandstruc_deriv[2])
        jey.append(bandstruc_deriv[3])
        jhx.append(bandstruc_deriv[0])
        jhy.append(bandstruc_deriv[1])

    print("before reshape: shape jex =", np.shape(jex), "shape fc =", np.shape(fc))

    jex_swapped = np.swapaxes(jex,0,1)
    jey_swapped = np.swapaxes(jey,0,1)
    jhx_swapped = np.swapaxes(jhx,0,1)
    jhy_swapped = np.swapaxes(jhy,0,1)

    print("shape jex_swapped =", np.shape(jex_swapped), "shape fc =", np.shape(fc))

    print("jex[0,101] =", jex[0][101], "jex_swapped[101,0] =", jex_swapped[101][0])
    print("jex[1,101] =", jex[1][101], "jex_swapped[101,1] =", jex_swapped[101][1])
    print("jex[0,41]  =", jex[0][41], "jex_swapped[41,0]  =",  jex_swapped[41 ][0])
    print("jex[1,41]  =", jex[1][41], "jex_swapped[41,1]  =",  jex_swapped[41 ][1])

    # we need tensordot for contracting the first two indices (2 kpoint directions)
    Jx = np.tensordot(jex_swapped,fc,2) - np.tensordot(jhx_swapped,fv,2)
    Jy = np.tensordot(jey_swapped,fc,2) - np.tensordot(jhy_swapped,fv,2)

    # Return the real part of each component
    return np.real(Jx), np.real(Jy)


def f(t, y, kpath, dk, gamma2, E0, w, alpha, bandstruc_in_path, dipole_in_path, k_cut, scale_dipole):
    return fnumba(t, y, kpath, dk, gamma2, E0, w, alpha, bandstruc_in_path, dipole_in_path, k_cut, scale_dipole)


@njit
def fnumba(t, y, kpath, dk, gamma2, E0, w, alpha, bandstruc_in_path, dipole_in_path, k_cut, scale_dipole):

    # x != y(t+dt)
    x = np.empty(np.shape(y), dtype=np.dtype('complex'))
    
    # Gradient term coefficient
    D = driving_field(E0, w, t, alpha)/(2*dk)

    # Update the solution vector
    Nk_path = kpath.shape[0]
    for k in range(Nk_path):
        kx = kpath[k,0]
        ky = kpath[k,1]

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
#        ecv = eband(2, ef, kx, ky) - eband(1, ef, kx, ky)
        ecv = bandstruc_in_path[k]
        ep_p = ecv + 1j*gamma2
        ep_n = ecv - 1j*gamma2

        # Rabi frequency: w_R = w_R(i,j,k,t) = d_ij(k).E(t)
        # Rabi frequency conjugate
        wr = rabi(1, 2, kx, ky, k, E0, w, t, alpha, dipole_in_path, k_cut, scale_dipole)
        wr_c = np.conjugate(wr)

        # Update each component of the solution vector
        x[i] = 1j*wr*y[i+1] - 1j*wr_c*y[i+2] + D*(y[m] - y[n])
        x[i+1] = 1j*wr_c*y[i] - 1j*ep_n*y[i+1] + 1j*wr_c*y[i+3] + D*(y[m+1] - y[n+1]) - 1j*wr_c
        x[i+2] = -1j*wr*y[i] + 1j*ep_p*y[i+2] - 1j*wr_c*y[i+3] + D*(y[m+2] - y[n+2]) + 1j*wr
        x[i+3] = 1j*wr*y[i+1] - 1j*wr_c*y[i+2] + D*(y[m+3] - y[n+3])

    return x


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
        ecv = eband(2, ef, kgrid[k1]) - eband(1, ef, kgrid[k1])

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
        drift_coef = driving_field(E0, w, t, alpha)/(2.0*dk)

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


def double_scale_plot(ax1, xdata, data1, data2, xlims, xlabel, label1, label2):
    '''
    Plots the two input sets: data1, data2 on the same x-scale, but with a secondary y-scale (twin of ax1).
    '''
    ax2 = ax1.twinx()                                        # Create secondary y-axis with shared x scale
    ax1.set_xlim(xlims)                                      # Set x limits
    ax2.set_xlim(xlims)                                      # Set x limits for secondary axis
    ax1.plot(xdata, data1, color='r', zorder=1)              # Plot data1 on the first y-axis
    ax1.set_xlabel(xlabel)                                   # Set the label for the x-axis
    ax1.set_ylabel(label1)                                   # Set the first y-axis label
    ax2.plot(xdata, data2, color='b', zorder=2, alpha=0.5)   # Plot data2 on the second y-axis
    ax2.set_ylabel(label2)                                   # Set the second y-axis label
    return ax1, ax2                                          # Returns these two axes with the data plotted

def BZ_plot(kpnts,a):
    
    R = 4.0*np.pi/(3*a)
    r = 2.0*np.pi/(np.sqrt(3)*a)
    print ("")

    BZ_fig = pl.figure()
    ax = BZ_fig.add_subplot(111,aspect='equal')
    
    ax.add_patch(patches.RegularPolygon((0,0),6,radius=R,orientation=np.pi/6,fill=False))

    pl.scatter(0,0,s=15,c='black')
    pl.text(0.05,0.05,r'$\Gamma$')
    pl.scatter(R,0,s=15,c='black')
    pl.text(R,0.05,r'$K$')
    pl.scatter(r*np.cos(np.pi/6),-r*np.sin(np.pi/6),s=15,c='black')
    pl.text(r*np.cos(np.pi/6),-r*np.sin(np.pi/6)-0.2,r'$M$')
    pl.scatter(kpnts[:,0],kpnts[:,1], s=15)
    pl.xlim(-4.5/a,4.5/a)
    pl.ylim(-4.5/a,4.5/a)
    
    return

def path_plot(paths):

    for path in paths:
        path = np.array(path)
        pl.plot(path[:,0], path[:,1])

    return

if __name__ == "__main__":
    main()

