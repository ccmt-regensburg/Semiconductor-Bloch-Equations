import params
import numpy as np
from numba import njit
import matplotlib.pyplot as pl
from matplotlib import patches
from scipy.integrate import ode
from scipy.special import erf

import hfsbe.dipole
import hfsbe.example
import hfsbe.utility

'''
TO DO ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
- arbitrary (circular) polarization (big task)
- make hex_mesh general for arbitrary direction
- change testing outputs for 1d case 
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
    w     = params.w*THz_conv                         # Driving pulse frequency
    wt    = params.wt*THz_conv                        # Pulse chirp frequency
    alpha = params.alpha*fs_conv                      # Gaussian pulse width
    phase = params.phase                              # Carrier-envelope phase
    
    # Time scales
    T2 = params.T2*fs_conv                            # Damping time
    gamma2 = 1/T2                                     # Gamma parameter
    t0 = int(params.t0*fs_conv)                       # Initial time condition
    tf = int(params.tf*fs_conv)                       # Final time
    dt = params.dt*fs_conv                            # Integration time step
    dt_out = 1/(2*params.dt)                          # Solution output time step

    # Brillouin zone type
    BZ_type = params.BZ_type                          # Type of Brillouin zone to construct
    
    # Brillouin zone type
    if BZ_type == 'full':
        Nk1   = params.Nk1                              # Number of kpoints in b1 direction
        Nk2   = params.Nk2                              # Number of kpoints in b2 direction
        Nk    = Nk1*Nk2                                 # Total number of kpoints
        align = params.align                            # E-field alignment
    elif BZ_type == '2line':
        Nk_in_path = params.Nk_in_path                    # Number of kpoints in each of the two paths
        Nk = 2*Nk_in_path                                 # Total number of k points, we have 2 paths
        rel_dist_to_Gamma = params.rel_dist_to_Gamma      # relative distance (in units of 2pi/a) of both paths to Gamma
        length_path_in_BZ = params.length_path_in_BZ      # Length of a single path in the BZ
        angle_inc_E_field = params.angle_inc_E_field      # Angle of driving electric field

    b1    = params.b1                               # Reciprocal lattice vectors
    b2    = params.b2

    user_out = params.user_out
    test = params.test                                # Testing flag for Travis

    # USER OUTPUT
    ###############################################################################################
    if user_out:
        print("Solving for...")
        print("Brillouin zone: " + BZ_type)
        if Nk < 20:
            print("***WARNING***: Convergence issues may result from Nk < 20")
        if params.dt > 1.0:
            print("***WARNING***: Time-step may be insufficiently small. Use dt < 1.0fs")
        print("Number of k-points              = " + str(Nk))
        if BZ_type == 'full':
            print("Driving field alignment         = " + align)
        elif BZ_type == '2line':
            print("Driving field direction         = " + str(angle_inc_E_field))
        print("Driving amplitude (MV/cm)[a.u.] = " + "(" + '%.6f'%(E0/E_conv) + ")" + "[" + '%.6f'%(E0) + "]")
        print("Pulse Frequency (THz)[a.u.]     = " + "(" + '%.6f'%(w/THz_conv) + ")" + "[" + '%.6f'%(w) + "]")
        print("Pulse Width (fs)[a.u.]          = " + "(" + '%.6f'%(alpha/fs_conv) + ")" + "[" + '%.6f'%(alpha) + "]")
        print("Chirp freq. (THz)[a.u.]         = " + "(" + '%.6f'%(wt/THz_conv) + ")" + "[" + '%.6f'%(wt) + "]")
        print("Damping time (fs)[a.u.]         = " + "(" + '%.6f'%(T2/fs_conv) + ")" + "[" + '%.6f'%(T2) + "]")
        print("Total time (fs)[a.u.]           = " + "(" + '%.6f'%((tf-t0)/fs_conv) + ")" + "[" + '%.5i'%(tf-t0) + "]")
        print("Time step (fs)[a.u.]            = " + "(" + '%.6f'%(dt/fs_conv) + ")" + "[" + '%.6f'%(dt) + "]")
    
    # INITIALIZATIONS
    ###############################################################################################
    # Form the E-field direction
   
    # Form the Brillouin zone in consideration
    if BZ_type == 'full':
        kpnts, paths = hex_mesh(Nk1, Nk2, a, b1, b2, align)
        dk = 1/Nk1
        if align == 'K':
            E_dir = np.array([1,0])
        elif align == 'M':
            E_dir = np.array([np.cos(-30/360*2*np.pi),np.sin(-30/360*2*np.pi)])
    elif BZ_type == '2line':
        E_dir = np.array([np.cos(angle_inc_E_field/360*2*np.pi),np.sin(angle_inc_E_field/360*2*np.pi)])
        dk, kpnts, paths = mesh(params, E_dir)

    # Number of integration steps, time array construction flag
    Nt = int((tf-t0)/dt)
    t_constructed = False
    
    # Solution containers
    t                           = []
    solution                    = []
    dipole_E_dir                = []
    berry_conn_E_dir            = []
    dipole_x                    = []
    dipole_y                    = []
    val_band                    = []
    cond_band                   = []

    # Initialize the ode solver
    solver = ode(f, jac=None).set_integrator('zvode', method='bdf', max_step= dt)

    # Get initialize sympy bandstructure, energies/derivatives, dipoles
    # Topological cone call
    bite = hfsbe.example.BiTe(C0=C0,C2=C2,A=A,R=R,kcut=k_cut)
    # Trivial cone call
    #bite = hfsbe.example.BiTeTrivial(C0=C0,C2=C2,R=R,vf=A,kcut=k_cut)
    h, ef, wf, ediff = bite.eigensystem(gidx=1)
    dipole = hfsbe.dipole.SymbolicDipole(h, ef, wf)

    #bite.evaluate_energy(kx, ky)
    #bite.plot_bands_3d(kx, ky)
    #bite.plot_bands_contour(kx, ky)

    # SOLVING 
    ###############################################################################################
    # Iterate through each path in the Brillouin zone
    path_num = 1
    for path in paths:
        if user_out: print('path: ' + str(path_num))

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
        solver.set_initial_value(y0,t0).set_f_params(path,dk,gamma2,E0,w,wt,alpha,phase,ecv_in_path,dipole_in_path,A_in_path)

        # Propagate through time
        ti = 0
        while solver.successful() and ti < Nt:
            # User output of integration progress
            if (ti%10000 == 0 and user_out):
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
        path_num += 1

        # Append path solutions to the total solution arrays
        solution.append(path_solution)

        '''
        # Append path values of dipole etc... to arrays for plotting
        dipole_E_dir.append(dipole_in_path)
        berry_conn_E_dir.append(A_in_path)
        dipole_x.append(di_x[0,1,:])
        dipole_y.append(di_y[0,1,:])
        val_band.append(bandstruct[0])
        cond_band.append(bandstruct[1])
        '''

    # Convert solution and time array to numpy arrays
    t        = np.array(t)
    solution = np.array(solution)

    # Slice solution along each path for easier observable calculation
    if BZ_type == 'full':
        solution = np.array_split(solution,Nk1,axis=2)
    elif BZ_type == '2line':
        solution = np.array_split(solution,Nk_in_path,axis=2)

    # Convert lists into numpy arrays
    solution         = np.array(solution)
    '''
    dipole_E_dir     = np.array(dipole_E_dir)
    berry_conn_E_dir = np.array(berry_conn_E_dir)
    dipole_x         = np.array(dipole_x)
    dipole_y         = np.array(dipole_y)
    val_band         = np.array(val_band)
    cond_band        = np.array(cond_band)
    '''
    # Now the solution array is structred as: first index is kx-index, second is ky-index, third is timestep, fourth is f_h, p_he, p_eh, f_e
    
    # COMPUTE OBSERVABLES
    ###############################################################################################
    # Calculate parallel and orthogonal components of observables
    # Polarization (interband) 
    P_E_dir, P_ortho = polarization(paths, solution[:,:,:,1], dipole, E_dir)
    # Current (intraband)
    J_E_dir, J_ortho = current(paths, solution[:,:,:,0], solution[:,:,:,3], bite, path, t, alpha, E_dir)
    # Emission in time
    I_E_dir, I_ortho = diff(t,P_E_dir)*Gaussian_envelope(t,alpha) + J_E_dir*Gaussian_envelope(t,alpha), \
                       diff(t,P_ortho)*Gaussian_envelope(t,alpha) + J_ortho*Gaussian_envelope(t,alpha)
    # Berry curvature current
    #J_Bcurv_E_dir, J_Bcurv_ortho = current_Bcurv(paths, solution[:,:,:,0], solution[:,:,:,3], bite, path, t, alpha, E_dir, E0, w, phase, dipole)

    # Polar emission in time
    Ir = []
    angles = np.linspace(0,2.0*np.pi,360)
    for angle in angles:
        Ir.append((I_E_dir*np.cos(angle) + I_ortho*np.sin(-angle)))

    # Fourier transforms
    dt_out   = t[1]-t[0]
    freq     = np.fft.fftshift(np.fft.fftfreq(np.size(t),d=dt_out))
    Iw_E_dir = np.fft.fftshift(np.fft.fft(I_E_dir, norm='ortho'))
    Iw_ortho = np.fft.fftshift(np.fft.fft(I_ortho, norm='ortho'))
    Iw_r     = np.fft.fftshift(np.fft.fft(Ir, norm='ortho'))
    Pw_E_dir = np.fft.fftshift(np.fft.fft(diff(t,P_E_dir), norm='ortho'))
    Pw_ortho = np.fft.fftshift(np.fft.fft(diff(t,P_ortho), norm='ortho'))
    #Jw_E_dir = np.absolute(np.fft.fftshift(np.fft.fft(J_E_dir*Gaussian_envelope(t,alpha), norm='ortho')))
    #Jw_ortho = np.absolute(np.fft.fftshift(np.fft.fft(J_ortho*Gaussian_envelope(t,alpha), norm='ortho')))
    Jw_E_dir = np.fft.fftshift(np.fft.fft(J_E_dir*Gaussian_envelope(t,alpha), norm='ortho'))
    Jw_ortho = np.fft.fftshift(np.fft.fft(J_ortho*Gaussian_envelope(t,alpha), norm='ortho'))
    fw_0     = np.fft.fftshift(np.fft.fft(solution[:,0,:,0], norm='ortho'),axes=(1,))

    # Emission intensity
    Int_E_dir = np.abs(freq*Pw_E_dir + 1j*Jw_E_dir)**2.0
    Int_ortho = np.abs(freq*Pw_ortho + 1j*Jw_ortho)**2.0

    # Save observables to file
    if (BZ_type == '2line'):
        Nk1 = Nk_in_path
        Nk2 = 2
    J_filename = str('J_Nk1-{}_Nk2-{}_w{:4.2f}_E{:4.2f}_a{:4.2f}_ph{:3.2f}.dat').format(Nk1,Nk2,w/THz_conv,E0/E_conv,alpha/fs_conv,phase)
    np.save(J_filename, [t/fs_conv, J_E_dir, J_ortho, Jw_E_dir, Jw_ortho])
    P_filename = str('P_Nk1-{}_Nk2-{}_w{:4.2f}_E{:4.2f}_a{:4.2f}_ph{:3.2f}.dat').format(Nk1,Nk2,w/THz_conv,E0/E_conv,alpha/fs_conv,phase)
    np.save(P_filename, [t/fs_conv,P_E_dir,P_ortho, Pw_E_dir, Pw_ortho])
    I_filename = str('I_Nk1-{}_Nk2-{}_w{:4.2f}_E{:4.2f}_a{:4.2f}_ph{:3.2f}.dat').format(Nk1,Nk2,w/THz_conv,E0/E_conv,alpha/fs_conv,phase)
    np.save(I_filename, [freq/w, I_E_dir, I_ortho, np.abs(Iw_E_dir), np.abs(Iw_ortho), Int_E_dir, Int_ortho])

    if (not test and user_out):
        real_fig, ((axE,axP),(axPdot,axJ)) = pl.subplots(2,2)
        t_lims = (-10*alpha/fs_conv, 10*alpha/fs_conv)
        freq_lims = (0,30)
        log_limits = (10e-20,100)
        axE.set_xlim(t_lims)
        axE.plot(t/fs_conv,driving_field(E0,w,wt,t,alpha,phase)/E_conv)
        axE.set_xlabel(r'$t$ in fs')
        axE.set_ylabel(r'$E$-field in MV/cm')
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

        four_fig, ((axPw,axJw),(axIw,axInt)) = pl.subplots(2,2)
        axPw.grid(True,axis='x')
        axPw.set_xlim(freq_lims)
        axPw.set_ylim(log_limits)
        axPw.semilogy(freq/w,np.abs(Pw_E_dir))
        axPw.semilogy(freq/w,np.abs(Pw_ortho))
        axPw.set_xlabel(r'Frequency $\omega/\omega_0$')
        axPw.set_ylabel(r'$[\dot P](\omega)$ (interband) in a.u. $\parallel \mathbf{E}_{in}$ (blue), $\bot \mathbf{E}_{in}$ (orange)')
        axJw.grid(True,axis='x')
        axJw.set_xlim(freq_lims)
        axJw.set_ylim(log_limits)
        axJw.semilogy(freq/w,np.abs(Jw_E_dir))
        axJw.semilogy(freq/w,np.abs(Jw_ortho))
        axJw.set_xlabel(r'Frequency $\omega/\omega_0$')
        axJw.set_ylabel(r'$[\dot P](\omega)$ (intraband) in a.u. $\parallel \mathbf{E}_{in}$ (blue), $\bot \mathbf{E}_{in}$ (orange)')
        axIw.grid(True,axis='x')
        axIw.set_xlim(freq_lims)
        axIw.set_ylim(log_limits)
        axIw.semilogy(freq/w,np.abs(Iw_E_dir))
        axIw.semilogy(freq/w,np.abs(Iw_ortho))
        axIw.set_xlabel(r'Frequency $\omega/\omega_0$')
        axIw.set_ylabel(r'$[\dot P](\omega)$ (total = emitted E-field) in a.u.')
        axInt.grid(True,axis='x')
        axInt.set_xlim(freq_lims)
        axInt.set_ylim(log_limits)
        axInt.semilogy(freq/w,np.abs(Int_E_dir))
        axInt.semilogy(freq/w,np.abs(Int_ortho))
        axInt.set_xlabel(r'Frequency $\omega/\omega_0$')
        axInt.set_ylabel(r'$[I](\omega)$ intensity in a.u.')

        # High-harmonic emission polar plots
        polar_fig = pl.figure()
        i_loop = 1
        i_max  = 20
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

        '''
        # Plot bandstructure
        bandstruct = bite.evaluate_energy(kpnts[:,0],kpnts[:,1])
        band_fig = pl.figure()
        axBS = band_fig.gca(projection='3d')
        axBS.plot_trisurf(kpnts[:,0],kpnts[:,1].T,bandstruct[0])
        axBS.plot_trisurf(kpnts[:,0],kpnts[:,1].T,bandstruct[1])
        '''

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

    vec_k_ortho = 2.0*np.pi/a*rel_dist_to_Gamma*np.array([E_dir[1],-E_dir[0]])

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

def hex_mesh(Nk1, Nk2, a, b1, b2, align):
    alpha1 = np.linspace(-0.5 + (1/(2*Nk1)), 0.5 - (1/(2*Nk1)), num = Nk1)
    alpha2 = np.linspace(-0.5 + (1/(2*Nk2)), 0.5 - (1/(2*Nk2)), num = Nk2)    

    def is_in_hex(p,a):
        # Returns true if the point is in the hexagonal BZ.
        # Checks if the absolute values of x and y components of p are within the first quadrant of the hexagon.
        x = np.abs(p[0])
        y = np.abs(p[1])
        return ((y <= 2.0*np.pi/(np.sqrt(3)*a)) and (np.sqrt(3.0)*x + y <= 4*np.pi/(np.sqrt(3)*a)))

    def reflect_point(p,a,b1,b2):
        x = p[0]
        y = p[1]
        if (y > 2*np.pi/(np.sqrt(3)*a)):   # Crosses top
            p -= b2
        elif (y < -2*np.pi/(np.sqrt(3)*a)): # Crosses right
            p += b2
        elif (np.sqrt(3)*x + y > 4*np.pi/(np.sqrt(3)*a)): #Crosses top-right
            p -= b1 + b2
        elif (-np.sqrt(3)*x + y < -4*np.pi/(np.sqrt(3)*a)): #Crosses bot-right
            p -= b1
        elif (np.sqrt(3)*x + y < -4*np.pi/(np.sqrt(3)*a)): #Crosses bot-left
            p += b1 + b2
        elif (-np.sqrt(3)*x + y > 4*np.pi/(np.sqrt(3)*a)): #Crosses top-left
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
    
    return np.array(mesh), paths

@njit
def driving_field(E0, w, wt, t, alpha, phase):
    '''
    Returns the instantaneous driving pulse field
    '''
    # Non-pulse
    #return E0*np.sin(2.0*np.pi*w*t)
    # Chirped Gaussian pulse
    return E0*np.exp(-t**2.0/(2.0*alpha)**2)*np.sin(2.0*np.pi*w*t + wt*t**2 + phase)

@njit
def rabi(k,E0,w,wt,t,alpha,phase,dipole_in_path):
    '''
    Rabi frequency of the transition. Calculated from dipole element and driving field
    '''
    return dipole_in_path[k]*driving_field(E0, w, wt, t, alpha, phase)

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

def polarization(paths,pcv,dipole,E_dir):
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

        # Evaluate the dipole moments in path
        di_x, di_y = dipole.evaluate(kx_in_path, ky_in_path)

        # Append the dot product d.E 
        d_E_dir.append(di_x[0,1,:]*E_dir[0] + di_y[0,1,:]*E_dir[1])
        d_ortho.append(di_x[0,1,:]*E_ort[0] + di_y[0,1,:]*E_ort[1])

    d_E_dir_swapped = np.swapaxes(d_E_dir,0,1)
    d_ortho_swapped = np.swapaxes(d_ortho,0,1)
    
    P_E_dir = 2*np.real(np.tensordot(d_E_dir_swapped,pcv,2))
    P_ortho = 2*np.real(np.tensordot(d_ortho_swapped,pcv,2))
    
    return P_E_dir, P_ortho


def current(paths,fv,fc,bite,path,t,alpha,E_dir):
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

def current_Bcurv(paths,fv,fc,bite,path,t,alpha,E_dir,E0,w,phase,dipole):

    # t contains all time points
    A_field   = get_A_field(E0, w, t, alpha)
    A_field_x = A_field*E_dir[0]
    A_field_y = A_field*E_dir[1]
    E_field   = driving_field(E0, w, t, alpha, phase)

    E_ort = np.array([E_dir[1], -E_dir[0]])

    print("shape f =", np.shape(fc))

    # Calculate the gradient analytically at each k-point
    J_E_dir, J_ortho = [], []

    curv = hfsbe.dipole.SymbolicCurvature(dipole.Ax,dipole.Ay)           

#HACK
#    for path in paths:
#       path = np.array(path)
#       kx_in_path = path[:,0]
#       ky_in_path = path[:,1]
#       bandstruc_deriv = bite.evaluate_ederivative(kx_in_path, ky_in_path)
#       curv_eval = curv.evaluate(kx_in_path, ky_in_path)
#HACK

    for j_time, time in enumerate(t):
       je_E_dir,je_ortho,jh_E_dir,jh_ortho = [],[],[],[]

       print("j_time =", j_time, "/", np.shape(fc[0,0,:]))

       for path in paths:
           path = np.array(path)
           kx_in_path = path[:,0]
           ky_in_path = path[:,1]

           kx_in_path = np.real(np.array([x - A_field_x[j_time] for x in kx_in_path]))
           ky_in_path = np.real(np.array([y - A_field_y[j_time] for y in ky_in_path]))

           bandstruc_deriv = bite.evaluate_ederivative(kx_in_path, ky_in_path)

           curv_eval = curv.evaluate(kx_in_path, ky_in_path)

#           print("shape curv_eval =", np.shape(curv_eval))

           # the cross product of Berry curvature and E-field points only in direction orthogonal to E
           cross_prod_ortho = E_field[j_time]*curv_eval

#           print("shape bandstruc_deriv =", np.shape(bandstruc_deriv))
#           print("shaoe cross_prod      =", np.shape(cross_prod_ortho))

           #0: v, x   1: v,y   2: c, x  3: c, y
           je_E_dir.append(bandstruc_deriv[2]*E_dir[0] + bandstruc_deriv[3]*E_dir[1])
           je_ortho.append(bandstruc_deriv[2]*E_ort[0] + bandstruc_deriv[3]*E_ort[1] + cross_prod_ortho[1,1,:])
           jh_E_dir.append(bandstruc_deriv[0]*E_dir[0] + bandstruc_deriv[1]*E_dir[1])
           jh_ortho.append(bandstruc_deriv[0]*E_ort[0] + bandstruc_deriv[1]*E_ort[1] + cross_prod_ortho[0,0,:])
   
       je_E_dir_swapped = np.swapaxes(je_E_dir,0,1)
       je_ortho_swapped = np.swapaxes(je_ortho,0,1)
       jh_E_dir_swapped = np.swapaxes(jh_E_dir,0,1)
       jh_ortho_swapped = np.swapaxes(jh_ortho,0,1)
 
#       print("shape f =", np.shape(fc))
#       print("shape j =", np.shape(je_E_dir_swapped))

       # we need tensordot for contracting the first two indices (2 kpoint directions)
       J_E_dir.append(np.tensordot(je_E_dir_swapped,fc[:,:,j_time],2) + np.tensordot(jh_E_dir_swapped,fv[:,:,j_time],2))
       J_ortho.append(np.tensordot(je_ortho_swapped,fc[:,:,j_time],2) + np.tensordot(jh_ortho_swapped,fv[:,:,j_time],2))

#    # we need tensordot for contracting the first two indices (2 kpoint directions)
#    J_E_dir = np.tensordot(je_E_dir_swapped,fc,2) + np.tensordot(jh_E_dir_swapped,fv,2)
#    J_ortho = np.tensordot(je_ortho_swapped,fc,2) + np.tensordot(jh_ortho_swapped,fv,2)

    # Return the real part of each component
    return np.real(J_E_dir), np.real(J_ortho)

def get_A_field(E0, w, t, alpha):
    '''
    Returns the analytical A-field as integration of the E-field
    '''
    w_eff = 4*np.pi*alpha*w
    return np.real(-alpha*E0*np.sqrt(np.pi)/2*np.exp(-w_eff**2/4)*(2+erf(t/2/alpha-1j*w_eff/2)-erf(-t/2/alpha-1j*w_eff/2)))


def f(t, y, kpath, dk, gamma2, E0, w, wt, alpha, phase, ecv_in_path, dipole_in_path, A_in_path):
    return fnumba(t, y, kpath, dk, gamma2, E0, w, wt, alpha, phase, ecv_in_path, dipole_in_path, A_in_path)


@njit
def fnumba(t, y, kpath, dk, gamma2, E0, w, wt, alpha, phase, ecv_in_path, dipole_in_path, A_in_path):

    # x != y(t+dt)
    x = np.empty(np.shape(y), dtype=np.dtype('complex'))
    
    # Gradient term coefficient
    D = driving_field(E0, w, wt, t, alpha, phase)/(2*dk)

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
        #wr          = dipole_in_path[k]*D
        wr          = rabi(k, E0, w, wt, t, alpha, phase, dipole_in_path)
        wr_c        = np.conjugate(wr)

        # Rabi frequency: w_R = (d_11(k) - d_22(k))*E(t)
        #wr_d_diag   = A_in_path[k]*D
        wr_d_diag   = rabi(k, E0, w, wt, t, alpha, phase, A_in_path)

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

def BZ_plot(kpnts,a,b1,b2,E_dir,paths):
    
    R = 4.0*np.pi/(3*a)
    r = 2.0*np.pi/(np.sqrt(3)*a)

    BZ_fig = pl.figure()
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

