import numpy as np
import matplotlib.pyplot as pl
from scipy.integrate import ode
import time, os, argparse

'''
TO DO ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
- legacy matrix method not compatible with two-dimensional case.
- current function not compatible with K-paths.
- arbitrary (circular) polarization (big task).
- plots for two-dimensional case
- Emission spectrum for arbitrary direction
'''


def main():

    # USER INPUT FROM COMMAND LINE
    ###############################################################################################
    parser = argparse.ArgumentParser(description='Simulation of the semiconductor-bloch equations')
    parser.add_argument('Nk1',   type=int,   nargs='?', default=4,     help='Number of k-points in b1 direction of the Brillouin zone')
    parser.add_argument('Nk2',   type=int,   nargs='?', default=4,     help='Number of k-points in b2 direction of the Brillouin zone')
    parser.add_argument('E0',    type=float, nargs='?', default=12.0,  help='Maximum pulse field value (in MV/cm)')
    parser.add_argument('w',     type=float, nargs='?', default=30.0,  help='Central pulse frequency (in THz)')
    parser.add_argument('alpha', type=float, nargs='?', default=48.0,  help='Width of pulse Gaussian envelope (in femtoseconds)')
    parser.add_argument('align', type=str,   nargs='?', default='M',   help='Alignment of the pulse polarization. Set to \'M\' for gamma-M direction, \'K\' for gamma-K direction.')
    parser.add_argument('T2',    type=float, nargs='?', default=1.0,   help='Phenomenological damping time (in femtoseconds)')
    parser.add_argument('t0',    type=float, nargs='?', default=-1000, help='Simulation start time. Note: pulse centered about t=0, start with negative values. (in femtoseconds)')
    parser.add_argument('tf',    type=float, nargs='?', default=1000,  help='Simulation final time. Note: Allow for ~200fs for current to decay. (in femtoseconds)')
    parser.add_argument('dt',    type=float, nargs='?', default=0.01,  help='Time step (in femtoseconds)')
    parser.add_argument('-m',    default=False, action='store_true',   help='Use matrix method for solving (legacy)')
    parser.add_argument('-t',    default=False, action='store_true',   help='Flag to output standard testing values: P(t=0), J(t=0), N_gamma(tf), emis(5/w), emis(12.5/w), emis(15/w). Standard parameters: Nk=20, E0=12, w=30, alpha=48, T2=1, t0=-1500, tf=1500, dt=0.01.')
    args = parser.parse_args()


    # SET SIMULATION PARAMETERS
    ###############################################################################################
    # Unit converstion factors
    fs_conv = 41.34137335                           #(1fs    = 41.341473335 a.u.)
    E_conv = 0.0001944690381                        #(1MV/cm = 1.944690381*10^-4 a.u.) 
    THz_conv = 0.000024188843266                    #(1THz   = 2.4188843266*10^-5 a.u.)
    amp_conv = 150.97488474                         #(1A     = 150.97488474)
    eV_conv = 0.03674932176                         #(1eV    = 0.036749322176 a.u.)

    matrix_method = args.m                          # 'Vector' or 'matrix' updates in f(t,y)

    # Set parameters
    Nk1 = args.Nk1                                  # Number of k_x points
    Nk2 = args.Nk2                                  # Number of k_y points
    Nk = Nk1*Nk2                                    # Total number of k points
    E0 = args.E0*E_conv                             # Driving field amplitude
    w = args.w*THz_conv                             # Driving frequency
    alpha = args.alpha*fs_conv                      # Gaussian pulse width
    align = args.align                              # Pulse polarization direction
    T2 = args.T2*fs_conv                            # Damping time
    gamma2 = 1/T2                                   # Gamma parameter
    t0 = int(args.t0*fs_conv)                       # Initial time condition
    tf = int(args.tf*fs_conv)                       # Final time
    dt = args.dt*fs_conv                            # Integration time step


    # USER OUTPUT
    ###############################################################################################
    print("Solving for...")
    if Nk < 20:
        print("***WARNING***: Convergence issues may result from Nk < 20")
    if args.dt > 1.0:
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
    
    
    # FILENAME/DIRECTORY DETAILS
    ###############################################################################################
    time_struct = time.localtime()
    right_now = time.strftime('%y%m%d_%H-%M-%S', time_struct)
    working_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = working_dir + '/' + right_now
    os.mkdir(save_dir)
    

    # INITIALIZATIONS
    ###############################################################################################
    # Form the Brillouin zone in consideration
    a = 1
    kpnts, M_paths, K_paths = hex_mesh(Nk1, Nk2, a)
    dk1 = 1/Nk1
    dk2 = 1/Nk2
    
    # Number of time steps, time vector
    Nt = int((tf-t0)/dt)
    t = np.linspace(t0,tf,Nt)

    # Solution container
    solution = []    

    # Initialize ode solver according to chosen method
    if matrix_method:
        solver = ode(f_matrix, jac=None).set_integrator('zvode', method='bdf', max_step= dt)
    else:
        solver = ode(f, jac=None).set_integrator('zvode', method='bdf', max_step= dt)

    # Determine the Brillouin zone paths to use
    if align == 'M':
        paths = M_paths
    elif align == 'K':
        paths = K_paths
    

    # SOLVING 
    ###############################################################################################
    # Iterate through each path in the Brillouin zone
    ki = 1
    for path in paths:
        # This step is needed for the gamma-K paths, as they are not uniform in length, thus not suitable to be stored as numpy array initially.
        path = np.array(path)

        # Solution container for the current path
        path_solution = []

        # Initialize the values of of each k point vector (rho_nn(k), rho_nm(k), rho_mn(k), rho_mm(k))
        y0 = []
        for k in path:
            y0.extend([0.0,0.0,0.0,0.0])

        # Set the initual values and function parameters for the current kpath
        solver.set_initial_value(y0,t0).set_f_params(path,dk1,gamma2,E0,w,alpha)

        # Propagate through time
        ti = 0
        print('k path: ' + str(ki))
        while solver.successful() and ti < Nt:
            if (ti%10000) == 0:
                print('{:5.2f}%'.format(ti/Nt*100))
            solver.integrate(solver.t + dt)
            path_solution.append(solver.y)
            ti += 1

        solution.append(path_solution)
        ki += 1
        
    # Slice solution along each for easier observable calculation
    solution = np.array(solution)
    solution = np.array_split(solution,Nk1,axis=2)
    solution = np.array(solution)
    # Now the solution array is structred as: first index is kx-index, second is ky-index, third is timestep, fourth is f_h, p_he, p_eh, f_e
    
    # COMPUTE OBSERVABLES
    ###############################################################################################
    
    # Electrons occupations
    
    # Current decay start time (fraction of final time)
    decay_start = 0.4

    Jx, Jy = current(paths, solution[:,:,:,0], solution[:,:,:,3])
    Px, Py = polarization(paths, solution[:,:,:,1], solution[:,:,:,2])
    Ix, Iy = emission(Px, Py, Jx, Jy)

    Ir_x = []
    Ir_y = []
    angles = np.linspace(0,2.0*np.pi,50)
    for angle in angles:
        e = [np.cos(angle),np.sin(angle)]
        Ir_x.append(Ix*e[0])
        Ir_y.append(Iy*e[1])

    Ir_x = np.array(Ir_x)
    Ir_y = np.array(Ir_y)
    
    freq = np.fft.fftshift(np.fft.fftfreq(Nt,d=dt))
    Iw_x = np.fft.fftshift(np.fft.fft(Ix, norm='ortho'))
    Iw_y = np.fft.fftshift(np.fft.fft(Iy, norm='ortho'))
    
    fig1, (ax0,ax1,ax2,ax3) = pl.subplots(1,4)
    t_lims = (-6*alpha/fs_conv, 6*alpha/fs_conv)
    freq_lims = (0,25)
    ax0.set_xlim(t_lims)
    ax0.plot(t/fs_conv,driving_field(E0,w,t,alpha))
    ax1.set_xlim(t_lims)
    ax1.plot(t/fs_conv,Px)
    ax1.plot(t/fs_conv,Py)
    ax2.set_xlim(t_lims)
    ax2.plot(t/fs_conv,Jx)
    ax2.plot(t/fs_conv,Jy)
    ax3.set_xlim(freq_lims)
    ax3.semilogy(freq/w,np.abs(Iw_x))
    ax3.semilogy(freq/w,np.abs(Iw_y))

    f5 = np.argwhere(np.logical_and(freq/w > 4.9, freq/w < 5.1))
    f125 = np.argwhere(np.logical_and(freq/w > 12.4, freq/w < 12.6))
    f15= np.argwhere(np.logical_and(freq/w > 14.9, freq/w < 15.1))
    f_5 = f5[int(np.size(f5)/2)]
    f_125 = f125[int(np.size(f125)/2)]
    f_15 = f15[int(np.size(f15)/2)]

    fig2 = pl.figure()
    pax0 = fig2.add_subplot(131,projection='polar')
    pax0.plot(angles,np.sqrt(Ir_x[:,f_5]**2 + Ir_y[:,f_5]**2))
    pax1 = fig2.add_subplot(132,projection='polar')
    pax1.plot(angles,np.sqrt(Ir_x[:,f_125]**2 + Ir_y[:,f_125]**2))
    pax2 = fig2.add_subplot(133,projection='polar')
    pax2.plot(angles,np.sqrt(Ir_x[:,f_15]**2 + Ir_y[:,f_15]**2))

    
    pl.show()
   
    
    # OUTPUT STANDARD TEST VALUES
    ##############################################################################################
    if args.t:
        test_vals = []
        test_names = []
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
        

    # FILE OUTPUT
    ###############################################################################################
    #part_filename = str('part_Nk{}_w{:4.2f}_E{:4.2f}_a{:4.2f}_dt{:3.2f}.dat').format(Nk,w/THz_conv,E0/E_conv,alpha/fs_conv,dt)
    #part_header = 't           N_elec_gamma   N_elec_mid     N_elec_K       N_elec_negmid  N_elec_negK'
    #np.savetxt(save_dir + '/' + part_filename, np.transpose([t/fs_conv,N_gamma,N_mid,N_K,N_negmid,N_negK]), header=part_header, fmt='%.16e') 
    
    #emis_filename = str('emis_Nk{}_w{:4.2f}_E{:4.2f}_a{:4.2f}_dt{:3.2f}.dat').format(Nk,w/THz_conv,E0/E_conv,alpha/fs_conv,dt)
    #emis_header = 'w/w0        emission spectrum'
    #np.savetxt(save_dir + '/' + emis_filename, np.transpose([freq/w,emis]), header=emis_header, fmt='%.16e')

    #pol_filename = str('pol_Nk{}_w{:4.2f}_E{:4.2f}_a{:4.2f}_dt{:3.2f}.dat').format(Nk,w/THz_conv,E0/E_conv,alpha/fs_conv,dt)
    #pol_header = 't            polarization   w/w0           pol_fourier'
    #np.savetxt(save_dir + '/' + pol_filename, np.transpose(np.real([t/fs_conv,pol,freq/w,polfourier])), header=pol_header, fmt='%.16e')

    #curr_filename = str('curr_Nk{}_w{:4.2f}_E{:4.2f}_a{:4.2f}_dt{:3.2f}.dat').format(Nk,w/THz_conv,E0/E_conv,alpha/fs_conv,dt)
    #curr_header = 't            current       w/w0           curr_fourier'
    #np.savetxt(save_dir + '/' + curr_filename, np.transpose(np.real([t,curr,freq,currfourier])), header=curr_header, fmt='%.16e')


    # PLOTTING OF DATA FOR EACH PARAMETER
    ###############################################################################################
    # Real-time plot limits
    '''
    real_t_lims = (-6*alpha/fs_conv, 6*alpha/fs_conv)
    
    # Frequency plot limits
    freq_lims = (0,25)

    # Create figure, establish the set of requried axes
    fig, ((band_ax, N_ax, P_ax, J_ax), (emis_ax, Efour_ax, Pfour_ax, Jfour_ax)) = pl.subplots(nrows=2, ncols=4, figsize=(18,10))
    
    # Plot band structure in the first set of axes
    band_ax.scatter(kgrid, eband(2, kgrid)/eV_conv, s=5)
    band_ax.scatter(kgrid, eband(1, kgrid)/eV_conv, s=5)
    band_ax.scatter(kgrid, diff(kgrid, eband(2,kgrid)/eV_conv), s=5, label='Conduction vel')
    band_ax.scatter(kgrid, diff(kgrid, eband(1,kgrid)/eV_conv), s=5, label='Valence vel')
    band_ax.set_xlabel(r'$ka$')
    band_ax.set_ylabel(r'$\epsilon(k)$')

    # Plot particle number (with driving field)
    N_ax, N_ax_E = double_scale_plot(N_ax, t/fs_conv, N_gamma, driving_field(E0, w, t, alpha)/E_conv, real_t_lims, r'$t\;(fs)$', r'$f_{e}(k=\Gamma)$', r'$E(t)\;(MV/cm)$')
    
    # Plot polarization (with driving field)
    P_ax, P_ax_E = double_scale_plot(P_ax, t/fs_conv, pol, driving_field(E0, w, t, alpha)/E_conv, real_t_lims, r'$t\;(fs)$', r'$P(t)\;[a.u.]$', r'$E(t)\;(MV/cm)$')
    
    # Plot current (with driving field)
    J_ax, J_ax_E = double_scale_plot(J_ax, t/fs_conv, curr/amp_conv, driving_field(E0, w, t, alpha)/E_conv, real_t_lims, r'$t\;(fs)$', r'$J(t)\;[Amp]$', r'$E(t)\;(MV/cm)$')
    
    # Plot emmision spectrum on a semi-log scale
    emis_ax.semilogy(freq/w, emis, label='Emission spectrum')
    emis_ax.set_xlim(freq_lims)
    emis_ax.set_ylabel(r'$I_{rad}(\omega)$')
    emis_ax.set_xlabel(r'$\omega/\omega_0$')
    
    # Plot fourier transform of driving field
    Efour_ax.semilogy(freq/w, np.abs(fieldfourier))
    Efour_ax.set_xlim(freq_lims)
    Efour_ax.set_ylabel(r'$E(\omega)$')
    Efour_ax.set_xlabel(r'$\omega/\omega_0$')

    # Plot fourier transform of polarization
    Pfour_ax.semilogy(freq/w, np.abs(polfourier), label='Polarization spectrum')
    Pfour_ax.set_xlim(freq_lims)
    Pfour_ax.set_ylabel(r'$P(\omega)$')
    Pfour_ax.set_xlabel(r'$\omega/\omega_0$')
    
    # Plot fourier transform of current
    Jfour_ax.semilogy(freq/w, np.abs(currfourier), label='Current spectrum')
    Jfour_ax.set_xlim(freq_lims)
    Jfour_ax.set_ylabel(r'$J(\omega)$')
    Jfour_ax.set_xlabel(r'$\omega/\omega_0$')

    # Countour plots of occupations and gradients of occupations
    fig4 = pl.figure()
    X, Y = np.meshgrid(t/fs_conv,kgrid)
    pl.contourf(X, Y, N_elec, 100)
    pl.colorbar().set_label(r'$f_e(k)$')
    pl.xlim([-5*alpha/fs_conv,5*alpha/fs_conv])
    pl.xlabel(r'$t\;(fs)$')
    pl.ylabel(r'$k$')
    pl.tight_layout()

    # Show the plot after everything
    pl.show()
    '''
    
def eband(n, kx, ky):
    '''
    Returns the energy of a band n, from the k-point.
    Band structure modeled as (e.g.)...
    E1(k) = (-1eV) + (1eV)exp(-10*k^2)*(4k^2-1)^2(4k^2+1)^2 (for kgrid = [-0.5,0.5])
    '''
    #envelope = ((2.0*kx**2 + 2.0*ky**2 - 1.0)**2.0)*((2.0*kx**2 +  2.0*ky**2 + 1.0)**2.0)
    if (n==1):   # Valence band
        #return np.zeros(np.shape(k)) # Flat structure
        #return (-1.0/27.211)+(1.0/27.211)*np.exp(-10.0*kx**2 - 10.0*ky**2)#*envelope
        return (-1.0/27.211)+(1.0/27.211)*np.exp(-0.4*kx**2 - 0.4*ky**2)#*envelope
    elif (n==2): # Conduction band
        #return (2.0/27.211)*np.ones(np.shape(k)) # Flat structure
        return (3.0/27.211)-(1.0/27.211)*np.exp(-0.2*kx**2 - 0.2*ky**2)#*envelope


def hex_mesh(Nk1, Nk2, a):
    # Calculate the alpha values needed based on the size of the Brillouin zone
    alpha1 = np.linspace(-0.5 + (1/(2*Nk1)), 0.5 - (1/(2*Nk1)), num = Nk1)
    alpha2 = np.linspace(-0.5 + (1/(2*Nk2)), 0.5 - (1/(2*Nk2)), num = Nk2)
    
    # Define the reciprocal lattice vectors
    b1 = 4.0*np.pi/(np.sqrt(3)*a)*np.array([0,1])
    b2 = 2.0*np.pi/(np.sqrt(3)*a)*np.array([np.sqrt(3),-1])

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


def dipole(kx, ky):
    '''
    Returns the dipole matrix element for making the transition from band n to band m at the current k-point
    '''
    return 1.0 #Eventually inputting some data from other calculations


def driving_field(E0, w, t, alpha):
    '''
    Returns the instantaneous driving electric field
    '''
    #return E0*np.sin(2.0*np.pi*w*t)
    return E0*np.exp(-t**2.0/(2.0*alpha)**2)*np.sin(2.0*np.pi*w*t)


def rabi(n,m,kx,ky,E0,w,t,alpha):
    '''
    Rabi frequency of the transition. Calculated from dipole element and driving field
    '''
    return dipole(kx,ky)*driving_field(E0, w, t, alpha)


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

    
def polarization(paths,pvc,pcv):
    '''
    Calculates the polarization by summing the contribtion from all kpoints.
    '''
    # Determine number of k-points in each direction
    Nk1 = np.size(pvc, axis=0)
    Nk2 = np.size(pvc, axis=1)

    d_x, d_y = [],[]
    for path in paths:
        for k in path:
            kx = k[0]
            ky = k[1]
            d_x.append(ky/np.sqrt(kx**2.0 + ky**2.0))
            d_y.append(-kx/np.sqrt(kx**2.0 + ky**2.0))

    d_x = np.array_split(d_x,Nk2)
    d_y = np.array_split(d_y,Nk2)

    px = np.dot(d_x,pvc) + np.dot(d_x,pcv)
    py = np.dot(d_y,pvc) + np.dot(d_y,pcv)
    
    Px = np.sum(np.sum(px,axis=0),axis=0)
    Py = np.sum(np.sum(py,axis=0),axis=0)

    # Sum over k points, take real-part
    return np.real(Px), np.real(Py)


def current(paths,fv,fc):

    Nk2 = np.size(fc,axis=1)
    
    Jx, Jy = [], []
    jex,jey,jhx,jhy = [],[],[],[]
    for path in paths:
        for k in path:
            kx = k[0]
            ky = k[1]
            # Band gradient at this k-point (for simplified band structure model)
            jex.append(-(0.8/27.211)*kx*np.exp(-0.4*(kx**2+ky**2)))
            jey.append(-(0.8/27.211)*ky*np.exp(-0.4*(kx**2+ky**2)))
            jhx.append(-(0.4/27.211)*kx*np.exp(-0.2*(kx**2+ky**2)))
            jhy.append(-(0.4/27.211)*ky*np.exp(-0.2*(kx**2+ky**2)))
            
    jex = np.array_split(jex,Nk2)
    jhx = np.array_split(jhx,Nk2)
    jey = np.array_split(jey,Nk2)
    jhy = np.array_split(jhy,Nk2)

    jx = np.dot(jex,fc) + np.dot(jhx,fv)
    jy = np.dot(jey,fc) + np.dot(jhy,fv)

    Jx = np.sum(np.sum(jx,axis=0), axis=0)
    Jy = np.sum(np.sum(jy,axis=0), axis=0)
    
    return np.real(Jx), np.real(Jy)

def emission(Px, Py, Jx, Jy):
    '''
    Input: The vector components of the time-dependent polarization and current
    Output: Vector components of the time-dependent emission
    '''      
    Ix = np.abs(1j*Jx + Px)**2.0
    Iy = np.abs(1j*Jy + Py)**2.0
      
    return np.array(Ix), np.array(Iy)


def f(t, y, kpath, dk, gamma2, E0, w, alpha):

    x = np.empty(np.shape(y),dtype='complex')
    
    '''
    Gradient term coefficient
    '''
    D = driving_field(E0, w, t, alpha)/(2*dk)

    '''
    Update the solution vector
    '''
    Nk_path = np.size(kpath, axis=0)
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

        '''
        Energy term eband(i,k) the energy of band i at point k
        '''
        ecv = eband(2, kx, ky) - eband(1, kx, ky)
        ep_p = ecv + 1j*gamma2
        ep_n = ecv - 1j*gamma2

        '''
        Rabi frequency: w_R = w_R(i,j,k,t) = d_ij(k).E(t)
        Rabi frequency conjugate
        '''
        wr = rabi(1, 2, kx, ky, E0, w, t, alpha)
        wr_c = np.conjugate(wr)

        '''
        Update each component of the solution vector
        '''
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

if __name__ == "__main__":
    main()

