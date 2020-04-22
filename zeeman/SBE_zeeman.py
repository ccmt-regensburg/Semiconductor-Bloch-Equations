#!/bin/python
import numpy as np
from numpy.fft import fft, fftfreq, fftshift
from numba import njit
import matplotlib.pyplot as pl
from matplotlib.patches import RegularPolygon
from scipy.integrate import ode

from hfsbe.utility import evaluate_njit_matrix as evmat


def sbe_zeeman_solver(sys, dipole, dipole_mb, params):
    # RETRIEVE PARAMETERS
    ###########################################################################
    # Flag evaluation
    user_out = params.user_out
    save_file = params.save_file
    save_full = params.save_full
    gauge = params.gauge

    # Unit converstion factors
    fs_conv = params.fs_conv
    E_conv = params.E_conv
    THz_conv = params.THz_conv
    # amp_conv = params.amp_conv
    eV_conv = params.eV_conv

    # System parameters
    a = params.a                                # Lattice spacing
    e_fermi = params.e_fermi*eV_conv            # Fermi energy
    temperature = params.temperature*eV_conv    # Temperature

    # Driving field parameters
    E0 = params.E0*E_conv                       # Driving pulse field amplitude
    w = params.w*THz_conv                       # Driving pulse frequency
    chirp = params.chirp*THz_conv               # Pulse chirp frequency
    alpha = params.alpha*fs_conv                # Gaussian pulse width
    phase = params.phase                        # Carrier-envelope phase

    # Time scales
    T1 = params.T1*fs_conv                      # Occupation damping time
    T2 = params.T2*fs_conv                      # Polarization damping time
    gamma1 = 1/T1                               # Occupation damping parameter
    gamma2 = 1/T2                               # Polarization damping param
    t0 = int(params.t0*fs_conv)                 # Initial time condition
    tf = int(params.tf*fs_conv)                 # Final time
    dt = params.dt*fs_conv                      # Integration time step
    dt_out = 1/(2*params.dt)                    # Solution output time step

    # Brillouin zone type
    BZ_type = params.BZ_type                    # Type of Brillouin zone

    # Brillouin zone type
    if BZ_type == 'full':
        Nk1 = params.Nk1                        # kpoints in b1 direction
        Nk2 = params.Nk2                        # kpoints in b2 direction
        Nk = Nk1*Nk2                            # Total number of kpoints
        align = params.align                    # E-field alignment
    elif BZ_type == '2line':
        Nk_in_path = params.Nk_in_path
        Nk = 2*Nk_in_path
        # rel_dist_to_Gamma = params.rel_dist_to_Gamma
        # length_path_in_BZ = params.length_path_in_BZ
        angle_inc_E_field = params.angle_inc_E_field

    b1 = params.b1                              # Reciprocal lattice vectors
    b2 = params.b2

    # USER OUTPUT
    ###########################################################################
    if user_out:
        print("Solving for...")
        print("Brillouin zone: " + BZ_type)
        print("Number of k-points              = " + str(Nk))
        if BZ_type == 'full':
            print("Driving field alignment         = " + align)
        elif BZ_type == '2line':
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
        BZ_plot(kpnts, a, b1, b2, paths)
    elif BZ_type == '2line':
        E_dir = np.array([np.cos(np.radians(angle_inc_E_field)),
                         np.sin(np.radians(angle_inc_E_field))])
        dk, kpnts, paths = mesh(params, E_dir)
        BZ_plot(kpnts, a, b1, b2, paths)

    # Number of integration steps, time array construction flag
    Nt = int((tf-t0)/dt)
    t_constructed = False

    # Solution containers
    t = []
    solution = []

    # Initialize the ode solver and create fnumba
    fnumba = make_fnumba(sys, dipole, dipole_mb, gauge=gauge)
    solver = ode(fnumba, jac=None)\
        .set_integrator('zvode', method='bdf', max_step=dt)

    # Vector field
    A_field = []
    # SOLVING
    ###########################################################################
    # Iterate through each path in the Brillouin zone

    path_num = 1
    for path in paths:
        if user_out:
            print('path: ' + str(path_num))

        # Solution container for the current path
        path_solution = []

        # Retrieve the set of k-points for the current path
        kx_in_path = path[:, 0]
        ky_in_path = path[:, 1]

        # Initialize the values of of each k point vector
        # (rho_nn(k), rho_nm(k), rho_mn(k), rho_mm(k))
        ec = sys.efjit[1](kx=kx_in_path, ky=ky_in_path, mb=zeeman_field(t0))
        y0 = initial_condition(e_fermi, temperature, ec)
        y0 = np.append(y0, [0.0])

        # Set the initual values and function parameters for the current kpath
        solver.set_initial_value(y0, t0)\
            .set_f_params(path, dk, gamma1, gamma2, E0, w, chirp, alpha, phase,
                          E_dir, y0) 

        # Propagate through time
        ti = 0
        while solver.successful() and ti < Nt:
            # User output of integration progress
            if (ti % 1000 == 0 and user_out):
                print('{:5.2f}%'.format(ti/Nt*100))

            # Integrate one integration time step
            solver.integrate(solver.t + dt)

            # Save solution each output step
            if (ti % dt_out == 0):
                # Do not append the last element (A_field)
                path_solution.append(solver.y[:-1])
                # Construct time array only once
                if not t_constructed:
                    # Construct time and A_field only in first round
                    t.append(solver.t)
                    A_field.append(solver.y[-1])

            # Increment time counter
            ti += 1

        # Flag that time array has been built up
        t_constructed = True
        path_num += 1

        # Append path solutions to the total solution arrays
        solution.append(path_solution)
        print(np.shape(path_solution))

    # Convert solution and time array to numpy arrays
    t = np.array(t)
    solution = np.array(solution)
    A_field = np.array(A_field)

    # Slice solution along each path for easier observable calculation
    # Split the last index into 100 subarrays, corresponding to kx
    # Consquently the new last axis becomes 4.
    if BZ_type == 'full':
        solution = np.array_split(solution, Nk1, axis=2)
    elif BZ_type == '2line':
        solution = np.array_split(solution, Nk_in_path, axis=2)

    # Convert lists into numpy arrays
    solution = np.array(solution)
    # The solution array is structred as: first index is Nk1-index,
    # second is Nk2-index, third is timestep, fourth is f_h, p_he, p_eh, f_e

    # COMPUTE OBSERVABLES
    ###########################################################################
    dt_out = t[1] - t[0]
    freq = fftshift(fftfreq(np.size(t), d=dt_out))

    I_exact_E_dir, I_exact_ortho = emission_exact(sys, paths, t, solution,
                                                  E_dir, A_field)
    Iw_exact_E_dir = fftshift(fft(I_exact_E_dir*gaussian_envelope(t, alpha),
                                  norm='ortho'))
    Iw_exact_ortho = fftshift(fft(I_exact_ortho*gaussian_envelope(t, alpha),
                                  norm='ortho'))
    Int_exact_E_dir = (freq**2)*np.abs(Iw_exact_E_dir)**2
    Int_exact_ortho = (freq**2)*np.abs(Iw_exact_ortho)**2

    # Save observables to file
    if (BZ_type == '2line'):
        Nk1 = Nk_in_path
        Nk2 = 2

    if (save_file):
        tail = 'Nk1-{}_Nk2-{}_w{:4.2f}_E{:4.2f}_a{:4.2f}_ph{:3.2f}_T2-{:05.2f}'\
            .format(Nk1, Nk2, w/THz_conv, E0/E_conv, alpha/fs_conv, phase, T2/fs_conv)

        I_exact_name = 'Iexact_' + tail
        np.save(I_exact_name, [t, I_exact_E_dir, I_exact_ortho, freq/w,
                Iw_exact_E_dir, Iw_exact_ortho,
                Int_exact_E_dir, Int_exact_ortho])

        if (save_full):
            S_name = 'Sol_' + tail
            np.savez(S_name, t=t, solution=solution, paths=paths,
                     driving_field=driving_field(E0, w, t, chirp, alpha, phase))

        driving_tail = 'w{:4.2f}_E{:4.2f}_a{:4.2f}_ph{:3.2f}_wc-{:4.3f}'\
            .format(w/THz_conv, E0/E_conv, alpha/fs_conv, phase, chirp/THz_conv)

        D_name = 'E_' + driving_tail
        np.save(D_name, [t, driving_field(E0, w, t, chirp, alpha, phase)])


###############################################################################
# FUNCTIONS
###############################################################################
def mesh(params, E_dir):
    Nk_in_path = params.Nk_in_path                    # Number of kpoints in each of the two paths
    rel_dist_to_Gamma = params.rel_dist_to_Gamma      # relative distance (in units of 2pi/a) of both paths to Gamma
    a = params.a                                      # Lattice spacing
    length_path_in_BZ = params.length_path_in_BZ      #

    alpha_array = np.linspace(-0.5 + (1/(2*Nk_in_path)),
                              0.5 - (1/(2*Nk_in_path)), num=Nk_in_path)
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
        # Checks if the absolute values of x and y components of p are within
        # the first quadrant of the hexagon.
        x = np.abs(p[0])
        y = np.abs(p[1])
        return ((y <= 2.0*np.pi/(np.sqrt(3)*a)) and
                (np.sqrt(3.0)*x + y <= 4*np.pi/(np.sqrt(3)*a)))

    def reflect_point(p, a, b1, b2):
        x = p[0]
        y = p[1]
        if (y > 2*np.pi/(np.sqrt(3)*a)):   # Crosses top
            p -= b2
        elif (y < -2*np.pi/(np.sqrt(3)*a)):
            # Crosses bottom
            p += b2
        elif (np.sqrt(3)*x + y > 4*np.pi/(np.sqrt(3)*a)):
            # Crosses top-right
            p -= b1 + b2
        elif (-np.sqrt(3)*x + y < -4*np.pi/(np.sqrt(3)*a)):
            # Crosses bot-right
            p -= b1
        elif (np.sqrt(3)*x + y < -4*np.pi/(np.sqrt(3)*a)):
            # Crosses bot-left
            p += b1 + b2
        elif (-np.sqrt(3)*x + y > 4*np.pi/(np.sqrt(3)*a)):
            # Crosses top-left
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
                # If current point is in BZ, append it to the mesh and path_M
                if (is_in_hex(kpoint, a)):
                    mesh.append(kpoint)
                    path_M.append(kpoint)
                # If current point is NOT in BZ, reflect it along
                # the appropriate axis to get it in the BZ, then append.
                else:
                    while (not is_in_hex(kpoint, a)):
                        kpoint = reflect_point(kpoint, a, b1, b2)
                    mesh.append(kpoint)
                    path_M.append(kpoint)
            # Append the a1'th path to the paths array
            paths.append(path_M)

    elif align == 'K':
        b_a1 = 8*np.pi/(a*3)*np.array([1, 0])
        b_a2 = 4*np.pi/(a*3)*np.array([1, np.sqrt(3)])
        # Extend over half of the b2 direction and 1.5x the b1 direction
        # (extending into the 2nd BZ to get correct boundary conditions)
        alpha1 = np.linspace(-0.5 + (1/(2*Nk1)), 1.0 - (1/(2*Nk1)), num=Nk1)
        alpha2 = np.linspace(0, 0.5 - (1/(2*Nk2)), num=Nk2)
        for a2 in alpha2:
            path_K = []
            for a1 in alpha1:
                kpoint = a1*b_a1 + a2*b_a2
                if is_in_hex(kpoint, a):
                    mesh.append(kpoint)
                    path_K.append(kpoint)
                else:
                    kpoint -= (2*np.pi/a) * np.array([1, 1/np.sqrt(3)])
                    mesh.append(kpoint)
                    path_K.append(kpoint)
            paths.append(path_K)

    return np.array(mesh), np.array(paths)


@njit
def driving_field(E0, w, t, chirp, alpha, phase):
    '''
    Returns the instantaneous driving pulse field
    '''
    # Non-pulse
    # return E0*np.sin(2.0*np.pi*w*t)
    # Chirped Gaussian pulse
    return E0*np.exp(-t**2.0/(2.0*alpha)**2) \
        * np.sin(2.0*np.pi*w*t*(1 + chirp*t) + phase)


@njit
def zeeman_field(t):
    mb = 0.000373195 
    return mb


def gaussian_envelope(t, alpha):
    '''
    Function to multiply a Function f(t) before Fourier transform
    to ensure no step in time between t_final and t_final + delta
    '''
    return np.exp(-t**2.0/(2.0*alpha)**2)


def emission_exact(sys, paths, tarr, solution, E_dir, A_field):

    E_ort = np.array([E_dir[1], -E_dir[0]])

    # n_time_steps = np.size(solution[0, 0, :, 0])
    n_time_steps = np.size(tarr)

    # I_E_dir is of size (number of time steps)
    I_E_dir = np.zeros(n_time_steps)
    I_ortho = np.zeros(n_time_steps)

    for i_time, t in enumerate(tarr):
        mb = zeeman_field(t)
        for i_path, path in enumerate(paths):
            path = np.array(path)
            kx_in_path = path[:, 0]
            ky_in_path = path[:, 1]

            kx_in_path_shifted = kx_in_path - A_field[i_time]*E_dir[0]
            ky_in_path_shifted = ky_in_path - A_field[i_time]*E_dir[1]

            h_deriv_x = evmat(sys.hderivfjit[0], kx=kx_in_path_shifted,
                              ky=ky_in_path_shifted, mb=mb)
            h_deriv_y = evmat(sys.hderivfjit[1], kx=kx_in_path_shifted,
                              ky=ky_in_path_shifted, mb=mb)

            h_deriv_E_dir = h_deriv_x*E_dir[0] + h_deriv_y*E_dir[1]
            h_deriv_ortho = h_deriv_x*E_ort[0] + h_deriv_y*E_ort[1]

            U = sys.Uf(kx=kx_in_path, ky=ky_in_path, mb=mb)
            U_h = sys.Uf_h(kx=kx_in_path, ky=ky_in_path, mb=mb)

            for i_k in range(np.size(kx_in_path)):

                dH_U_E_dir = np.matmul(h_deriv_E_dir[:, :, i_k], U[:, :, i_k])
                U_h_H_U_E_dir = np.matmul(U_h[:, :, i_k], dH_U_E_dir)

                dH_U_ortho = np.matmul(h_deriv_ortho[:, :, i_k], U[:, :, i_k])
                U_h_H_U_ortho = np.matmul(U_h[:, :, i_k], dH_U_ortho)

                I_E_dir[i_time] += np.real(U_h_H_U_E_dir[0, 0])\
                    * np.real(solution[i_k, i_path, i_time, 0])
                I_E_dir[i_time] += np.real(U_h_H_U_E_dir[1, 1])\
                    * np.real(solution[i_k, i_path, i_time, 3])
                I_E_dir[i_time] += 2*np.real(U_h_H_U_E_dir[0, 1]
                                             * solution[i_k, i_path, i_time, 2])

                I_ortho[i_time] += np.real(U_h_H_U_ortho[0, 0])\
                    * np.real(solution[i_k, i_path, i_time, 0])
                I_ortho[i_time] += np.real(U_h_H_U_ortho[1, 1])\
                    * np.real(solution[i_k, i_path, i_time, 3])
                I_ortho[i_time] += 2*np.real(U_h_H_U_ortho[0, 1]
                                             * solution[i_k, i_path, i_time, 2])

    return I_E_dir, I_ortho


def make_fnumba(sys, dipole, dipole_mb, gauge='velocity'):
    # Wire the energies
    evf = sys.efjit[0]
    ecf = sys.efjit[1]

    # Wire the dipoles
    # kx-parameter
    di_00xf = dipole.Axfjit[0][0]
    di_01xf = dipole.Axfjit[0][1]
    di_11xf = dipole.Axfjit[1][1]

    # ky-parameter
    di_00yf = dipole.Ayfjit[0][0]
    di_01yf = dipole.Ayfjit[0][1]
    di_11yf = dipole.Ayfjit[1][1]

    # mb - Zeeman z parameter
    di_00mbf = dipole_mb.Apfjit[0][0]
    di_01mbf = dipole_mb.Apfjit[0][1]
    di_11mbf = dipole_mb.Apfjit[1][1]

    @njit
    def flength(t, y, kpath, dk, gamma1, gamma2, E0, w, chirp, alpha, phase,
                E_dir, y0):

        # WARNING! THE LENGTH GAUGE ONLY WORKS WITH
        # TIME CONSTANT MAGNETIC FIELDS FOR NOW
        # Preparing system parameters, energies, dipoles
        mb = zeeman_field(t)
        kx = kpath[:, 0]
        ky = kpath[:, 1]
        ev = evf(kx=kx, ky=ky, mb=mb)
        ec = ecf(kx=kx, ky=ky, mb=mb)
        ecv_in_path = ec - ev

        di_00x = di_00xf(kx=kx, ky=ky, mb=mb)
        di_01x = di_01xf(kx=kx, ky=ky, mb=mb)
        di_11x = di_11xf(kx=kx, ky=ky, mb=mb)
        di_00y = di_00yf(kx=kx, ky=ky, mb=mb)
        di_01y = di_01yf(kx=kx, ky=ky, mb=mb)
        di_11y = di_11yf(kx=kx, ky=ky, mb=mb)

        dipole_in_path = E_dir[0]*di_01x + E_dir[1]*di_01y
        A_in_path = E_dir[0]*di_00x + E_dir[1]*di_00y \
            - (E_dir[0]*di_11x + E_dir[1]*di_11y)

        # x != y(t+dt)
        x = np.empty(np.shape(y), dtype=np.dtype('complex'))

        # Gradient term coefficient
        driving_f = driving_field(E0, w, t, chirp, alpha, phase)
        D = driving_f/(2*dk)

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

            # Energy term eband(i,k) the energy of band i at point k
            ecv = ecv_in_path[k]

            # Rabi frequency: w_R = d_12(k).E(t)
            # Rabi frequency conjugate
            wr = dipole_in_path[k]*driving_f
            wr_c = wr.conjugate()

            # Rabi frequency: w_R = (d_11(k) - d_22(k))*E(t)
            # wr_d_diag   = A_in_path[k]*D
            wr_d_diag = A_in_path[k]*driving_f

            # Update each component of the solution vector
            # i = f_v, i+1 = p_vc, i+2 = p_cv, i+3 = f_c
            x[i] = 2*(wr*y[i+1]).imag + D*(y[m] - y[n]) \
                - gamma1*(y[i]-y0[i])

            x[i+1] = (-1j*ecv - gamma2 + 1j*wr_d_diag)*y[i+1] \
                - 1j*wr_c*(y[i]-y[i+3]) + D*(y[m+1] - y[n+1])

            x[i+2] = x[i+1].conjugate()

            x[i+3] = -2*(wr*y[i+1]).imag + D*(y[m+3] - y[n+3]) \
                - gamma1*(y[i+3]-y0[i+3])

        x[-1] = -driving_f
        return x

    @njit
    def fvelocity(t, y, kpath, dk, gamma1, gamma2, E0, w, chirp, alpha, phase,
                  E_dir, y0):

        # Preparing system parameters, energies, dipoles
        mb = zeeman_field(t)
        k_shift = y[-1].real

        kx = kpath[:, 0] + E_dir[0]*k_shift
        ky = kpath[:, 1] + E_dir[1]*k_shift

        ev = evf(kx=kx, ky=ky, mb=mb)
        ec = ecf(kx=kx, ky=ky, mb=mb)
        ecv_in_path = ec - ev

        di_00x = di_00xf(kx=kx, ky=ky, mb=mb)
        di_01x = di_01xf(kx=kx, ky=ky, mb=mb)
        di_11x = di_11xf(kx=kx, ky=ky, mb=mb)
        di_00y = di_00yf(kx=kx, ky=ky, mb=mb)
        di_01y = di_01yf(kx=kx, ky=ky, mb=mb)
        di_11y = di_11yf(kx=kx, ky=ky, mb=mb)

        dipole_in_path = E_dir[0]*di_01x + E_dir[1]*di_01y
        A_in_path = E_dir[0]*di_00x + E_dir[1]*di_00y \
            - (E_dir[0]*di_11x + E_dir[1]*di_11y)

        # x != y(t+dt)
        x = np.empty(np.shape(y), dtype=np.dtype('complex'))

        # Gradient term coefficient
        driving_f = driving_field(E0, w, t, chirp, alpha, phase)

        # Update the solution vector
        Nk_path = kpath.shape[0]
        for k in range(Nk_path):
            i = 4*k
            # Energy term eband(i,k) the energy of band i at point k
            ecv = ecv_in_path[k]

            # Rabi frequency: w_R = d_12(k).E(t)
            # Rabi frequency conjugate
            wr = dipole_in_path[k]*driving_f
            wr_c = wr.conjugate()

            # Rabi frequency: w_R = (d_11(k) - d_22(k))*E(t)
            # wr_d_diag   = A_in_path[k]*D
            wr_d_diag = A_in_path[k]*driving_f

            # Update each component of the solution vector
            # i = f_v, i+1 = p_vc, i+2 = p_cv, i+3 = f_c
            x[i] = 2*(wr*y[i+1]).imag - gamma1*(y[i]-y0[i])

            x[i+1] = (-1j*ecv - gamma2 + 1j*wr_d_diag)*y[i+1] \
                - 1j*wr_c*(y[i]-y[i+3])

            x[i+2] = x[i+1].conjugate()

            x[i+3] = -2*(wr*y[i+1]).imag - gamma1*(y[i+3]-y0[i+3])

        x[-1] = -driving_f
        return x

    freturn = None
    if (gauge == 'velocity'):
        print("Using velocity gauge")
        freturn = fvelocity
    if (gauge == 'length'):
        print("Using length gauge")
        freturn = flength

    def f(t, y, kpath, dk, gamma1, gamma2, E0, w, chirp, alpha, phase,
          E_dir, y0):
        return freturn(t, y, kpath, dk, gamma1, gamma2, E0, w, chirp, alpha,
                       phase, E_dir, y0)

    return f


def initial_condition(e_fermi, temperature, e_c):
    knum = e_c.size
    ones = np.ones(knum)
    zeros = np.zeros(knum)
    if (temperature > 1e-5):
        distrib = 1/(np.exp((e_c-e_fermi)/temperature)+1)
        return np.array([ones, zeros, zeros, distrib]).flatten('F')
    else:
        return np.array([ones, zeros, zeros, zeros]).flatten('F')


def BZ_plot(kpnts, a, b1, b2, paths):

    R = 4.0*np.pi/(3*a)
    r = 2.0*np.pi/(np.sqrt(3)*a)

    BZ_fig = pl.figure(figsize=(10, 10))
    ax = BZ_fig.add_subplot(111, aspect='equal')

    for b in ((0, 0), b1, -b1, b2, -b2, b1+b2, -b1-b2):
        poly = RegularPolygon(b, 6, radius=R, orientation=np.pi/6, fill=False)
        ax.add_patch(poly)

#    ax.arrow(-0.5*E_dir[0], -0.5*E_dir[1], E_dir[0], E_dir[1],
#             width=0.005, alpha=0.5, label='E-field')

    pl.scatter(0, 0, s=15, c='black')
    pl.text(0.01, 0.01, r'$\Gamma$')
    pl.scatter(r*np.cos(-np.pi/6), r*np.sin(-np.pi/6), s=15, c='black')
    pl.text(r*np.cos(-np.pi/6)+0.01, r*np.sin(-np.pi/6)-0.05, r'$M$')
    pl.scatter(R, 0, s=15, c='black')
    pl.text(R, 0.02, r'$K$')
    pl.scatter(kpnts[:, 0], kpnts[:, 1], s=15)
    pl.xlim(-7.0/a, 7.0/a)
    pl.ylim(-7.0/a, 7.0/a)
    pl.xlabel(r'$k_x$ ($1/a_0$)')
    pl.ylabel(r'$k_y$ ($1/a_0$)')

    for path in paths:
        path = np.array(path)
        pl.plot(path[:, 0], path[:, 1])

    pl.show()
