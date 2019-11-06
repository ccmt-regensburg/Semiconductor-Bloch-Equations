import numpy as np
import matplotlib.pyplot as pl
import time, os
from scipy.integrate import ode
import sys

def eband(n, k):
    '''
    Returns the energy of a band n, from the k-point.
    Band structure modeled as (e.g.)...
    E1(k) = (-1eV) + (1eV)exp(-10*k^2)*(4k^2-1)^2(4k^2+1)^2
    '''
    envelope = ((4.0*k**2 - 1.0)**2.0)*((4.0*k**2 + 1.0)**2.0)
    if (n==1):   # Valence band
        return (-1.0/27.211)+(1.0/27.211)*np.exp(-10.0*k**2.0)*envelope
    elif (n==2): # Conduction band
        return (3.0/27.211)-(1.0/27.211)*np.exp(-5.0*k**2.0)*envelope

def dipole(k):
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

def rabi(n,m,k,E0,w,t,alpha):
    '''
    Rabi frequency of the transition. Calculated from dipole element and driving field
    '''
    return dipole(k)*driving_field(E0, w, t, alpha)
    
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

def polarization(pvc,pcv):
    '''
    Calculates the polarization by summing the contribtion from all kpoints.
    '''
    Nk = np.size(pvc, axis=0)
    return np.real(np.sum(pvc + pcv, axis=0))/Nk
 
def current(k,fv,fc):
    '''
    Calculates current according to 
    J(t) = sum_k[sum_n j_n(k)*f_n(k)]
    where n represents the band index and j_n(k) is the band velocity
    calculated as j_n(k) = grad_k eband(n,k)
    '''
    Nk = np.size(fc, axis=0)
    return np.real(np.dot(diff(k, eband(2,k)), fc) + np.dot(diff(k, eband(1,k)), fv))/Nk

def double_scale_plot(ax1, xdata, data1, data2, xlims, xlabel, label1, label2):
    '''
    Plots the two input sets: data1, data2 on the same x-scale, but with a secondary y-scale (twin of ax1).
    '''
    ax1.set_xlim(xlims)                                      # Set x limits
    ax2 = ax1.twinx()                                        # Create secondary y-axis with shared x scale
    ax1.plot(xdata, data1, color='r', zorder=1)              # Plot data1 on the first y-axis
    ax1.set_xlabel(xlabel)                                   # Set the label for the x-axis
    ax1.set_ylabel(label1)                                   # Set the first y-axis label
    ax1.tick_params(axis=ax1, color = 'r')                   # Color the first y-axis
    ax2.plot(xdata, data2, color='b', zorder=2, alpha=0.5)   # Plot data2 on the second y-axis
    ax2.set_ylabel(label2)                                   # Set the second y-axis label
    ax2.tick_params(axis=ax2, color = 'b')                   # Color the second y-axis
    return ax1, ax2                                          # Returns these two axes with the data plotted

def f(t, y, kgrid, Nk, dk, gamma2, E0, w, alpha):
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
        ecv = eband(2, kgrid[k1]) + eband(1, kgrid[k1])

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

    #print(np.array2string(M, max_line_width=np.inf))

    # Calculate the timestep
    svec = np.dot(M, y) + b 
    return svec

def main():

    # PARAMETERS
    ###############################################################################################
    # All physical parameters in atomic units (hbar = charge = mass = 1)
    gamma2 = 0.0242131                          # Gamma2 parameter
    Nk = 25                                     # Number of k-points
    w = 0.000725665                             # Driving frequency
    E0 = 0.015557                               # Driving field amplitude
    alpha = 2000.0                              # Gaussian pulse width
    t0 = -15000                                 # Initial time condition
    tf = 50000                                  # Final time
    dt = 0.1                                    # Integration time step
    ###############################################################################################

    # UNIT CONVERSION FACTORS
    ###############################################################################################
    fs_conv = 41.34137335        #(1fs = 41.341473335 a.u.)
    E_conv = 0.0001944690381     #(1MV/cm = 1.944690381*10^-4 a.u.) 
    THz_conv = 0.000024188843266 #(1THz = 2.4188843266*10^-5 a.u.)
    amp_conv = 150.97488474      #(1A = 150.97488474)

    print("Solving for...")
    print("Pulse Frequency (THz)[a.u.]     = " + "(" + '%.6f'%(w/THz_conv) + ")" + "[" + '%.6f'%(w) + "]")
    print("Pulse Width (fs)[a.u.]          = " + "(" + '%.6f'%(alpha/fs_conv) + ")" + "[" + '%.6f'%(alpha) + "]")
    print("Driving amplitude (MV/cm)[a.u.] = " + "(" + '%.6f'%(E0/E_conv) + ")" + "[" + '%.6f'%(E0) + "]")
    print("Total time (fs)[a.u.]           = " + "(" + '%.6f'%((tf-t0)/fs_conv) + ")" + "[" + '%.5i'%(tf-t0) + "]")

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
    kgrid = np.linspace(-0.5, 0.5, Nk, endpoint=False)
    dk = (kgrid[-1]-kgrid[0])/Nk
    
    # Initial condition for density matrix and time
    # Initially no excited electrons (and thus no holes) all values set to zero. 
    y0= []
    for k in kgrid:
        y0.extend([0.0,0.0,0.0,0.0])

    # Number of time steps, time vector
    Nt = int((tf-t0)/dt)
    t = np.linspace(t0,tf,Nt)

    # Solutions container
    solution = []

    # Set up solver
    solver = ode(f, jac=None).set_integrator('zvode', method='bdf')
    ###############################################################################################

    # SOLVING THE MATRIX SBE
    ###############################################################################################
    # Set solver
    solver.set_initial_value(y0, t0).set_f_params(kgrid, Nk, dk, gamma2, E0, w, alpha)

    # Integrate each time step
    tn = 0
    while solver.successful() and tn < Nt:
        solver.integrate(solver.t + dt)    # Integrate the next time step
        solution.append(solver.y)          # Append last step to the solution array
        tn += 1                            # Keep track of t-steps 

    # Slice solution along each kpoint for easier observable calculation
    solution = np.array(solution)
    solution = np.array_split(solution,Nk,axis=1)
    solution = np.array(solution)
    ###############################################################################################
    
    # Output the solution vector to a file
    ###############################################################################################
    np.savetxt(save_dir + "/solution.dat", np.transpose([t,solution[1,:,0],solution[1,:,3],solution[1,:,1],solution[1,:,2]]), fmt='%.12f')

    # COMPUTE POLARIZATION,CURRENT,EMISSION,AVG.ABSORPTION
    ###############################################################################################
    # First index of solution is kpoint, second is timestep, third is fv, pvc, pcv, fc

    # Average number of electrons per k-point
    N_elec = np.sum(solution[:,:,3],axis=0)/Nk # Electron number as a function of time

    N_elec_gamma = solution[int(Nk/2),:,3]
    N_elec_k = solution[Nk-1,:,3]
    N_elec_mid = solution[int(Nk*(3/4)),:,3]

    # Current decay start time (fraction of final time)
    decay_start = 0.4
    pol = polarization(solution[:,:,1],solution[:,:,2]) # Polarization
    curr = current(kgrid, solution[:,:,0], solution[:,:,3])*np.exp(-0.5*(np.sign(t-decay_start*tf)+1)*(t-decay_start*tf)**2.0/(2.0*5000)**2.0) # Current 

    # Average energy per time
    #print("Avg. energy absorption (per time): " + str(simps(curr * rabi(omega0, Omega, t), t)))

    # Fourier transform (shift frequencies for better plots)
    freq = np.fft.fftshift(np.fft.fftfreq(Nt, d=dt))                                                    # Frequencies
    fieldfourier = np.fft.fftshift(np.fft.fft(driving_field(E0, w, t, alpha), norm='ortho'))            # Driving field
    polfourier = np.fft.fftshift(np.fft.fft(pol, norm='ortho'))                                         # Polarization
    currfourier = np.fft.fftshift(np.fft.fft(curr, norm='ortho'))                                       # Current
    emis = np.abs(freq*polfourier + 1j*currfourier)**2                                                  # Emission spectrum
    ###############################################################################################

    # FILE OUTPUT
    ###############################################################################################
    emis_filename = 'emis_k' + str(Nk) + '_g2-' + str('%.3f'%(gamma2)) + '.dat'
    emis_header = 'omega        emission spectrum'
    np.savetxt(save_dir + '/' + emis_filename, np.transpose([freq,emis]), header=emis_header, fmt='%.16f')

    pol_filename = 'pol_k' + str(Nk) + '_g2-' + str('%.3f'%(gamma2)) + '.dat'
    pol_header = 't            polarization   omega          pol_fourier'
    np.savetxt(save_dir + '/' + pol_filename, np.transpose(np.real([t,pol,freq,polfourier])), header=pol_header, fmt='%.16f')

    curr_filename = 'curr_k' + str(Nk) + '_g2-' + str('%.3f'%(gamma2)) + '.dat'
    curr_header = 't            polarization   omega          curr_fourier'
    np.savetxt(save_dir + '/' + curr_filename, np.transpose(np.real([t,curr,freq,currfourier])), header=curr_header, fmt='%.16f')
    ###############################################################################################

    # PLOTTING OF DATA FOR EACH PARAMETER
    ###############################################################################################
    # Real-time plot limits
    real_t_lims = (-6*alpha/fs_conv, 6*alpha/fs_conv)
    
    # Frequency plot limits
    freq_lims = (0,20)

    # Create figure, establish the set of requried axes
    fig, ((band_ax, N_ax, P_ax, J_ax), (emis_ax, Efour_ax, Pfour_ax, Jfour_ax)) = pl.subplots(nrows=2, ncols=4, figsize=(8,4))
    
    # Plot band structure in the first set of axes
    band_ax.scatter(kgrid, eband(2, kgrid), s=5)
    band_ax.scatter(kgrid, eband(1, kgrid), s=5)
    band_ax.scatter(kgrid, diff(kgrid, eband(2,kgrid)), s=5, label='Conduction vel')
    band_ax.scatter(kgrid, diff(kgrid, eband(1,kgrid)), s=5, label='Valence vel')
    
    # Plot particle number (with driving field)
    N_ax, N_ax_E = double_scale_plot(N_ax, t/fs_conv, N_elec_gamma, driving_field(E0, w, t, alpha)/E_conv, real_t_lims, r'$t\;(fs)$', r'f_{e}', r'E (MV/cm)')
    #N_ax, N_ax_E = double_scale_plot(N_ax, t/fs_conv, N_elec_gamma, N_elec_k, real_t_lims, r'$t\;(fs)$', r'f_{e}', r'E (MV/cm)')
    
    # Plot polarization (with driving field)
    P_ax, P_ax_E = double_scale_plot(P_ax, t/fs_conv, pol, driving_field(E0, w, t, alpha)/E_conv, real_t_lims, r'$t\;(fs)$', r'P (a.u.)', r'E (MV/cm)')
    
    # Plot current (with driving field)
    J_ax, J_ax_E = double_scale_plot(J_ax, t/fs_conv, curr/amp_conv, driving_field(E0, w, t, alpha)/E_conv, real_t_lims, r'$t\;(fs)$', r'J\;(A)', r'E (MV/cm)')
    
    # Plot emmision spectrum on a semi-log scale
    emis_ax.semilogy(freq/w, emis)
    emis_ax.set_xlim(freq_lims)
    
    # Plot fourier transform of driving field
    Efour_ax.plot(freq/w, fieldfourier)
    Efour_ax.set_xlim(freq_lims)

    # Plot fourier transform of polarization
    Pfour_ax.semilogy(freq/w, polfourier)
    Pfour_ax.set_xlim(freq_lims)
    
    # Plot fourier transform of current
    Jfour_ax.semilogy(freq/w, currfourier)
    Jfour_ax.set_xlim(freq_lims)

    # Show the plot after everything
    pl.show()

if __name__ == "__main__":
    main()

