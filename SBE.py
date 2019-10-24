import numpy as np
import matplotlib.pyplot as pl
import time, os
from scipy.integrate import ode

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
   
def driving_field(E0, w, t, pulse_width):
    '''
    Returns the instantaneous driving electric field
    '''
    return E0*np.sin(2.0*np.pi*w*t)
    #return E0*np.exp(-t**2.0/(2.0*pulse_width)**2)*np.sin(2*np.pi*w*t)

def rabi(n,m,k,E0,w,t,pulse_width):
    '''
    Rabi frequency of the transition. Calculated from dipole element and driving field
    '''
    return dipole(k)*driving_field(E0, w, t, pulse_width)

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

#def current(k, hop, Delta, fc, fv):
#    '''
#    Calculates the current by summing the contribution from all kpoints.
#    '''
#    Nk = np.size(fc, axis=0)
#    return np.real(np.dot(diff(k, econd(k, hop, Delta)), fc) + np.dot(diff(k, -econd(k, hop, Delta)), fv))/Nk

def current(k,fc,fv):
    '''
    Calculates current according to 
    J(t) = sum_k[sum_n j_n(k)*f_n(k)]
    where n represents the band index and j_n(k) is the band velocity
    calculated as j_n(k) = grad_k eband(n,k)
    '''
    Nk = np.size(fc, axis=0)
    return np.real(np.dot(diff(k, eband(2,k)), fc) + np.dot(diff(k, eband(1,k)), fv))/Nk

def f(t, y, kgrid, Nk, gamma1, gamma2, E0, w, pulse_width):
    '''
    Function driving the dynamics of the system.
    This is required as input parameter to the ode solver
    '''
    # Constant added vector
    b = []

    # Create propogation matrix for this time step
    for k1 in range(Nk): # Iterate down all the rows
        # Construct each block
        ecv = eband(2, k1) - eband(1, k1)
        wr = rabi(1, 2, k1, E0, w, t, pulse_width)
        wr_c = np.conjugate(wr)
        drift_coef = 0.0#driving_field(E0, w, t, pulse_width)/(2*(1/Nk))
        diag_block = 1j*np.array([[0.0,wr,-wr_c,0.0],
                                  [wr,-(ecv-1j*gamma2),0.0,wr],\
                                  [-wr_c,0.0,(ecv+1j*gamma2),-wr_c],\
                                  [0.0,-wr_c,wr,0.0]])
                      # 1j*np.array([[0.0,-wr_c,wr,0.0],
                      #            [-wr,(ecv+1j*gamma2),0.0,wr],\
                      #            [wr_c,0.0,-(ecv-1j*gamma2),-wr_c],\
                      #            [0.0,wr_c,-wr,0.0]])
        for_deriv = np.array([[drift_coef,0.0,0.0,0.0],\
                              [0.0,drift_coef,0.0,0.0],\
                              [0.0,0.0,drift_coef,0.0],\
                              [0.0,0.0,0.0,drift_coef]])
        back_deriv = np.array([[-drift_coef,0.0,0.0,0.0],\
                               [0.0,-drift_coef,0.0,0.0],\
                               [0.0,0.0,-drift_coef,0.0],\
                               [0.0,0.0,0.0,-drift_coef]])
        zero_block = np.zeros((4,4),dtype='float') 
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
        #b.extend([gamma1, 0.0, 0.0 , gamma1])
        b.extend([0.0,-wr,wr_c,0.0])

    b = 1j*np.array(b)

    # Calculate the timestep
    svec = np.dot(M, y) + b
    return svec

def main():

    # PARAMETERS
    ###############################################################################################
    # All physical parameters in atomic units (hbar = charge = mass = 1)
    gamma2 = 0.242131 #(1/T2, T2=1fs)          # Gamma2 parameter
    gamma1 = 0.0                                # Gamma1 parameter
    Nk = 10                                     # Number of k-points
    w = 0.1                                     # Driving frequency
    E0 = 1.0                                    # Driving field amplitude
    pulse_width = 2017.5                        # Gaussian pulse width
    t0 = 0                                      # Initial time condition
    tf = 100                                    # Final time
    dt = 0.01                                    # Integration time step
    ###############################################################################################

    # FILENAME/DIRECTORY DETAILS
    ###############################################################################################
    time_struct = time.localtime()
    right_now = time.strftime('%y%m%d_%H-%M-%S', time_struct)
    working_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = working_dir + '/' + right_now
    os.mkdir(save_dir)

    print("Solving for: " + "gamma1 = " + str(gamma1) + ", gamma2 = " + str(gamma2))

    # INITIALIZATIONS
    ###############################################################################################
    # Form the kgrid
    kgrid = np.linspace(-0.5, 0.5, Nk, endpoint=False)
    #kgrid = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
    #dk = 1.0/Nk
    
    # Initial condition for density matrix and time
    # Initially all particles in valence band: fv = 1.0, transition matrix element pvc = 1.0
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
    solver.set_initial_value(y0, t0).set_f_params(kgrid, Nk, gamma1, gamma2, E0, w, pulse_width)

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

    # COMPUTE POLARIZATION,CURRENT,EMISSION,AVG.ABSORPTION
    ##############################################################################################
    # First index of solution is kpoint, second is timestep, third is fv, pvc, pcv, fc
    
    #N = np.sum(solution[:,:,0]+solution[:,:,3],axis=0) particle number
    pol = polarization(solution[:,:,1],solution[:,:,2]) # Polarization
    curr = current(kgrid, solution[:,:,0], solution[:,:,3]) # Current

    # Average energy per time
    #print("Avg. energy absorption (per time): " + str(simps(curr * rabi(omega0, Omega, t), t)))

    # Fourier transform (shift frequencies for better plots)
    freq = np.fft.fftshift(np.fft.fftfreq(Nt, d=dt))                                          # Frequencies
    fieldfourier = np.fft.fftshift(np.fft.fft(driving_field(E0, w, t, pulse_width), norm='ortho'))  # Driving field
    polfourier = np.fft.fftshift(np.fft.fft(pol, norm='ortho'))                                         # Polarization
    currfourier = np.fft.fftshift(np.fft.fft(curr, norm='ortho'))                                       # Current
    emis = np.abs(freq*polfourier + 1j*currfourier)**2                                                  # Emission spectrum
    ###############################################################################################

    # FILE OUTPUT
    ###############################################################################################
    emis_filename = 'emis_k' + str(Nk) + '_g1-' + str('%.3f'%(gamma1)) + '_g2-' + str('%.3f'%(gamma2)) + '.dat'
    emis_header = 'omega        emission spectrum'
    np.savetxt(save_dir + '/' + emis_filename, np.transpose([freq,emis]), header=emis_header, fmt='%.12f')

    pol_filename = 'pol_k' + str(Nk) + '_g1-' + str('%.3f'%(gamma1)) + '_g2-' + str('%.3f'%(gamma2)) + '.dat'
    pol_header = 't            polarization   omega          pol_fourier'
    np.savetxt(save_dir + '/' + pol_filename, np.transpose(np.real([t,pol,freq,polfourier])), header=pol_header, fmt='%.12f')

    curr_filename = 'curr_k' + str(Nk) + '_g1-' + str('%.3f'%(gamma1)) + '_g2-' + str('%.3f'%(gamma2)) + '.dat'
    curr_header = 't            polarization   omega          curr_fourier'
    np.savetxt(save_dir + '/' + curr_filename, np.transpose(np.real([t,curr,freq,currfourier])), header=curr_header, fmt='%.12f')
    ###############################################################################################

    # PLOTTING OF DATA FOR EACH PARAMETER
    ###############################################################################################
    # Real-time driving field, polarization, current
    pl.figure(1)
    pl.plot(t/41.3, driving_field(E0, w, t, pulse_width), label = 'Driving field')
    pl.plot(t/41.3, pol, label = 'Polarization')
    pl.plot(t/41.3, curr, label = 'Current')
    ax = pl.gca()
    ax.set_xlabel(r'$t (fs)$', fontsize = 14)
    ax.legend(loc = 'best')

    # Fourier spectra
    pl.figure(2)
    pl.plot(freq/w, np.abs(fieldfourier), label = 'Driving field')
    pl.plot(freq/w, np.abs(polfourier),  label = 'Polarization')
    pl.plot(freq/w, np.abs(currfourier), label = 'Current')
    pl.xlim((-10,10))
    ax = pl.gca()
    ax.set_xlabel(r'$\omega/\omega_0$', fontsize = 14)
    ax.legend(loc = 'best')

    # Emission spectrum
    pl.figure(3)
    pl.yscale('log')
    pl.xlim((0,30))
    pl.grid(True)
    pl.plot(freq/w, np.real(emis), label = 'Emission')
    ax = pl.gca()
    ax.set_xlabel(r'$\omega/\omega_0$', fontsize = 14)
    ax.set_ylabel(r'$Intensity$', fontsize = 14)
    ax.legend(loc = 'best')
    ###############################################################################################

    # OUTPUT BANDSTRUCTURE (SAME FOR EACH PARAMETER)
    ###############################################################################################
    # Band structure/velocities
    pl.figure(4)
    pl.scatter(kgrid, eband(2, kgrid), s = 5, label = 'Conduction band')
    pl.scatter(kgrid, eband(1, kgrid), s = 5, label = 'Valence band')
    #pl.scatter(kgrid, diff(kgrid, eband(1,kgrid)), s = 5, label = 'Conduction band velocity')
    #pl.scatter(kgrid, diff(kgrid, eband(2, kgrid)), s = 5, label = 'Valence band velocity')
    ax = pl.gca()
    ax.set_xlabel(r'$k$a', fontsize = 14)
    ax.set_ylabel(r'$\epsilon(k)$',fontsize = 14)
    ax.legend(loc = 'best')
    ###############################################################################################

    pl.show()
    
if __name__ == "__main__":
    main()

