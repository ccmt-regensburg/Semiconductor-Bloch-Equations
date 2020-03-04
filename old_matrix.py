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

    #for path in paths:
    #   path = np.array(path)
    #   kx_in_path = path[:,0]
    #   ky_in_path = path[:,1]
    #   bandstruc_deriv = bite.evaluate_ederivative(kx_in_path, ky_in_path)
    #   curv_eval = curv.evaluate(kx_in_path, ky_in_path)


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

           #print("shape curv_eval =", np.shape(curv_eval))

           # the cross product of Berry curvature and E-field points only in direction orthogonal to E
           cross_prod_ortho = E_field[j_time]*curv_eval

           print("shape bandstruc_deriv =", np.shape(bandstruc_deriv))
           print("shaoe cross_prod      =", np.shape(cross_prod_ortho))

           #0: v, x   1: v,y   2: c, x  3: c, y
           je_E_dir.append(bandstruc_deriv[2]*E_dir[0] + bandstruc_deriv[3]*E_dir[1])
           je_ortho.append(bandstruc_deriv[2]*E_ort[0] + bandstruc_deriv[3]*E_ort[1] + cross_prod_ortho[1,1,:])
           jh_E_dir.append(bandstruc_deriv[0]*E_dir[0] + bandstruc_deriv[1]*E_dir[1])
           jh_ortho.append(bandstruc_deriv[0]*E_ort[0] + bandstruc_deriv[1]*E_ort[1] + cross_prod_ortho[0,0,:])

       je_E_dir_swapped = np.swapaxes(je_E_dir,0,1)
       je_ortho_swapped = np.swapaxes(je_ortho,0,1)
       jh_E_dir_swapped = np.swapaxes(jh_E_dir,0,1)
       jh_ortho_swapped = np.swapaxes(jh_ortho,0,1)

       # we need tensordot for contracting the first two indices (2 kpoint directions)
       J_E_dir.append(np.tensordot(je_E_dir_swapped,fc[:,:,j_time],2) + np.tensordot(jh_E_dir_swapped,fv[:,:,j_time],2))
       J_ortho.append(np.tensordot(je_ortho_swapped,fc[:,:,j_time],2) + np.tensordot(jh_ortho_swapped,fv[:,:,j_time],2))

     # we need tensordot for contracting the first two indices (2 kpoint directions)
     #J_E_dir = np.tensordot(je_E_dir_swapped,fc,2) + np.tensordot(jh_E_dir_swapped,fv,2)
     #J_ortho = np.tensordot(je_ortho_swapped,fc,2) + np.tensordot(jh_ortho_swapped,fv,2)

    # Return the real part of each component
    return np.real(J_E_dir), np.real(J_ortho)


def get_A_field(E0, w, t, alpha):
    '''
    Returns the analytical A-field as integration of the E-field
    '''
    w_eff = 4*np.pi*alpha*w
    return np.real(-alpha*E0*np.sqrt(np.pi)/2*np.exp(-w_eff**2/4)*(2+erf(t/2/alpha-1j*w_eff/2)-erf(-t/2/alpha-1j*w_eff/2)))

