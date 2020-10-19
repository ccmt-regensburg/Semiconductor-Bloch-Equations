import numpy as np
import params
import os
import efield
from efield import driving_field
import matplotlib.pyplot as pl
from matplotlib import patches

import data_directory
import rotation_analytical
import scipy.signal as signal

def main():
#    directory	= "../generated_data/" + params.gauge + "_" + str(params.Nk_in_path) + "/"
#    emis	= np.loadtxt(directory + "emission.txt")
#
#    construct_plots(*emis)
    construct_plots()

    return 0

def construct_plots():
    ############ load params #############
    user_out            = params.user_out      
    print_J_P_I_files   = params.print_J_P_I_files   
    energy_plots        = params.energy_plots        
    dipole_plots        = params.dipole_plots        
    test                = params.test                
    matrix_method       = params.matrix_method       
    emission_wavep      = params.emission_wavep      
    Bcurv_in_B_dynamics = params.Bcurv_in_B_dynamics 
    store_all_timesteps = params.store_all_timesteps 
    fitted_pulse        = params.fitted_pulse        
    KK_emission         = params.KK_emission         
    normalize_emission  = params.normalize_emission  
    normalize_f_valence = params.normalize_f_valence 

    gauge               = params.gauge

    a = params.a                                      # Lattice spacing
    b1 = params.b1                                        # Reciprocal lattice vectors
    b2 = params.b2

    BZ_type             = params.BZ_type                          # Type of Brillouin zone to construct
    angle_inc_E_field   = params.angle_inc_E_field      # Angle of driving electric field
    B0                  = params.B0

    fs_conv         = params.fs_conv
    THz_conv        = params.THz_conv
    E_conv          = params.E_conv

    E0              = efield.nir_E0
    alpha           = efield.nir_sigma
    w               = efield.nir_w
    phase           = efield.nir_phi
    
    with_nir        = params.with_nir
    with_transient  = params.with_transient
    realistic_system    = params.realistic_system
    nir_t0          = params.nir_mu

    if BZ_type == 'full':
        Nk1   = params.Nk1                                # Number of kpoints in b1 direction
        Nk2   = params.Nk2                                # Number of kpoints in b2 direction
        Nk    = Nk1*Nk2                                   # Total number of kpoints
        align = params.align                              # E-field alignment
        Nk_in_path  = Nk1
    elif BZ_type == 'full_for_velocity':
        Nk1   = params.Nk1_vel                            # Number of kpoints in b1 direction
        Nk2   = params.Nk2_vel                            # Number of kpoints in b2 direction
        Nk    = Nk1*Nk2                                   # Total number of kpoints
        Nk_in_path  = Nk1
        angle_inc_E_field = params.angle_inc_E_field      # Angle of driving electric field
    elif BZ_type == '2line':
        Nk_in_path = params.Nk_in_path                    # Number of kpoints in each of the two paths
        rel_dist_to_Gamma = params.rel_dist_to_Gamma      # relative distance (in units of 2pi/a) of both paths to Gamma
        length_path_in_BZ = params.length_path_in_BZ      # Length of a single path in the BZ
        angle_inc_E_field = params.angle_inc_E_field      # Angle of driving electric field
        Nk1   = params.Nk_in_path                         # for printing file names, we use Nk1 and ...
        Nk2   = params.num_paths                          # ... and Nk2 = 2
        Nk    = Nk1*Nk2                                   # Total number of k points, we have 2 paths

    T2              = params.T2*fs_conv

    ############ load the data from the files in the given path ###########
    old_directory   = os.getcwd()

    params.with_nir         = False
    params.nir_mu           = 0
    folder  = data_directory.change_to_data_directory()
    params.with_nir         = with_nir
    params.nir_mu           = nir_t0
    nir_t0                  *= fs_conv
    time_window             = 10*alpha

    if not os.path.exists(folder):
        ref_data    = False
        print("Failing to load: ", folder)
        print("For this parameter setting is no reference data available yet.")
        print()
    else:
        ref_data    = False
        os.chdir(folder)
        t_ref, A_field, I_exact_E_dir_ref, I_exact_ortho_ref, I_exact_diag_E_dir, I_exact_diag_ortho, I_exact_offd_E_dir, I_exact_offd_ortho        = np.transpose(np.loadtxt('time.txt') )

        t_ref           *= fs_conv

        I_exact_E_dir_ref   = I_exact_diag_E_dir
        I_exact_ortho_ref   = I_exact_diag_ortho
    
        ref_indices     = np.where(np.abs(t_ref-nir_t0) < time_window)[0]
        
        I_exact_E_dir_ref   = I_exact_E_dir_ref[ref_indices]
        I_exact_ortho_ref   = I_exact_ortho_ref[ref_indices]
        t_ref               = t_ref[ref_indices]
    
    directory           = data_directory.change_to_data_directory()

    if not os.path.exists(directory):
        print("Failing to load: ", directory)
        print("This parameter configuration has not yet been calculated")
        print()
        return 0
    else:
        os.chdir(directory)

    print("Loading data out of " + directory)
    print()

    t, A_field, I_exact_E_dir, I_exact_ortho, I_exact_diag_E_dir, I_exact_diag_ortho, I_exact_offd_E_dir, I_exact_offd_ortho        = np.transpose(np.loadtxt('time.txt') )
    freq, Int_exact_E_dir, Int_exact_ortho, Int_exact_diag_E_dir, Int_exact_diag_ortho, Int_exact_offd_E_dir, Int_exact_offd_ortho  = np.transpose(np.loadtxt('frequency.txt') )
    f_c             = np.transpose(np.loadtxt("conduction_occupation.txt") )
 
    t       *= fs_conv
    dt      = t[1]-t[0]
    freq    *= w
    w_min   = np.argmin(np.abs(freq/w - 0.5 ) )

    time_indices        = np.where(np.abs(np.array(t)-nir_t0) < time_window)[0]
    cut_time            = t[time_indices]

    I_exact_E_dir  = I_exact_E_dir[time_indices]
    I_exact_ortho  = I_exact_ortho[time_indices]

    if ref_data:
        I_nir_E_dir     = I_exact_E_dir[time_indices] - 1*I_exact_E_dir_ref
        I_nir_ortho     = I_exact_ortho[time_indices] - 1*I_exact_ortho_ref
        E_nir_E_dir     = np.diff(I_nir_E_dir)
        E_nir_ortho     = np.diff(I_nir_ortho)


    valence_calculated  = os.path.exists("valence_occupation.txt")
    if valence_calculated:
        f_v         = np.transpose(np.loadtxt("valence_occupation.txt") )

    #os.chdir(old_directory)
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
    if BZ_type == '2line':
        E_dir = np.array([np.cos(np.radians(angle_inc_E_field)),
                         np.sin(np.radians(angle_inc_E_field))])
        dk, kpnts, paths = mesh(params, E_dir)

    if B0 > 1e-15:
        do_B_field = True
    else: 
        do_B_field = False

    Ir = []
    angles = np.linspace(0,2.0*np.pi,360)
    for angle in angles:
        Ir.append((Int_exact_E_dir*np.cos(angle) + Int_exact_ortho*np.sin(-angle)))
    Int_exact_r     = np.fft.fftshift(np.fft.fft(Ir, norm='ortho'))

    freq_indices_near_base_freq = np.argwhere(np.logical_and(freq/w > 0.9, freq/w < 1.1))
    freq_index_base_freq = int((freq_indices_near_base_freq[0] + freq_indices_near_base_freq[-1])/2)
    if normalize_emission:
        Int_tot_base_freq = Int_exact_E_dir[freq_index_base_freq] + Int_exact_ortho[freq_index_base_freq]
        log_limits = (1e-7,1e1)
    else:
        # no normalization at all, no k-point weights
        Int_tot_base_freq = 1

        I_max = (Int_exact_E_dir[freq_index_base_freq] + Int_exact_ortho[freq_index_base_freq]) / Int_tot_base_freq

        freq_indices_near_base_freq = np.argwhere(np.logical_and(freq/w > 19.9, freq/w < 20.1))
        freq_index_base_freq = int((freq_indices_near_base_freq[0] + freq_indices_near_base_freq[-1])/2)
        I_min = (Int_exact_E_dir[freq_index_base_freq] + Int_exact_ortho[freq_index_base_freq] ) / Int_tot_base_freq

        log_limits = ( 10**(np.ceil(np.log10(I_min))-2) , 10**(np.ceil(np.log10(I_max)) + 1) )

    ############ generate plots ###########
    if (not test and user_out):
        real_fig, (axE,axA,axI,axJ,axP) = pl.subplots(5,1,figsize=(10,10))
        factor  = 4
        t_lims = ((-factor*alpha+nir_t0)/fs_conv, (factor*alpha+nir_t0)/fs_conv)
        freq_lims = (0,25)

        axE.set_xlim(t_lims)
        axE.plot(t/fs_conv, driving_field(E0, t)/E_conv)
        axE.set_xlabel(r'$t$ in fs')
        axE.set_ylabel(r'$E$-field in MV/cm')

        axA.set_xlim(t_lims)
        axA.plot(t/fs_conv,A_field/E_conv/fs_conv)
        axA.set_xlabel(r'$t$ in fs')
        axA.set_ylabel(r'$A$-field in MV/cm$\cdot$fs')

        axI.set_xlim(t_lims)
        axI.plot(cut_time/fs_conv,I_exact_E_dir)
        axI2     = axI.twinx()
        axI2.plot(cut_time/fs_conv,I_exact_ortho, color="orange")
        axI.set_xlabel(r'$t$ in fs')
        axI2.set_ylabel(r'$J_{\perp}$ in at.u.')
        axI.set_ylabel(r'$J_{\parallel}$ in at.u.')

        axJ.set_xlim(t_lims)
        axJ2     = axJ.twinx()                
        if ref_data:
            axJ.plot(cut_time/fs_conv, I_nir_E_dir)
            axJ2.plot(cut_time/fs_conv,I_nir_ortho, color="orange")
        axJ.plot(t/fs_conv, I_exact_diag_E_dir, color='blue')
        axJ.plot(t/fs_conv, rotation_analytical.I_exact_diag_E_dir(t), color="green")
        axJ2.plot(t/fs_conv,I_exact_diag_ortho, color="orange")
        axJ.set_xlabel(r'$t$ in fs')
        axJ.set_ylabel(r'$J_{\parallel}^{diag}$ in at.u.')
        axJ2.set_ylabel(r'$J_{\perp}^{diag}$ in at.u.')

        axP.set_xlim(t_lims)
        #axP.plot(cut_time[1:]/fs_conv,np.sign(E_nir_E_dir*E_nir_ortho))
        #axP2     = axP.twinx()                    
        #axP.plot(t/fs_conv,I_exact_offd_E_dir, color='blue')
        #axP.plot(t/fs_conv,rotation_analytical.I_exact_offd_E_dir(t) , color="green")
        axP.plot(t/fs_conv,I_exact_ortho, color="black", label="numerical")
        axP.plot(t/fs_conv,I_exact_offd_ortho, color="orange", label="numerical offd")
        axP.plot(t/fs_conv,rotation_analytical.I_exact_offd_ortho(t), color="red", label="analytical offd")
        #axP2.plot(cut_time/fs_conv,I_analyt_offd_ortho, color="green")
        axP.set_xlabel(r'$t$ in fs')
        #axP.set_ylabel(r'$J_{\parallel}^{offd}$ in at.u.')
        axP.set_ylabel(r'$J_{\perp}^{offd}$ in at.u.')
        axP.legend()


        #axJ.set_xlim(t_lims)
        #axJ.plot(t/fs_conv,I_exact_diag_E_dir)
        #axJ.plot(t/fs_conv,I_exact_diag_ortho)
        #axJ.set_xlabel(r'$t$ in fs')
        #axJ.set_ylabel(r'$J_{diag}$ in at.u. $\parallel \mathbf{E}_{in}$ (blue), $\bot \mathbf{E}_{in}$ (orange)')

        #axP.set_xlim(t_lims)
        #axP.plot(t/fs_conv,I_exact_offd_E_dir)
        #axP.plot(t/fs_conv,I_exact_offd_ortho)
        #axP.set_xlabel(r'$t$ in fs')
        #axP.set_ylabel(r'$J_{offdiag}$ in at.u.')

        #pl.savefig("EAJ.pdf", dpi=300)
        pl.show()
        return

##########################

        if do_B_field:
           label_emission_E_dir = '$I_{\parallel E}(t) = q\sum_{nn\'}\int d\mathbf{k}\;\langle n\overline{\mathbf{k}}_n(t)|\hat{e}_E\cdot \partial h/\partial \mathbf{k}|n\'\overline{\mathbf{k}}_{n\'}(t) \\rangle\\varrho_{nn\'}(\mathbf{k};t)$'
           label_emission_ortho = '$I_{\\bot E}(t) = q\sum_{nn\'}\int d\mathbf{k}\;\langle n\overline{\mathbf{k}}_n(t)|\hat{e}_{\\bot E}\cdot \partial h/\partial \mathbf{k}|n\'\overline{\mathbf{k}}_{n\'}(t) \\rangle\\varrho_{nn\'}(\mathbf{k};t)$'
        else:
           label_emission_E_dir = '$I_{\parallel E}(t) = q\sum_{nn\'}\int d\mathbf{k}\;\langle u_{n\mathbf{k}}|\hat{e}_E\cdot \partial h/\partial \mathbf{k}|u_{n\'\mathbf{k}} \\rangle\\rho_{nn\'}(\mathbf{k},t)$'
           label_emission_ortho = '$I_{\\bot E}(t) = q\sum_{nn\'}\int d\mathbf{k}\;\langle u_{n\mathbf{k}}|\hat{e}_{\\bot E}\cdot \partial h/\partial \mathbf{k}|u_{n\'\mathbf{k}} \\rangle\\rho_{nn\'}(\mathbf{k},t)$'

        if KK_emission:
           five_fig, ((ax_I_E_dir,ax_I_ortho,ax_I_total)) = pl.subplots(3,1,figsize=(10,10))
           ax_I_E_dir.grid(True,axis='x')
           ax_I_E_dir.set_xlim(freq_lims)
           ax_I_E_dir.set_ylim(log_limits)
           ax_I_E_dir.semilogy(freq/w,Int_exact_E_dir / Int_tot_base_freq, label=label_emission_E_dir)
           ax_I_E_dir.semilogy(freq/w, Int_exact_diag_E_dir / Int_tot_base_freq,
               label='$I_{\mathrm{intra}\parallel E}(t) = q\sum_{n= n\'}\int d\mathbf{k}\;\langle u_{n\mathbf{k}}|\hat{e}_E\cdot \partial h/\partial \mathbf{k}|u_{n\'\mathbf{k}} \\rangle\\rho_{nn\'}(\mathbf{k},t)$')
           ax_I_E_dir.semilogy(freq/w, Int_exact_offd_E_dir / Int_tot_base_freq, linestyle='dashed',
               label='$I_{\mathrm{inter}\parallel E}(t) = q\sum_{n\\neq n\'}\int d\mathbf{k}\;\langle u_{n\mathbf{k}}|\hat{e}_E\cdot \partial h/\partial \mathbf{k}|u_{n\'\mathbf{k}} \\rangle\\rho_{nn\'}(\mathbf{k},t)$')

           if not do_B_field:
              ax_I_E_dir.semilogy(freq/w, Iw_E_dir / Int_tot_base_freq, 
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
           ax_I_ortho.semilogy(freq/w, Int_exact_diag_ortho / Int_tot_base_freq,
               label='$I_{\mathrm{intra}\\bot E}(t) = q\sum_{n= n\'}\int d\mathbf{k}\;\langle u_{n\mathbf{k}}|\hat{e}_{\\bot E}\cdot \partial h/\partial \mathbf{k}|u_{n\'\mathbf{k}} \\rangle\\rho_{nn\'}(\mathbf{k},t)$')
           ax_I_ortho.semilogy(freq/w, Int_exact_offd_ortho / Int_tot_base_freq, linestyle='dashed',
               label='$I_{\mathrm{inter}\\bot E}(t) = q\sum_{n\\neq n\'}\int d\mathbf{k}\;\langle u_{n\mathbf{k}}|\hat{e}_{\\bot E}\cdot \partial h/\partial \mathbf{k}|u_{n\'\mathbf{k}} \\rangle\\rho_{nn\'}(\mathbf{k},t)$')
           if not do_B_field:
              ax_I_ortho.semilogy(freq/w,Iw_ortho / Int_tot_base_freq, 
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
              ax_I_total.semilogy(freq/w,(Iw_E_dir+Iw_ortho) / Int_tot_base_freq, 
                 label='$I_{\mathrm{i+i}}(t) = I_{\mathrm{i+i} \parallel E}(t) + I_{\mathrm{i+i} \\bot E}(t)$')
           ax_I_total.set_xlabel(r'Frequency $\omega/\omega_0$')
           ax_I_total.set_ylabel(r'Total emission $I(\omega)$')
           ax_I_total.legend(loc='upper right')
   
           pl.savefig("emission_KKR.pdf", dpi=300)


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

        pl.savefig("emission_exact.pdf", dpi=300)

        # High-harmonic emission polar plots
        polar_fig = pl.figure(figsize=(10, 10))
        i_loop = 1
        i_max = 0
        while i_loop <= i_max:
            freq_indices = np.argwhere(np.logical_and(freq/w > float(i_loop)-0.1, freq/w < float(i_loop)+0.1))
            freq_index   = freq_indices[int(np.size(freq_indices)/2)]
            pax          = polar_fig.add_subplot(1,i_max,i_loop,projection='polar')
            pax.plot(angles,np.abs(Int_exact_r[:,freq_index]))
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

        if BZ_type == "2line":
            kp_array = length_path_in_BZ/(np.pi/a)*np.linspace(-0.5 + (1/(2*Nk1)), 0.5 - (1/(2*Nk1)), num = Nk1)
            # Countour plots of occupations and gradients of occupations
            fig5    = pl.figure()
            if f_c[:,0].size != kp_array.size:
                factor = int(kp_array.size/f_c[:,0].size )
                kp_array    = kp_array[::factor]

            X, Y = np.meshgrid(t/fs_conv,kp_array)
            if gauge == "velocity":
                Y += A_field
            pl.contourf(X, Y, np.real(f_c), 100)
            pl.colorbar().set_label(r'$f_c(k)$ in path 0')
            pl.xlim([t[0]/fs_conv,t[-1]/fs_conv])
            pl.xlabel(r'$t\;[fs]$')
            pl.ylabel(r'$k \; [\frac{\pi}{a}]$')
            pl.tight_layout()
            pl.savefig("conduction.pdf", dpi=300)

            if valence_calculated:
                fig6    = pl.figure()
                X, Y = np.meshgrid(t/fs_conv,kp_array)
                if gauge == "velocity":
                    Y += A_field
                pl.contourf(X, Y, np.real(f_v), 100)
                pl.colorbar().set_label(r'$f_v(k)$ in path 0')
                pl.xlim([t[0]/fs_conv,t[-1]/fs_conv])
                pl.xlabel(r'$t\;(fs)$')
                pl.ylabel(r'$k$')
                pl.tight_layout()
                pl.savefig("valence.pdf", dpi=300)

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

    return

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
    pl.xlim(-25.0/a,25.0/a)
    pl.ylim(-5.0/a,5.0/a)
    pl.xlabel(r'$k_x$ ($1/a_0$)')
    pl.ylabel(r'$k_y$ ($1/a_0$)')

    for path in paths:
        path = np.array(path)
        pl.plot(path[:,0],path[:,1])

    return

def mesh(params, E_dir):
    Nk_in_path        = params.Nk_in_path                    # Number of kpoints in each of the two paths
    rel_dist_to_Gamma = params.rel_dist_to_Gamma      # relative distance (in units of 2pi/a) of both paths to Gamma
    a                 = params.a                                      # Lattice spacing
    length_path_in_BZ = params.length_path_in_BZ      #
    num_paths         = params.num_paths

    alpha_array = np.linspace(-0.5 + (1/(2*Nk_in_path)), 0.5 - (1/(2*Nk_in_path)), num = Nk_in_path)
    vec_k_path = E_dir*length_path_in_BZ

    vec_k_ortho = 2.0*np.pi/a*rel_dist_to_Gamma*np.array([E_dir[1], -E_dir[0]])

    # Containers for the mesh, and BZ directional paths
    mesh = []
    paths = []

    # Create the kpoint mesh and the paths
#    for path_index in [-1, 1]:
    for path_index in np.linspace(-num_paths+1,num_paths-1, num = num_paths):

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



if __name__ == "__main__":
    main()
