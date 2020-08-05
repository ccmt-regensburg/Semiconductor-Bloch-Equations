import matplotlib.pyplot as pl
import matplotlib.units as un
import numpy as np

import params
import SBE
def epsilon(Nk_in_Path, angle_inc_E_field, paths, dk, E_dir):
    beta = params.beta                                            #strength of H_diag
    a = params.a
    gamma = params.gamma*params.angstr_conv**3           #strength of H_SO, units: eV 
    length = -paths[0,0,0]

    #bandstruct = np.zeros(shape=(Nk_in_Path,3))
    #bandstruct[:,0] = np.arange(-length, length+dk, dk)
    kx =  np.arange(-length, length+dk, dk)
    ky = np.arange(-length, length+dk, dk)
    X, Y = np.meshgrid(kx,ky)

    if params.structure_type == "zinc-blende":
        Z1 = -1*(beta*(np.cos(X/length*np.pi)+np.cos(Y/length*np.pi)) + 0.5*gamma*np.sqrt((X**4*Y**2+X**2*Y**4)))           # e_minus or e_v
        Z2 = -1*(beta*(np.cos(X/length*np.pi)+np.cos(Y/length*np.pi)) - 0.5*gamma*np.sqrt((X**4*Y**2+X**2*Y**4)))           # e_plus or e_c

    if params.structure_type == "wurtzite":
        alpha = params.alpha_wz*params.angstr_conv
        Z1 = -1*(beta*(np.cos(X/length*np.pi)+np.cos(Y/length*np.pi)) + 0.5*np.sqrt((alpha-gamma*(X**2+Y**2))**2*(X**2+Y**2)))
        Z2 = -1*(beta*(np.cos(X/length*np.pi)+np.cos(Y/length*np.pi)) - 0.5*np.sqrt((alpha-gamma*(X**2+Y**2))**2*(X**2+Y**2)))

    return X, Y, Z1, Z2, length

Nk_in_Path = params.Nk_in_path
angle_inc_E_field = params.angle_inc_E_field
a = params.a

E_dir = np.array([np.cos(np.radians(angle_inc_E_field)),np.sin(np.radians(angle_inc_E_field))])
dk, kpnts, paths = SBE.mesh(params, E_dir)


levels = [-2.0,-1.8,-1.5, -1.0, -0.5, 0.0,0.5, 1.5,2.5]
fig, (ax,axx) = pl.subplots(2)

X, Y, Z1, Z2, length = epsilon(Nk_in_Path, angle_inc_E_field, paths, dk, E_dir)
tick_locs   = [-length,-0.5*length,0,length*0.5,length]
tick_lbls   = ['-1','-0.5','0','0.5','1']


CS = ax.contour(X,Y,Z2,levels)
ax.clabel(CS, inline=1,fontsize=10)
ax.set_xticks(tick_locs)
ax.set_xticklabels(tick_lbls)
ax.set_yticks(tick_locs)
ax.set_yticklabels(tick_lbls)
ax.set_title(r'$\epsilon_{conduction}$')
ax.set_aspect('equal')
ax.set_xlabel(r'$k_x$ in ($\pi/a$)')
ax.set_ylabel(r'$k_y$ in ($\pi/a$)')
CS2 = axx.contour(X,Y,Z1,levels)
axx.clabel(CS2, inline=1, fontsize=10)
axx.set_xticks(tick_locs)
axx.set_xticklabels(tick_lbls)
axx.set_yticks(tick_locs)
axx.set_yticklabels(tick_lbls)
axx.set_aspect('equal')
axx.set_title(r'$\epsilon_{valence}$')
axx.set_xlabel(r'$k_x$ in ($\pi/a$)')
axx.set_ylabel(r'$k_y$ in ($\pi/a$)')
pl.show()


