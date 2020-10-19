import params

import numpy as np
import os
import matplotlib.pyplot as pl
from matplotlib import patches

from systems import system, dipole
import os
import data_directory

def main():
    #bandstructure()
    dipoles()
    

def dipoles():
    prependix   = constr_prependix()
    n           = 10
    kmax        = np.pi/params.a
    eV_conv     = params.eV_conv
    e_fermi     = params.e_fermi*params.eV_conv

    kx          = np.linspace(-kmax, kmax, n)
    ky          = kx
    x, y        = np.meshgrid(kx, ky)
    Ax, Ay      = dipole.evaluate(x, y)
    meta        = np.sqrt(Ax[1,0,:,:,np.newaxis].real**2+Ay[1,0,:,:,np.newaxis].real**2)
    data        = np.concatenate((x[:,:,np.newaxis]/kmax, y[:,:,np.newaxis]/kmax, Ax[1,0,:,:,np.newaxis].real/meta, Ay[1,0,:,:,np.newaxis].real/meta, (meta)), axis=2).reshape((-1, 5))
    np.savetxt(prependix + 'dipole_cv_real.dat', data, header='x y u v m', comments='')

    meta        = np.sqrt(Ax[1,0,:,:,np.newaxis].imag**2+Ay[1,0,:,:,np.newaxis].imag**2)
    data        = np.concatenate((x[:,:,np.newaxis]/kmax, y[:,:,np.newaxis]/kmax, Ax[1,0,:,:,np.newaxis].imag/meta, Ay[1,0,:,:,np.newaxis].imag/meta, (meta)), axis=2).reshape((-1, 5))
    np.savetxt(prependix + 'dipole_cv_imag.dat', data, header='x y u v m', comments='')

    meta        = np.sqrt(Ax[0,0,:,:,np.newaxis].real**2+Ay[0,0,:,:,np.newaxis].real**2)
    data        = np.concatenate((x[:,:,np.newaxis]/kmax, y[:,:,np.newaxis]/kmax, Ax[0,0,:,:,np.newaxis].real/meta, Ay[0,0,:,:,np.newaxis].real/meta, (meta)), axis=2).reshape((-1, 5))
    np.savetxt(prependix + 'dipole_vv_real.dat', data, header='x y u v m', comments='')

    meta        = np.sqrt(Ax[1,1,:,:,np.newaxis].real**2+Ay[1,1,:,:,np.newaxis].real**2)
    data        = np.concatenate((x[:,:,np.newaxis]/kmax, y[:,:,np.newaxis]/kmax, Ax[1,1,:,:,np.newaxis].real/meta, Ay[1,1,:,:,np.newaxis].real/meta, (meta)), axis=2).reshape((-1, 5))
    np.savetxt(prependix + 'dipole_cc_real.dat', data, header='x y u v m', comments='')

def bandstructure():
    prependix   = constr_prependix()

    n           = 500
    kmax        = np.pi/params.a
    eV_conv     = params.eV_conv
    e_fermi     = params.e_fermi*params.eV_conv

    kx          = np.linspace(0, kmax, n)
    ky          = np.zeros(n)
    zeroEnergy  = system.evaluate_energy(0, 0)[0]
    bandstruct  = np.array(system.evaluate_energy(kx, ky) ) - zeroEnergy

    kx_ind      = np.where(bandstruct[1] - e_fermi < 0)

    pl.plot(kx, bandstruct[1])
    pl.plot(kx, bandstruct[0])
    np.savetxt(prependix + 'bandstructure_conduction_kx.dat', np.transpose([kx/kmax, bandstruct[1]/eV_conv ]) )
    np.savetxt(prependix + 'bandstructure_valence_kx.dat', np.transpose([kx/kmax, bandstruct[0]/eV_conv ]) )

    kx          = np.zeros(n)
    ky          = np.linspace(-kmax, 0, n)
    bandstruct  = system.evaluate_energy(kx, ky) - zeroEnergy

    ky_ind      = np.where(bandstruct[1] - e_fermi < 0)

    pl.plot(ky, bandstruct[1])
    pl.plot(ky, bandstruct[0])
    np.savetxt(prependix + 'bandstructure_conduction_ky.dat', np.transpose([ky/kmax, bandstruct[1]/eV_conv ]) )
    np.savetxt(prependix + 'bandstructure_valence_ky.dat', np.transpose([ky/kmax, bandstruct[0]/eV_conv ]) )

    kx          = np.linspace(0, kmax, n)
    k_gesamt    = np.concatenate((kx[kx_ind], ky[ky_ind]), axis=None)
    
    pl.plot(k_gesamt, np.ones(k_gesamt.size)*e_fermi)
    np.savetxt(prependix + 'bandstructure_fermi_contour.dat', np.transpose([k_gesamt/kmax, np.ones(k_gesamt.size)*e_fermi/eV_conv ]) )
    pl.show()

def constr_prependix():
    prependix   = '/home/nim60855/Documents/masterthesis/thesis/bericht/document/chapters/data/'
    if not os.path.exists(prependix):
        prependix   = '/home/maximilian/Documents/uniclone/masterthesis/thesis/bericht/document/chapters/data/'
    return prependix

if __name__ == "__main__":
    main()
