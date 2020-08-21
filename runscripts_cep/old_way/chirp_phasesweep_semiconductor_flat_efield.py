import numpy as np
import os
from params import params

import hfsbe.dipole
import hfsbe.example
import hfsbe.utility
# Set BZ type independent parameters
# Hamiltonian parameters
from SBE import main as solver


def run():
    A = 0.19732     # Fermi velocity

    # mz_max = 0.027562
    mx_max = 0.0165372
    mxlist = np.linspace(0, mx_max, 7)
    mx = mxlist[6]
    At = 0.5*(A*3*np.pi/(2*params.a))

    params.e_fermi = 0.2
    params.rel_dist_to_Gamma = 0.03

    dirname = 'mx_{:.7f}'.format(mx)
    if (not os.path.exists(dirname)):
        os.mkdir(dirname)
    os.chdir(dirname)

    E_max = 20
    Elist = np.linspace(2.5, E_max, 8)
    E = Elist[3]

    params.E0 = E
    params.e_fermi = 0

    dirname_E = 'E_{:.1f}'.format(params.E0)
    if (not os.path.exists(dirname_E)):
        os.mkdir(dirname_E)
    os.chdir(dirname_E)

    for chirp in np.linspace(-0.920, 0.920, 11):
        params.chirp = chirp
        print("Current chirp: ", params.chirp)
        dirname_chirp = 'chirp_{:1.3f}'.format(params.chirp)
        if (not os.path.exists(dirname_chirp)):
            os.mkdir(dirname_chirp)
        os.chdir(dirname_chirp)

        for phase in np.linspace(0, np.pi, 20):
            params.phase = phase
            print("Current phase: ", params.phase)
            dirname_phase = 'phase_{:1.2f}'.format(params.phase)
            if (not os.path.exists(dirname_phase)):
                os.mkdir(dirname_phase)
            os.chdir(dirname_phase)

            system = hfsbe.example.Semiconductor(A=At, mx=mx, mz=0, a=params.a,
                                                 align=True)
            h_sym, ef_sym, wf_sym, ediff_sym = system.eigensystem(gidx=1)
            dipole = hfsbe.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)
            solver(system, dipole, params)
            os.chdir('..')

        os.chdir('..')


if __name__ == "__main__":
    run()
