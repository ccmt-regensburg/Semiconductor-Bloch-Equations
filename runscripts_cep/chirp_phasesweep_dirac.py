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
    mz_max = 0.0165372
    mzlist = np.linspace(0, mz_max, 7)
    mz = mzlist[6]

    params.e_fermi = 0.2
    params.rel_dist_to_Gamma = 0.03

    dirname_mz = 'mz_{:.7f}'.format(mz)
    if (not os.path.exists(dirname_mz)):
        os.mkdir(dirname_mz)
    os.chdir(dirname_mz)

    chirplist = np.linspace(-0.920, 0.920, 11)
    # [-0.920, -0.736, -0.552, -0.368, -0.184, 0.000, 0.184, 0.368, 0.552, 0.736, 0.920]
    for chirp in chirplist[6:]:
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

            system = hfsbe.example.BiTe(C0=0, C2=0, A=A, R=0, mz=mz)
            h_sym, ef_sym, wf_sym, ediff_sym = system.eigensystem(gidx=1)
            dipole = hfsbe.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)
            solver(system, dipole, params)
            os.chdir('..')

        os.chdir('..')


if __name__ == "__main__":
    run()
