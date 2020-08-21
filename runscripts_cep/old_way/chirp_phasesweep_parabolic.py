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
    mz = mzlist[1]



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

            system = hfsbe.example.BiTe(C0=0, C2=0, A=A, R=0, mz=mz)
            h_sym, ef_sym, wf_sym, ediff_sym = system.eigensystem(gidx=1)
            dipole = hfsbe.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)
            solver(system, dipole, params)
            os.chdir('..')

        os.chdir('..')


if __name__ == "__main__":
    run()
