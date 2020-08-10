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

    mz = 0

    params.e_fermi = 0.2
    params.rel_dist_to_Gamma = 0.03

    E_max = 10
    Elist = np.linspace(2.5, E_max, 4)
    E = Elist[1]

    params.E0 = E

    dirname_E = 'E_{:.1f}'.format(params.E0)

    # if (not os.path.exists(dirname_E)):
        # os.mkdir(dirname_E)
    # os.chdir(dirname_E)

    # chirplist = np.linspace(-0.920, 0.920, 11)
    chirplist = np.array([-0.920])
    # [-0.920, -0.736, -0.552, -0.368, -0.184, 0.000, 0.184, 0.368, 0.552, 0.736, 0.920]
    for chirp in chirplist[:]:
        params.chirp = chirp
        print("Current chirp: ", params.chirp)
        dirname_chirp = 'chirp_{:1.3f}'.format(params.chirp)
        if (not os.path.exists(dirname_chirp)):
            os.mkdir(dirname_chirp)
        os.chdir(dirname_chirp)

        phaselist = np.linspace(0, np.pi, 20)
        # phaselist = [phaselist[0], phaselist[9]]
        for phase in phaselist[0:2]:
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
