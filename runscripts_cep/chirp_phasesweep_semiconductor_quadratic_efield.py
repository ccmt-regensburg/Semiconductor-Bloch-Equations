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

    # Gaps used in the dirac system
    mz_max = 0.0165372
    mzlist = np.linspace(0, mz_max, 7)
    mz = mzlist[6]

    params.e_fermi = 0.46
    params.rel_dist_to_Gamma = 0.03

    # Adjust bandwidth to the gap and dirac bandwidth
    At = 0.5*(A*3*np.pi/(2*params.a) - mz)
    # Adjusted gap in the semicondcutor
    mz_adjust = mz/At

    # SOC/Dipole parameter
    mx = mz_max/10

    dirname = 'mz_{:.7f}'.format(mz_adjust) + '_A_{:.7f}'.format(At)
    if (not os.path.exists(dirname)):
        os.mkdir(dirname)
    os.chdir(dirname)

    E_max = 10
    Elist = np.linspace(2.5, E_max, 4)
    E = Elist[3]

    params.E0 = E

    dirname_E = 'E_{:.1f}'.format(params.E0)
    if (not os.path.exists(dirname_E)):
        os.mkdir(dirname_E)
    os.chdir(dirname_E)

    # chirplist = np.linspace(-0.920, 0.920, 11)
    chirplist = np.array([-0.920, 0.000])
    for chirp in chirplist:
        params.chirp = chirp
        print("Current chirp: ", params.chirp)
        dirname_chirp = 'chirp_{:1.3f}'.format(params.chirp)
        if (not os.path.exists(dirname_chirp)):
            os.mkdir(dirname_chirp)
        os.chdir(dirname_chirp)

        phaselist = np.linspace(0, np.pi, 20)
        phaselist = [phaselist[0], phaselist[9]]
        for phase in phaselist:
            params.phase = phase
            print("Current phase: ", params.phase)
            dirname_phase = 'phase_{:1.2f}'.format(params.phase)
            if (not os.path.exists(dirname_phase)):
                os.mkdir(dirname_phase)
            os.chdir(dirname_phase)

            system = hfsbe.example.Semiconductor(A=At, mz=mz_adjust, mx=mx,
                                                 a=params.a, align=True)
            h_sym, ef_sym, wf_sym, ediff_sym = system.eigensystem(gidx=1)
            dipole = hfsbe.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)
            solver(system, dipole, params)
            os.chdir('..')

        os.chdir('..')


if __name__ == "__main__":
    run()
