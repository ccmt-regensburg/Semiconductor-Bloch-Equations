import numpy as np
import os
from params import params

import hfsbe.dipole
import hfsbe.example
import hfsbe.utility
from hfsbe.solver import sbe_solver

def chirp_phasesweep():
    # Param file adjustments
    # System parameters
    C2 = 5.39018
    A = 0.19732     # Fermi velocity
    R = 5.52658
    k_cut = 0.05

    chirplist = np.array([-0.920, -0.460, -0.307])
    for chirp in chirplist[:]:
        params.chirp = chirp
        print("Current chirp: ", params.chirp)
        dirname_chirp = 'chirp_{:1.3f}'.format(params.chirp)
        if (not os.path.exists(dirname_chirp)):
            os.mkdir(dirname_chirp)
        os.chdir(dirname_chirp)

        phaselist = np.linspace(0, np.pi, 201)
        for phase in phaselist[0:1]:
            params.phase = phase
            print("Current phase: ", params.phase)
            dirname_phase = 'phase_{:1.2f}'.format(params.phase)
            if (not os.path.exists(dirname_phase)):
                os.mkdir(dirname_phase)
            os.chdir(dirname_phase)

            system = hfsbe.example.BiTe(C0=0, C2=C2, A=A, R=R, kcut=k_cut, mz=0)
            h_sym, ef_sym, wf_sym, ediff_sym = system.eigensystem(gidx=1)
            dipole = hfsbe.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)
            sbe_solver(system, dipole, params)
            os.chdir('..')

        os.chdir('..')


if __name__ == "__main__":
    params.w = 25
    params.alpha = 75
    params.e_fermi = 0.2
    params.rel_dist_to_Gamma = 0.03

    # Double time for broader pulses
    params.t0 *= 2
    params.Nt *= 2

    T1list = [1000, 10]

    for T1 in T1list:
        params.T1 = T1
        params.T2 = 1
        dirname_T = 'T1_' + str(params.T1) + '_T2_' + str(params.T2)
        if (not os.path.exists(dirname_T)):
            os.mkdir(dirname_T)
        os.chdir(dirname_T)

        chirp_phasesweep()
