import numpy as np
import os
from params import params

import hfsbe.dipole
import hfsbe.example
from hfsbe.solver.runloops import mkdir_chdir, chirp_phasesweep


def bite():
    # Param file adjustments
    # System parameters
    C2 = 5.39018
    A = 0.19732     # Fermi velocity
    R = 5.52658
    k_cut = 0.05

    system = hfsbe.example.BiTe(C0=0, C2=C2, A=A, R=R, kcut=k_cut, mz=0)
    h_sym, ef_sym, wf_sym, ediff_sym = system.eigensystem(gidx=1)
    dipole = hfsbe.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)

    return system, dipole


if __name__ == "__main__":
    params.w = 25
    params.E0 = 5
    params.e_fermi = 0.2

    params.alpha = 25


    # Double time for broader pulses
    params.t0 *= 2
    params.Nt *= 2

    distlist = [0.01, 0.03]
    T1list = [1000, 10]
    chirplist = [-0.920, -0.460, -0.307]
    phaselist = np.linspace(0, np.pi, 20)

    system, dipole = bite()

    for dist in distlist:
        params.rel_dist_to_Gamma = dist
        dirname_dist = '{:.2f}'.format(dist)  + '_dist'
        mkdir_chdir(dirname_dist)

        for T1 in T1list:
            params.T1 = T1
            params.T2 = 1
            dirname_T = 'T1_' + str(params.T1) + '_T2_' + str(params.T2)
            mkdir_chdir(dirname_T)

            chirp_phasesweep(chirplist, phaselist, system, dipole, params)

            os.chdir('..')
        os.chdir('..')
