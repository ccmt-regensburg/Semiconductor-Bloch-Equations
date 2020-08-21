import numpy as np
import os
from params import params

import hfsbe.dipole
import hfsbe.example
from hfsbe.solver.runloops import mkdir_chdir, chirp_phasesweep


def semich():
    # Hamiltonian Parameters
    A = 0.19732     # Fermi velocity

    # Gaps used in the dirac system
    mz = 0.0110248
    mz *= 0.3

    # Adjust bandwidth to the gap and dirac bandwidth
    At = 0.5*(A*3*np.pi/(2*params.a) - mz)
    # Adjusted gap in the semicondcutor
    mz_adjust = mz/At

    # SOC/Dipole parameter
    mx = 0.00165372
    mx *= 5

    system = hfsbe.example.Semiconductor(A=At, mz=mz_adjust, mx=mx,
                                         a=params.a, align=True)
    h_sym, ef_sym, wf_sym, ediff_sym = system.eigensystem(gidx=1)
    dipole = hfsbe.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)

    return system, dipole


if __name__ == "__main__":
    params.w = 40
    params.E0 = 5
    params.e_fermi = 0.2

    params.alpha = 75
    params.rel_dist_to_Gamma = 0.03

    # Double time for broader pulses
    params.t0 *= 2
    params.Nt *= 2

    distlist = [0.01, 0.03]
    T1list = [1000, 10]

    system, dipole = semich()

    for dist in distlist:
        chirplist = [-0.920, -0.460, -0.307]
        phaselist = [0]
        dirname_dist = 'dist_' + '{:.3f}'.format(dist)
        mkdir_chdir(dirname_dist)

        for T1 in T1list:
            params.T1 = T1
            params.T2 = 1
            dirname_T = 'T1_' + str(params.T1) + '_T2_' + str(params.T2)
            mkdir_chdir(dirname_T)

            chirp_phasesweep(chirplist, phaselist, system, dipole, params)

            os.chdir('..')
        os.chdir('..')
