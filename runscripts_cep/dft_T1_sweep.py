import numpy as np
import os
from params import params

import hfsbe.dipole
import hfsbe.example
from hfsbe.solver.runloops import mkdir_chdir, chirp_phasesweep 


def dft():
    C0 = -0.00647156                  # C0
    c2 = 0.0117598                    # k^2 coefficient
    A = 0.0422927                     # Fermi velocity
    r = 0.109031                      # k^3 coefficient
    ksym = 0.0635012                  # k^2 coefficent dampening
    kasym = 0.113773                  # k^3 coeffcient dampening

    system = hfsbe.example.BiTeResummed(C0=C0, c2=c2, A=A, r=r, ksym=ksym, kasym=kasym)
    h_sym, ef_sym, wf_sym, ediff_sym = system.eigensystem(gidx=1)
    dipole = hfsbe.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)

    return system, dipole


if __name__ == "__main__":
    params.w = 25
    params.E0 = 5
    params.e_fermi = 0.0

    params.alpha = 25

    # Double time for broader pulses
    params.t0 *= 2
    params.Nt *= 2

    distlist = [0.01, 0.03]
    T1list = [1000, 10]
    chirplist = [-0.920, -0.460, -0.307]
    phaselist = np.linspace(0, np.pi, 20)

    system, dipole = dft()

    for dist in distlist:
        params.rel_dist_to_Gamma = dist
        dirname_dist =  '{:.2f}'.format(dist) +  '_dist'
        mkdir_chdir(dirname_dist)

        for T1 in T1list:
            params.T1 = T1
            params.T2 = 1
            dirname_T = 'T1_' + str(params.T1) + '_T2_' + str(params.T2)
            mkdir_chdir(dirname_T)

            chirp_phasesweep(chirplist, phaselist, system, dipole, params)

            os.chdir('..')
        os.chdir('..')
