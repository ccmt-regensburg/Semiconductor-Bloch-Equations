import numpy as np
import os
from params import params

import hfsbe.dipole
from hfsbe.example import BiTeResummed
import hfsbe.utility
# Set BZ type independent parameters
# Hamiltonian parameters
from SBE import main as solver


def run():

    C0 = -0.00647156                  # C0
    c2 = 0.0117598                    # k^2 coefficient
    A = 0.0422927                     # Fermi velocity
    r = 0.109031                      # k^3 coefficient
    ksym = 0.0635012                  # k^2 coefficent dampening
    kasym = 0.113773                  # k^3 coeffcient dampening
    # Initialize sympy bandstructure, energies/derivatives, dipoles
    # ## Bismuth Teluride calls
    # system = hfsbe.example.BiTe(C0=C0, C2=C2, A=A, R=R, kcut=k_cut)
    # Sweep Wilson mass
    for phase in np.linspace(0, np.pi, 20):
        params.phase = phase
        print("Current phase: ", params.phase)
        dirname = 'phase_{:1.2f}'.format(params.phase)
        if (not os.path.exists(dirname)):
            os.mkdir(dirname)
        os.chdir(dirname)

        system = BiTeResummed(C0=C0, c2=c2, A=A, r=r, ksym=ksym, kasym=kasym)

        h_sym, ef_sym, wf_sym, ediff_sym = system.eigensystem(gidx=1)
        dipole = hfsbe.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)
        solver(system, dipole, params)
        os.chdir('..')


if __name__ == "__main__":
    run()
