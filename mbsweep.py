import numpy as np
import os
from params import params

import hfsbe.dipole
from hfsbe.example import BiTeResummed
import hfsbe.utility


from SBE import main as solver


def run():

    C0 = -0.00647156                  # C0
    c2 = 0.0117598                    # k^2 coefficient
    A = 0.0422927                     # Fermi velocity
    r = 0.109031                      # k^3 coefficient
    ksym = 0.0635012                  # k^2 coefficent dampening
    kasym = 0.113773                  # k^3 coeffcient dampening
    mb = 0.000373195                  # Splitting of cones.(10 meV)

    # Initialize sympy bandstructure, energies/derivatives, dipoles

    # Sweep gap
    for m_pre in np.arange(3.0, 5.5, 1.0):
        print("Current mass: ", m_pre*mb)
        dirname = 'mb_{:.0f}meV'.format(m_pre*10)
        if (not os.path.exists(dirname)):
            os.mkdir(dirname)
        os.chdir(dirname)

        system = BiTeResummed(C0=C0, c2=c2, A=A, r=r, ksym=ksym, kasym=kasym,
                              mb=m_pre*mb)
        h_sym, ef_sym, wf_sym, ediff_sym = system.eigensystem(gidx=1)
        dipole = hfsbe.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)
        solver(system, dipole, params)
        os.chdir('..')


if __name__ == "__main__":
    run()
