import numpy as np
import sympy as sp
import os
from params_zeeman import params

from hfsbe.dipole import SymbolicDipole, SymbolicParameterDipole
from hfsbe.example import BiTeResummed

from SBE_zeeman import sbe_zeeman_solver


def run():

    C0 = -0.00647156                  # C0
    c2 = 0.0117598                    # k^2 coefficient
    A = 0.0422927                     # Fermi velocity
    r = 0.109031                      # k^3 coefficient
    ksym = 0.0635012                  # k^2 coefficent dampening
    kasym = 0.113773                  # k^3 coeffcient dampening
    # mb = 0.000373195                  # Splitting of cones.(10 meV)

    # Initialize sympy bandstructure, energies/derivatives, dipoles

    # Sweep electric field
    for E in np.arange(5.00, 5.10, 0.50):

        params.E0 = E
        print("Current E-field: ", params.E0)
        dirname = "E_{:1.2f}".format(params.E0)
        if (not os.path.exists(dirname)):
            os.mkdir(dirname)
        os.chdir(dirname)

        system = BiTeResummed(C0=C0, c2=c2, A=A, r=r, ksym=ksym, kasym=kasym)
        h_sym, e_sym, wf_sym, ediff_sym = system.eigensystem(gidx=1)
        dipole = SymbolicDipole(h_sym, e_sym, wf_sym)

        mb = sp.Symbol("mb", real=True)
        dipole_mb = SymbolicParameterDipole(h_sym, wf_sym, mb)
        sbe_zeeman_solver(system, dipole, dipole_mb, params)
        os.chdir('..')


if __name__ == "__main__":
    run()
