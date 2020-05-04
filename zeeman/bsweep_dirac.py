import numpy as np
import os
from params_zeeman import params

from hfsbe.dipole import SymbolicDipole, SymbolicZeemanDipole
from hfsbe.example import BiTe


from SBE_zeeman import sbe_zeeman_solver


def run():

    # C2                  = 5.39018     # k^2 coefficient
    # A                   = 0.19732     # Fermi velocity
    A = 0.1974
    # R                   = 5.52658     # k^3 coefficient
    # mb                  = 0.000373195 # Splitting of cones.(10 meV)
    # k_cut               = 0.05        # Model hamiltonian cutoff
    # Sweep electric field
    for B in np.arange(0.00, 10.10, 2.00):

        params.B0 = B
        print("Current B-field: ", params.B0)
        dirname = 'B_{:1.2f}'.format(params.B0)
        if (not os.path.exists(dirname)):
            os.mkdir(dirname)
        os.chdir(dirname)

        system = BiTe(C0=0, C2=0, A=A, R=0, zeeman=True)
        h_sym, ef_sym, wf_sym, ediff_sym = system.eigensystem(gidx=1)
        dipole = SymbolicDipole(h_sym, ef_sym, wf_sym)

        dipole_B = SymbolicZeemanDipole(h_sym, wf_sym)
        sbe_zeeman_solver(system, dipole, dipole_B, params)
        os.chdir('..')


if __name__ == "__main__":
    run()
