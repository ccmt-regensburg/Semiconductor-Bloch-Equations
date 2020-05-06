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
    for mu_z in np.arange(1.00, 51.00, 10.00):

        params.mu_z = mu_z
        print("Current mu_z: ", params.mu_z)
        dirname = 'mu_z_{:2.0f}'.format(params.mu_z)
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
