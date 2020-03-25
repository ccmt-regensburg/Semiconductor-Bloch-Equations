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

    C2                  = 5.39018     # k^2 coefficient
    A                   = 0.19732     # Fermi velocity
    R                   = 5.52658     # k^3 coefficient
    k_cut               = 0.05        # Model hamiltonian cutoff
    m                   = 0.70        # Wilson mass
    order               = 4           # hz order in periodic hamiltonian
     
    # Initialize sympy bandstructure, energies/derivatives, dipoles
    # ## Bismuth Teluride calls
    # system = hfsbe.example.BiTe(C0=C0, C2=C2, A=A, R=R, kcut=k_cut)
    # ## Periodic Bismuth Teluride call
    for m in np.linspace(0.0, 1.0, 0.05):
        dirname = 'm_{:1.2f}'.format(m)
        os.mkdir(dirname)
        os.chdir(dirname)
        system = hfsbe.example.BiTePeriodic(C2=C2, A=A, R=R, a=params.a, m=m,
                                            order=order)
        h_sym, ef_sym, wf_sym, ediff_sym = system.eigensystem(gidx=1)
        dipole = hfsbe.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)
        solver(system, dipole)


if __name__ == "__main__":
    run()
