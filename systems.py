import params

import hfsbe.dipole
import hfsbe.example
import hfsbe.utility

# Set BZ type independent parameters
# Hamiltonian parameters
C0 = params.C0                                    # Dirac point position
C2 = params.C2                                    # k^2 coefficient
A = params.A                                      # Fermi velocity
R = params.R                                      # k^3 coefficient
k_cut = params.k_cut                              # Model hamiltonian cutoff parameter

# Initialize sympy bandstructure, energies/derivatives, dipoles
### Bismuth Teluride calls
system = hfsbe.example.BiTe(C0=C0,C2=C2,A=A,R=R,kcut=k_cut)
### Trivial Bismuth Teluride call
#system = hfsbe.example.BiTeTrivial(C0=C0,C2=C2,R=R,vf=A,kcut=k_cut)
### Periodic Bismuth Teluride call
#system = hfsbe.example.BiTePeriodic(C0=C0,C2=C2,A=A,R=R)
# system = hfsbe.example.BiTePeriodic(default_params=True)
### Haldane calls
#system = hfsbe.example.Haldane(t1=1,t2=1,m=1,phi=np.pi/6,b1=b1,b2=b2)
### Graphene calls
#system = hfsbe.example.Graphene(t=1)
### Dirac calls
#system = hfsbe.example.Dirac(m=0.1)

# Get symbolic hamiltonian, energies, wavefunctions, energy derivatives
#h, ef, wf, ediff = system.eigensystem(gidx=1)
h_symbolic, ef_symbolic, wf_symbolic, ediff_symbolic = system.eigensystem(gidx=1)
eigenvalues = system.ef
# Get symbolic dipoles
dipole = hfsbe.dipole.SymbolicDipole(h_symbolic, ef_symbolic, wf_symbolic)
dipoles_x = dipole.Axf
dipoles_y = dipole.Ayf
