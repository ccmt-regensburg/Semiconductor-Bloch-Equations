import params

import hfsbe.dipole
import hfsbe.example
import hfsbe.utility

# Set BZ type independent parameters
# Hamiltonian parameters
C0                  = 0           # Dirac point position
C2                  = 0           # k^2 coefficient
A                   = 0.1974      # Fermi velocity
R                   = 11.06       # k^3 coefficient
k_cut               = 0.05        # Model hamiltonian cutoff
m                   = 1.0         # Wilson mass
order               = 4           # hz order in periodic hamiltonian
 
# Initialize sympy bandstructure, energies/derivatives, dipoles
# ## Bismuth Teluride calls
# system = hfsbe.example.BiTe(C0=C0, C2=C2, A=A, R=R, kcut=k_cut)
# ## Periodic Bismuth Teluride call
print("mass in params", m)
system = hfsbe.example.BiTePeriodic(A=A, R=R, a=params.a, m=m, order=order)

# Get symbolic hamiltonian, energies, wavefunctions, energy derivatives
# h, ef, wf, ediff = system.eigensystem(gidx=1)
h_sym, ef_sym, wf_sym, ediff_sym = system.eigensystem(gidx=1)

# Assign all energy band functions
evjit, ecjit = system.efjit[0], system.efjit[1]
evdxjit, evdyjit = system.ederivjit[0], system.ederivjit[1]
ecdxjit, ecdyjit = system.ederivjit[2], system.ederivjit[3]

# Get symbolic dipoles
dipole = hfsbe.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)

# Assign all dipole moment functions
di_00xjit = dipole.Axfjit[0][0]
di_01xjit = dipole.Axfjit[0][1]
di_11xjit = dipole.Axfjit[1][1]

di_00yjit = dipole.Ayfjit[0][0]
di_01yjit = dipole.Ayfjit[0][1]
di_11yjit = dipole.Ayfjit[1][1]
