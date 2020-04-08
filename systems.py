import params
from copy import deepcopy

import hfsbe.dipole
import hfsbe.example

# Set BZ type independent parameters
# Hamiltonian parameters
C0 = params.C0                             # Dirac point position
C2 = params.C2                             # k^2 coefficient
A = params.A                               # Fermi velocity
R = params.R                               # k^3 coefficient
k_cut = params.k_cut                       # Model hamiltonian cutoff parameter

# Initialize sympy bandstructure, energies/derivatives, dipoles
# ## Bismuth Teluride calls
system = hfsbe.example.BiTe(C0=C0, C2=C2, A=A, R=R, kcut=k_cut)
# ## Trivial Bismuth Teluride call
# system = hfsbe.example.BiTeTrivial(C0=C0,C2=C2,R=R,vf=A,kcut=k_cut)
# ## Periodic Bismuth Teluride call
# system = hfsbe.example.BiTePeriodic(C0=C0,C2=C2,A=A,R=R)
# system = hfsbe.example.BiTePeriodic(default_params=True)
# ## Haldane calls
# system = hfsbe.example.Haldane(t1=1,t2=1,m=1,phi=np.pi/6,b1=b1,b2=b2)
# ## Graphene calls
# system = hfsbe.example.Graphene(t=1)
# ## Dirac calls
# system = hfsbe.example.Dirac(m=0.1)

# Get symbolic hamiltonian, energies, wavefunctions, energy derivatives
# h, ef, wf, ediff = system.eigensystem(gidx=1)
h_sym, ef_sym, wf_sym, ediff_sym = system.eigensystem(gidx=1)

# Assign all energy band functions
evjit, ecjit = system.efjit[0], system.efjit[1]

# for improved emission formula, we need derivative of the Hamiltonian 
h_deriv = system.hderivfjit

# for B-field dynamics, we need fast bandstructure derivative
ev_dx = system.ederivfjit[0]
ev_dy = system.ederivfjit[1]
ec_dx = system.ederivfjit[2]
ec_dy = system.ederivfjit[3]

# 
wf = system.Uf
wf_h = system.Uf_h

# Get symbolic dipoles
dipole = hfsbe.dipole.SymbolicDipole(h_sym, ef_sym, wf_sym)


# Assign all dipole moment functions
di_00xjit = dipole.Axfjit[0][0]
di_01xjit = dipole.Axfjit[0][1]
di_11xjit = dipole.Axfjit[1][1]

di_00yjit = dipole.Ayfjit[0][0]
di_01yjit = dipole.Ayfjit[0][1]
di_11yjit = dipole.Ayfjit[1][1]

curv = hfsbe.dipole.SymbolicCurvature(dipole.Ax, dipole.Ay)
cu_00jit = curv.Bfjit[0][0]
cu_01jit = curv.Bfjit[0][1]
cu_11jit = curv.Bfjit[1][1]
