import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import params

# Todo: add labels to figures!

# Fetch parameters from params
Nk1      = 400 #params.Nk1
Nk2      = 2 #params.Nk2
w        = params.w
E0       = params.E0
alpha    = params.alpha
THz_conv = params.THz_conv
E_conv   = params.E_conv
fs_conv  = params.fs_conv

# Load the files for all phases
phases = [0.00,0.31,0.63,0.94,1.26,1.57,1.88,2.20,2.51,2.83,3.14]
I = []
z_I_Edir = []
z_I_ortho = []
z_Int_Edir = []
z_Int_ortho = []
for i_phase, phase in enumerate(phases):
    I_filename = str('I_Nk1-{}_Nk2-{}_w{:4.2f}_E{:4.2f}_a{:4.2f}_ph{:3.2f}.dat.npy').format(Nk1,Nk2,w,E0,alpha,phase)
    I = np.load(I_filename)
    x  = I[0]
    z_I_Edir.append(I[3])
    z_I_ortho.append(I[4])
    z_Int_Edir.append(I[5])
    z_Int_ortho.append(I[6])

# Plot contours
logspace = np.flip(np.logspace(2,-12,100))
X, Y = np.meshgrid(x, phases)
fig, ((ax_Int_Edir,ax_Int_ortho),(ax_I_Edir,ax_I_ortho)) = plt.subplots(2,2)

ax_Int_Edir.set_title('Parallel Intensity')
cont_Int_Edir = ax_Int_Edir.contourf(X, Y, z_Int_Edir, levels=logspace, locator=ticker.LogLocator(), cmap=cm.inferno)
cbar1 = fig.colorbar(cont_Int_Edir, ax=ax_Int_Edir)

ax_Int_ortho.set_title('Orthogonal Intensity')
cont_Int_ortho = ax_Int_ortho.contourf(X, Y, z_Int_ortho, levels=logspace, locator=ticker.LogLocator(), cmap=cm.inferno)
cbar2 = fig.colorbar(cont_Int_ortho, ax=ax_Int_ortho)

ax_I_Edir.set_title('Parallel Emission')
cont_I_Edir = ax_I_Edir.contourf(X, Y, z_I_Edir, levels=logspace, locator=ticker.LogLocator(), cmap=cm.inferno)
cbar3 = fig.colorbar(cont_I_Edir, ax=ax_I_Edir)

ax_I_ortho.set_title('Orthogonal Emission')
cont_I_ortho = ax_I_ortho.contourf(X, Y, z_I_ortho, levels=logspace, locator=ticker.LogLocator(), cmap=cm.inferno)
cbar4 = fig.colorbar(cont_I_ortho, ax=ax_I_ortho)

plt.show()
