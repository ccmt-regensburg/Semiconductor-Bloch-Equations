import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import params

# Fetch parameters from params.py
N_phases  = int(sys.argv[1])
xlims     = [13,22]
phaselims = [0,np.pi]
Nk1       = params.Nk1
Nk2       = params.Nk2
w         = params.w
T2        = params.T2
E0        = params.E0
alpha     = params.alpha
THz_conv  = params.THz_conv
E_conv    = params.E_conv
fs_conv   = params.fs_conv

def cep_plot(x, y, z, xlims, zlabel):
    # Determine maximums of the spectra for color bar values
    x_indices = np.argwhere(np.logical_and(x<xlims[1], x>xlims[0]))
    log_max = np.log(np.max(z[:,x_indices]))/np.log(10)
    log_min = np.log(np.min(z[:,x_indices]))/np.log(10)
    # Set contour spacing
    logspace = np.flip(np.logspace(log_max,log_min,100))
    # Set color bar ticks
    logticks = [np.exp(log_max*np.log(10)),
                np.exp(0.5*(log_min+log_max)*np.log(10)),
                np.exp(log_min*np.log(10))]
    # Meshgrid for xy-plane
    X, Y = np.meshgrid(x, y)
    # Do the plotting
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$\omega/\omega_0$')
    ax.set_ylabel(r'$CEP\ \phi$')
    ax.set_yticks([0,phases[int(np.size(phases)/2)],phases[-1]])
    ax.set_yticklabels([0,'{:d}'.format(int(phases[int(np.size(phases)/2)]/np.pi))
                        +str(r'$\pi$'),'{:d}'.format(int(phases[-1]/np.pi))+str(r'$\pi$')])
    ax.set_yticklabels([0,str(r'$\pi/2$'),str(r'$\pi$')])
    ax.set_xlim(xlims)
    cont = ax.contourf(X, Y, z, levels=logspace, locator=ticker.LogLocator(), cmap=cm.nipy_spectral)
    cbar = fig.colorbar(cont, ax=ax, label=zlabel)
    cbar.set_ticks(logticks)
    cbar.set_ticklabels(['{:.2e}'.format(tick) for tick in logticks])

# Load the files for all phases
phases = np.linspace(phaselims[0],phaselims[1],N_phases+1,endpoint=True)
I         = []
I_Edir    = []
I_ortho   = []
Int_Edir  = []
Int_ortho = []
for i_phase, phase in enumerate(phases):
    filestring = 'I_Nk1-{}_Nk2-{}_w{:4.2f}_E{:4.2f}_a{:4.2f}_ph{:3.2f}_T2-{:05.2f}.npy'
    I_filename = str(filestring).format(Nk1,Nk2,w,E0,alpha,phase,T2)
    I = np.load(I_filename)
    freq  = I[0]
    I_Edir.append(I[5])
    I_ortho.append(I[5])
    Int_Edir.append(I[6])
    Int_ortho.append(I[7])
I_Edir,I_ortho,Int_Edir,Int_ortho = np.array(I_Edir),np.array(I_ortho),np.array(Int_Edir),np.array(Int_ortho)

cep_plot(freq, phases, Int_Edir+Int_ortho, xlims, r'Intensity (a.u.)')
#cep_plot(freq, phases, I_ortho, xlims, r'$I_{\bot}(\omega)$')
#cep_plot(freq, phases, Int_Edir, xlims, r'$E_{\parallel}(\omega)$')
#cep_plot(freq, phases, Int_ortho, xlims, r'$E_{\bot}(\omega)$')

fig, ax = plt.subplots()
ax.semilogy(freq, I_Edir[0]+I_ortho[0], '-', lw=7, zorder=1, label=r'$\phi=0$')
#ax.semilogy(freq, I_Edir[10]+I_ortho[10], '-', lw=5, zorder=2, label=r'$\phi=0$')
#ax.semilogy(freq, I_Edir[20]+I_ortho[20], '--', lw=3, zorder=3, label=r'$\phi=\pi$')
ax.set_xlabel(r'$\omega/\omega_0$')
ax.set_ylabel(r'$Intensity$')
ax.legend()

plt.show()
