import matplotlib.pyplot as plt
from matplotlib import ticker, cm, colors
import numpy as np
import os


def read_data(orderpath, dirpath, parampaths):
    """
    Read data saved by SBE.py
    """
    Idata = []
    Iexactdata = []
    Jdata = []
    Pdata = []

    print("Evaluating " + orderpath + dirpath + " data", end='\n\n')
    for i, massp in enumerate(parampaths):
        totalpath = orderpath + dirpath + massp
        filelist = os.listdir(totalpath)

        for filename in filelist:
            # Emissions I
            # [t, I_E_dir, I_ortho, freq/w, abs(Iw_E_dir), abs(Iw_ortho),
            # Int_E_dir, Int_ortho]
            if ('I_' in filename):
                print("Reading :", massp, filename)
                Idata.append(np.load(totalpath + filename))

            # Emissions Iexact
            # [t, I_exact_E_dir, I_exact_ortho, freq/w, Iw_exact_E_dir,
            # Iw_exact_ortho, Int_exact_E_dir, Int_exact_ortho]
            if ('Iexact_' in filename):
                print("Reading :", massp, filename)
                Iexactdata.append(np.load(totalpath + filename))

            # Currents J
            # [t, J_E_dir, J_ortho, freq/w, Jw_E_dir, Jw_Ortho]
            if ('J_' in filename):
                print("Reading :", massp, filename)
                Jdata.append(np.load(totalpath + filename))

            # Polarizations P
            # [t, P_E_dir, P_ortho, freq/w, Pw_E_dir, Pw_ortho]
            if ('P_' in filename):
                print("Reading :", massp, filename)
                Pdata.append(np.load(totalpath + filename))

        print('\n')

    return np.array(Idata), np.array(Iexactdata), np.array(Jdata), \
        np.array(Pdata)


def dir_ortho_fourier(freqw, data_dir, data_ortho,
                      xlim=(0, 30), ylim=(10e-15, 1),
                      xlabel=r'Frequency $\omega/\omega_0$', ylabel=r'a.u.',
                      paramlegend=None, dirname='dir', savename='data'):

    fig, ax = plt.subplots(2)
    for a in ax:
        a.set_xlim(xlim)
        a.set_ylim(ylim)
        a.set_xticks(np.arange(31))
        a.grid(True, axis='x', ls='--')
        a.set_ylabel(ylabel)
    ax[0].set_title(r'$\mathbf{E}$ parallel')
    ax[1].set_title(r'$\mathbf{E}$ orthogonal')
    ax[1].set_xlabel(xlabel)
    for freq, data_d, data_o in zip(freqw, data_dir, data_ortho):
        ax[0].semilogy(freq, data_d)
        ax[1].semilogy(freq, data_o)
    ax[0].legend(paramlegend)
    ax[1].legend(paramlegend)
    fig.suptitle(dirname)
    plt.savefig(savename)


def simple_fourier(freqw, data,
                   xlim=(0, 30), ylim=(10e-15, 1),
                   xlabel=r'Frequency $\omega/\omega_0$', ylabel=r'a.u.',
                   paramlegend=None, dirname='dir', savename='data'):
    """
    Plots only one dataset
    """
    fig, ax = plt.subplots(1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(np.arange(31))
    ax.grid(True, axis='x', ls='--')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.semilogy(freqw[0], data.T)

    ax.legend(paramlegend)
    fig.suptitle(dirname)
    plt.savefig(savename)


def total_fourier(freqw, data_dir, data_ortho,
                  xlim=(0, 30), ylim=(10e-15, 1),
                  xlabel=r'Frequency $\omega/\omega_0$', ylabel=r'a.u.',
                  paramlegend=None, dirname='dir', savename='data'):
    """
    Plots parallel and orthogonal data
    """
    fig, ax = plt.subplots(1)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(np.arange(31))
    ax.grid(True, axis='x', ls='--')
    ax.set_ylabel(ylabel)
    ax.set_title(r'Total Intensity')
    ax.set_xlabel(xlabel)
    for freq, data in zip(freqw, data_dir+data_ortho):
        ax.semilogy(freq, data)
    ax.legend(paramlegend)
    fig.suptitle(dirname)
    plt.savefig(savename)


def cep_plot_tmp(phases, x, y, z, xlims, zlabel):
    # Determine maximums of the spectra for color bar values
    x_indices = np.argwhere(np.logical_and(x < xlims[1], x > xlims[0]))
    log_max = np.log(np.max(z[:, x_indices]))/np.log(10)
    log_min = np.log(np.min(z[:, x_indices]))/np.log(10)
    log_min = log_max - np.ceil(log_max-log_min)

    # Set contour spacing
    logspace = np.flip(np.logspace(log_max, log_min, 100))
    # Set color bar ticks
    exp_of_ticks = np.linspace(log_min, log_max, int(log_max)-int(log_min)+1)
    logticks = np.exp(exp_of_ticks*np.log(10))

    # Meshgrid for xy-plane
    X, Y = np.meshgrid(x[x_indices[:, 0]], y)
    # Do the plotting
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$\omega/\omega_0$')
    ax.set_ylabel(r'$CEP\ \phi$')
    ax.set_yticks([0, phases[-1]/2, phases[-1]])
    ax.set_yticklabels([0, str(r'$\pi/2$'), str(r'$\pi$')])
    ax.set_xlim(xlims)
    cont = ax.contourf(X, Y, z[:, x_indices[:, 0]], levels=logspace,
                       locator=ticker.LogLocator(), cmap=cm.nipy_spectral)
    cbar = fig.colorbar(cont, ax=ax, label=zlabel)
    cbar.set_ticks(logticks)
    cbar.set_ticklabels(['$10^{{{}}}$'.format(int(round(tick-exp_of_ticks[-1]))) for tick in exp_of_ticks])


def cep_plot(freqw, phases, data, title, xlim=(0, 30), max=None):
    data = np.real(data)
    if (max is not None):
        data /= np.real(max)

    min = 1e-14
    log_max = np.log(np.max(data))/np.log(10)
    log_min = np.log(np.min(data[data > min]))/np.log(10)
    F, P = np.meshgrid(freqw[0], phases)

    # breakpoint()
    logspace = np.logspace(log_min, log_max, 100)
    cont = plt.contourf(F, P, data, levels=logspace,
                        locator=ticker.LogLocator(),
                        cmap=cm.nipy_spectral,
                        norm=colors.LogNorm(vmin=min, vmax=1))
    plt.xlim(xlim)
    plt.xlabel(r'$\omega/\omega_0$')
    plt.ylabel(r'phase $\phi$')
    cb = plt.colorbar(cont, ticks=[1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-0])
    cb.set_label(r'$I/\bar{I}_{\mathrm{max}}$')
    if (max is not None):
        cb.ax.set_title(r'$\bar{I}_{\mathrm{max}} =' + '{:.2e}'.format(max) + r'$')
    # cb.ax.tick_params(labelsize=7)
    # cb.ax.set_yticklabels(np.logspace(1e-14, 1, 11))
    plt.title(title)
    plt.show()


def find_base_freq(freqw, data_dir, data_ortho):
    """
    Find the amplitude at the base frequency for every data set
    for normalizations
    """

    base_frequency = []
    for i, freq in enumerate(freqw):
        # Closest frequency index to 1
        idx = np.abs(freqw - 1).argmin()
        base_frequency.append(data_dir[i, idx] + data_ortho[i, idx])

    return np.array(base_frequency)


def find_max_intens(freqw, data_dir, data_ortho):
    """
    Find the amplitude at the base frequency for every data set
    for normalizations
    """
    size = np.size(freqw, axis=1)
    # Only take right half of results
    data = data_dir[:, size//2:] + data_ortho[:, size//2:]
    data_max = np.max(data, axis=1)
    max = np.average(data_max)

    return np.real(max)
