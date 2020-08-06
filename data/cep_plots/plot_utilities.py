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

    if (paramlegend is not None):
        ax.legend(paramlegend)

    fig.suptitle(dirname)
    plt.savefig(savename)


def total_fourier(freqw, data_dir, data_ortho,
                  xlim=(0, 30), ylim=(10e-15, 1),
                  xlabel=r'Frequency $\omega/\omega_0$', ylabel=r'a.u.',
                  paramlegend=None, dirname='dir', savename=None):
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
    data_total = data_dir + data_ortho
    for freq, data in zip(freqw, data_total):
        ax.semilogy(freq, data)

    fig.suptitle(dirname)

    if (paramlegend is not None):
        ax.legend(paramlegend)

    if (savename is None):
        plt.show()
    else:
        plt.savefig(savename)


def cep_plot_time(time, phases, data, title=None, xlim=(-8000, 8000), max=None,
                  show=True):
    data = np.real(data)
    if (max is not None):
        data /= np.real(max)

    F, P = np.meshgrid(time[0], phases)
    data_min = np.min(data)
    data_max = np.max(data)

    fig, ax = plt.subplots()
    levels = np.linspace(data_min, data_max, 1000)
    cont = ax.contourf(F, P, data, vmin=data_min, vmax=data_max,
                       cmap=cm.cool, levels=levels)

    cb = plt.colorbar(cont)
    cb.set_label(r'$I/\bar{I}_{\mathrm{max}}$')
    if (max is not None):
        cb.ax.set_title(r'$\bar{I}_{\mathrm{max}} =' + '{:.2e}'.format(max) + r'$')

    # ax.set_xticks(np.arange(xlim[1] + 1))
    # ax.grid(True, axis='x', ls='--')
    ax.set_xlim(xlim)
    ax.set_xlabel(r'$time$')
    ax.set_ylabel(r'phase $\phi$')
    if(title is not None):
        plt.title(title)

    if (show):
        plt.show()


def cep_plot(freqw, phases, data, suptitle=None, title=None, xlim=(0, 35),
             max=1, min=1e-14, show=True):
    data = np.real(data)
    if (max != 1):
        data /= np.real(max)

    log_max = np.log(np.max(data))/np.log(10)
    log_min = np.log(np.min(data[data > min]))/np.log(10)
    F, P = np.meshgrid(freqw[0], phases)

    fig, ax = plt.subplots()
    logspace = np.logspace(log_min, log_max, 100)
    cont = ax.contourf(F, P, data, levels=logspace,
                       locator=ticker.LogLocator(),
                       cmap=cm.gist_ncar,
                       norm=colors.LogNorm(vmin=min, vmax=1e-0))

    min_exponent = int(np.log10(min))
    tickposition = np.logspace(min_exponent, 0, num=np.abs(min_exponent)+1)

    cb = plt.colorbar(cont, ticks=tickposition)
    cb.set_label(r'$I/\bar{I}_{\mathrm{max}}$')
    if (max is not None):
        cb.ax.set_title(r'$\bar{I}_{\mathrm{max}} =' + '{:.2e}'.format(max)
                        + r'\si{[a.u.]}$')

    ax.set_xticks(np.arange(xlim[1] + 1))
    ax.grid(True, axis='x', ls='--')
    ax.set_xlim(xlim)
    ax.set_xlabel(r'$\omega/\omega_0$')
    ax.set_ylabel(r'phase $\phi$')
    if(title is not None):
        plt.title(title)

    if(suptitle is not None):
        plt.suptitle(suptitle)

    if (show):
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
    max_average = np.average(data_max)

    return np.real(max_average), data_max
