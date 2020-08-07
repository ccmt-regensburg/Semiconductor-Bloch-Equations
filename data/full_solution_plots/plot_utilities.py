import matplotlib.pyplot as plt
from matplotlib import ticker, cm, colors
import numpy as np
import os


fs_to_au = 41.34137335                  # (1fs    = 41.341473335 a.u.)
E_to_au = 0.0001944690381               # (1MV/cm = 1.944690381*10^-4 a.u.)
THz_to_au = 0.000024188843266           # (1THz   = 2.4188843266*10^-5 a.u.)
ev_to_au = 0.03674932176                # (1eV    = 0.036749322176 a.u.)
au_to_as = 0.529177


def read_full(orderpath, dirpath, parampaths):
    """
    Read data saved by SBE.py
    """
    Soldata = []
    Iexactdata = []

    print("Evaluating " + orderpath + dirpath + " data", end='\n\n')
    for i, massp in enumerate(parampaths):
        totalpath = orderpath + dirpath + massp
        filelist = os.listdir(totalpath)

        for filename in filelist:
            # Emissions I
            # [t, solution, paths, electric_field]
            if ('Sol_' in filename):
                print("Reading :", massp, filename)
                Soldict = np.load(totalpath + filename)
                Soldata.append([Soldict['t'], Soldict['solution'],
                                Soldict['electric_field'], Soldict['paths']])

            # Emissions Iexact
            # [t, I_exact_E_dir, I_exact_ortho, freq/w, Iw_exact_E_dir,
            # Iw_exact_ortho, Int_exact_E_dir, Int_exact_ortho]
            if ('Iexact_' in filename):
                print("Reading :", massp, filename)
                Iexactdata.append(np.load(totalpath + filename))

        print('\n')

    return np.array(Iexactdata), np.array(Soldata)


def read_specific(path):
    """
    Read the data from a specific folder
    """
    filelist = os.listdir(path)
    Soldata = None
    Iexactdata = None

    for filename in filelist:
        # Emissions I
        # [t, solution, electric_field, paths]
        if ('Sol_' in filename):
            print("Reading :", path, filename)
            Soldict = np.load(path + filename)
            Soldata = np.array([Soldict['t'], Soldict['solution'],
                                Soldict['electric_field'], Soldict['paths']])

        # Emissions Iexact
        # [t, I_exact_E_dir, I_exact_ortho, freq/w, Iw_exact_E_dir,
        # Iw_exact_ortho, Int_exact_E_dir, Int_exact_ortho]
        if ('Iexact_' in filename):
            print("Reading :", path, filename)
            Iexactdata = np.array(np.load(path + filename))

    return Iexactdata, Soldata


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
    # ax.set_title(r'Total Intensity')
    ax.set_xlabel(xlabel)
    data_total = data_dir + data_ortho
    for freq, data in zip(freqw, data_total):
        ax.semilogy(freq, data/np.max(data))

    # fig.suptitle(dirname)

    if (paramlegend is not None):
        ax.legend(paramlegend)

    if (savename is None):
        plt.show()
    else:
        plt.savefig(savename)


def plot_time_grid(time, kpath, electric_field, current, band_structure,
                   density_center, standard_deviation=None,
                   electric_field_legend=None,
                   current_legend=None, band_structure_legend=None,
                   density_center_legend=None,
                   standard_deviation_legend=None,
                   timelim=None, energylim=None,
                   bzboundary=None, savename=None):

    # plt.rcParams['figure.figsize'] = (14, 9)
    ########################################
    # Electric field
    ########################################
    ax1 = plt.subplot2grid((2, 6), (0, 0), colspan=3)
    ax1.plot(time/fs_to_au, electric_field.T/E_to_au)
    ax1.set_xlabel(r'$t \text{ in } \si{fs}$')
    ax1.set_ylabel(r'$E \text{ in } \si{MV/cm}$')
    ax1.set_title(r'Electric Field')
    ax1.grid(which='major', axis='x', linestyle='--')
    if (electric_field_legend is not None):
        ax1.legend(electric_field_legend)
    ########################################
    # Current
    ########################################
    ax2 = plt.subplot2grid((2, 6), (0, 3), colspan=3)
    ax2.plot(time/fs_to_au, current.T)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_xlabel(r'$t \text{ in } \si{fs}$')
    ax2.set_ylabel(r'$j$ in atomic units')
    ax2.set_title(r'Current Density')
    ax2.grid(which='major', axis='x', linestyle='--')
    ax2.axhline(y=0, linestyle='--', color='grey')
    if (current_legend is not None):
        ax2.legend(current_legend)

    ########################################
    # Band structure
    ########################################
    kpath_min = np.min(density_center)
    kpath_max = np.max(density_center)
    ax3 = plt.subplot2grid((2, 6), (1, 0), colspan=2)
    # Number of band structures to plot
    band_num = np.size(band_structure, axis=0)
    ax3.plot(band_structure.T, np.tile(kpath, (band_num, 1)).T)
    if (energylim is not None):
        ax3.set_xlim(energylim)
    ax3.set_ylim(-kpath_max-0.05, kpath_max+0.05)
    ax3.set_xlabel(r'$\epsilon \text{ in } \si{eV}$')
    ax3.set_ylabel(r'$k_x \text{ in } \si{1/\angstrom}$')
    ax3.axhline(y=kpath_min, linestyle='--', color='grey')
    ax3.axhline(y=0, linestyle='--', color='grey')
    ax3.axhline(y=kpath_max, linestyle='--', color='grey')
    ax3.axvline(x=0.2, linestyle=':', color='black')
    ax3.set_title(r'Band Structure')
    if (band_structure_legend is not None):
        ax3.legend(band_structure_legend)

    ########################################
    # Density data
    ########################################
    ax4 = plt.subplot2grid((2, 6), (1, 2), colspan=2, sharey=ax3)
    ax4.plot(time/fs_to_au, density_center[:-1].T)
    ax4.plot(time/fs_to_au, density_center[-1], linestyle=':', color='red')
    ax4.set_xlabel(r'$t \text{ in } \si{fs}$')
    ax4.set_title(r'Density Center of Mass')
    ax4.grid(which='major', axis='x', linestyle='--')
    ax4.axhline(y=kpath_min, linestyle='--', color='grey')
    ax4.axhline(y=0, linestyle='--', color='grey')
    ax4.axhline(y=kpath_max, linestyle='--', color='grey')
    plt.setp(ax4.get_yticklabels(), visible=False)
    if (density_center_legend is not None):
        ax4.legend(density_center_legend)

    ax5 = plt.subplot2grid((2, 6), (1, 4), colspan=2)
    ax5.set_title(r'Density Standard Deviation')
    ax5.plot(time/fs_to_au, standard_deviation.T)
    ax5.yaxis.set_label_position("right")
    ax5.yaxis.tick_right()
    ax5.set_xlabel(r'$t \text{ in } \si{fs}$')
    ax5.set_ylabel(r'$\sigma \text{ in } \si{1/\angstrom}$')

    if (standard_deviation_legend is not None):
        ax5.legend(standard_deviation_legend)

    if (timelim is not None):
        ax1.set_xlim(timelim)
        ax2.set_xlim(timelim)
        ax4.set_xlim(timelim)
        ax5.set_xlim(timelim)

    if (bzboundary is not None):
        ax3.set_title(r'Band Structure $k_\mathrm{BZ}='
                      + '{:.3f}'.format(bzboundary) + r'[\si{1/\angstrom}]$')
        ax3.axhline(y=bzboundary, linestyle=':', color='green')
        ax3.axhline(y=-bzboundary, linestyle=':', color='green')
        ax4.axhline(y=bzboundary, linestyle=':', color='green')
        ax4.axhline(y=-bzboundary, linestyle=':', color='green')

    plt.tight_layout()
    if (savename is not None):
        plt.savefig(savename)
    else:
        plt.show()
