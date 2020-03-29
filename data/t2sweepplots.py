import os
import numpy as np
import matplotlib.pyplot as plt


fs_conv = 41.34137335                  # (1fs    = 41.341473335 a.u.)
E_conv = 0.0001944690381               # (1MV/cm = 1.944690381*10^-4 a.u.)
THz_conv = 0.000024188843266           # (1THz   = 2.4188843266*10^-5 a.u.)
amp_conv = 150.97488474                # (1A     = 150.97488474)
eV_conv = 0.03674932176                # (1eV    = 0.036749322176 a.u.)


plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['font.size'] = 20

# # Mass evaluation
orderpath = './order_sweep_complete_bz/T2sweep_kcut_dt0.01_C2on_Nk1-400_mb10meV/'
parampaths = ['T2_{:1.0f}/'.format(T2) for T2 in np.arange(1, 4)]

# Use kcut evaluational instead
# orderpath = './kcut/NK2_10/'
# parampaths = ['k_05/', 'k_10/', 'k_15/', 'k_20/', 'k_25/', 'k_30/']

# # Compare evaluation electric fields
# orderpath = './compare/'
# parampaths = ['E_03/', 'E_06/', 'E_12/', 'E_20/', 'E_30/', 'E_40/']

dirpath = 'K_dir/'
dirname = dirpath.strip('/').replace('_', '-').replace('/', '-')


def read_data():
    Idata = []
    Iexactdata = []
    Jdata = []
    Pdata = []

    print("Evaluating " + orderpath + dirpath + " data", end='\n\n')
    for i, massp in enumerate(parampaths):
        totalpath = orderpath + dirpath + massp
        filelist = os.listdir(totalpath)

        for filename in filelist:
            # Read electric field only once
            if (i == 0):
                print("Reading electric field:")
                print(filelist[0], end='\n\n')

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

    return np.array(Idata), np.array(Iexactdata), np.array(Jdata), np.array(Pdata)


def logplot_fourier(freqw, data_dir, data_ortho,
                    xlim=(0, 30), ylim=(10e-15, 10),
                    xlabel=r'Frequency $\omega/\omega_0$', ylabel=r'a.u.',
                    savename='data'):

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
    paramlegend = [m.strip('/').replace('_', '=') + ' fs' for m in parampaths]
    ax[0].legend(paramlegend)
    ax[1].legend(paramlegend)
    fig.suptitle(dirname)
    # plt.show()
    plt.savefig(savename)


if __name__ == "__main__":
    Idata, Iexactdata, Jdata, Pdata = read_data()
    freqw = Idata[:, 3]
    Int_E_dir = Idata[:, 6]
    Int_ortho = Idata[:, 7]
    ylabel = r'$[I](\omega)$ intensity in a.u.'
    logplot_fourier(freqw, Int_E_dir, Int_ortho, ylabel=ylabel,
                    savename='Int-' + dirname)

    freqw = Iexactdata[:, 3]
    Int_exact_E_dir = Iexactdata[:, 6]
    Int_exact_ortho = Iexactdata[:, 7]
    ylabel = r'$[I_\mathrm{exact}](\omega)$ intensity in a.u.'
    logplot_fourier(freqw, Int_exact_E_dir, Int_exact_ortho, ylabel=ylabel,
                    savename='Int-exact-' + dirname)


    # Iw_E_dir = Idata[:, 4]
    # Iw_ortho = Idata[:, 5]
    # ylabel = r'$[\dot P](\omega)$ (total = emitted E-field) in a.u.'
    # logplot_fourier(freqw, np.abs(Iw_E_dir), np.abs(Iw_ortho), ylabel=ylabel,
    #                 savename='Iw-' + dirname)

    # Jw_E_dir = Jdata[:, 4]
    # Jw_ortho = Jdata[:, 5]
    # ylabel = r'$[\dot P](\omega)$ (intraband) in a.u. $\parallel \mathbf{E}_{in}$ (blue), $\bot \mathbf{E}_{in}$ (orange)'
    # logplot_fourier(freqw, np.abs(Jw_E_dir), np.abs(Jw_ortho), ylabel=ylabel,
    #                 savename='Jw-' + dirname)

    # Pw_E_dir = Pdata[:, 4]
    # Pw_ortho = Pdata[:, 5]
    # ylabel = r'$[\dot P](\omega)$ (interband) in a.u. $\parallel \mathbf{E}_{in}$ (blue), $\bot \mathbf{E}_{in}$ (orange)'
    # logplot_fourier(freqw, np.abs(Pw_E_dir), np.abs(Pw_ortho), ylabel=ylabel,
    #                 savename='Pw-' + dirname)
