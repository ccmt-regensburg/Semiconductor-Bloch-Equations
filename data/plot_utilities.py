import matplotlib.pyplot as plt
import os


def read_data(orderpath, dirpath, parampaths):

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
                    xlim=(0, 30), ylim=(10e-15, 100),
                    xlabel=r'Frequency $\omega/\omega_0$', ylabel=r'a.u.',
                    dirname, savename='data'):

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
    paramlegend = [m.strip('/').replace('_', '=') for m in parampaths]
    ax[0].legend(paramlegend)
    ax[1].legend(paramlegend)
    fig.suptitle(dirname)
    # plt.show()
    plt.savefig(savename)
