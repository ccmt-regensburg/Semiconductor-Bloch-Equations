import os
import numpy as np
import matplotlib.pyplot as plt

fs_conv = 41.34137335                  # (1fs    = 41.341473335 a.u.)
E_conv = 0.0001944690381               # (1MV/cm = 1.944690381*10^-4 a.u.)
THz_conv = 0.000024188843266           # (1THz   = 2.4188843266*10^-5 a.u.)
amp_conv = 150.97488474                # (1A     = 150.97488474)
eV_conv = 0.03674932176                # (1eV    = 0.036749322176 a.u.)


plt.rcParams['text.usetex'] = True

orderpath = './order_4/NK2_10/'
dirpath = 'M_dir/'
masspaths = ['m_00/', 'm_02/', 'm_04/', 'm_06/', 'm_08/', 'm_10/']


def read_data():
    Idata = []
    Jdata = []
    Pdata = []

    print("Evaluating " + orderpath + dirpath + " data", end='\n\n')
    for i, massp in enumerate(masspaths):
        totalpath = orderpath + dirpath + massp
        filelist = os.listdir(totalpath)
        filelist.sort()

        # Read electric field only once
        if (i == 0):
            print("Reading electric field:")
            print(filelist[0], end='\n\n')

        # Emissions I
        # [t, I_E_dir, I_ortho, freq/w, abs(Iw_E_dir), abs(Iw_ortho),
        # Int_E_dir, Int_ortho]
        print("Reading :", massp, filelist[3], end='\n\n')
        Idata.append(np.load(totalpath + filelist[3]))

        # Currents J
        # [t, J_E_dir, J_ortho, freq/w, Jw_E_dir, Jw_Ortho]
        print("Reading :", massp, filelist[2])
        Jdata.append(np.load(totalpath + filelist[2]))

        # Polarizations P
        # [t, P_E_dir, P_ortho, freq/w, Pw_E_dir, Pw_ortho]
        print("Reading :", massp, filelist[1])
        Pdata.append(np.load(totalpath + filelist[1]))

    return Idata, Jdata, Pdata


def Pplot(Pdata):
    fig, ax = plt.subplots(len(masspaths))
    for i, Pdat in enumerate(Pdata):
        t = Pdat[0]
        P_E_dir = Pdat[1]
        P_ortho = Pdat[2]
        freqw = Pdat[3]
        Pw_E_dir = Pdat[4]
        Pw_ortho = Pdat[5]

        ax[i].grid(True, axis='x')
        ax[i].semilogy(freqw, np.abs(Pw_E_dir))
        ax[i].semilogy(freqw, np.abs(Pw_ortho))

    plt.show()


def Jplot(Jdata):
    fig, ax = plt.subplots(len(masspaths))
    for i, Jdat in enumerate(Jdata):
        t = Jdat[0]
        J_E_dir = Jdat[1]
        J_ortho = Jdat[2]
        freqw = Jdat[3]
        Jw_E_dir = Jdat[4]
        Jw_Ortho = Jdat[5]

        ax[i].grid(True, axis='x')
        ax[i].semilogy(freqw, np.abs(Jw_E_dir))
        ax[i].semilogy(freqw, np.abs(Jw_Ortho))

    plt.show()


def Iplot(Idata):
    fig, ax = plt.subplots(2)
    for a in ax:
        a.set_xlim((0, 30))
        a.grid(True, axis='x')
    ax[0].set_ylim((10e-20, 10))
    ax[1].set_ylim((10e-20, 10e-7))

    for i, Idat in enumerate(Idata):
        t = Idat[0]
        I_E_dir = Idat[1]
        I_ortho = Idat[2]
        freqw = Idat[3]
        Iw_E_dir = Idat[4]
        Iw_ortho = Idat[5]
        Int_E_dir = Idat[6]
        Int_ortho = Idat[7]
        ax[0].semilogy(freqw, Int_E_dir)
        ax[1].semilogy(freqw, Int_ortho)

    plt.show()



if __name__ == "__main__":
    Pdata, Jdata, Idata = read_data()
    Iplot(Idata)
