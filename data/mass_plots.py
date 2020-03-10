import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

orderpath = './order_4/'
dirpath = 'K_dir/'
masspaths = ['m_00/', 'm_02/', 'm_04/', 'm_06/', 'm_08/', 'm_10/']

plt.subplots(len(masspaths))
print("Evaluating " + orderpath + dirpath + " data", end='\n\n')
for i, massp in enumerate(masspaths):
    totalpath = orderpath + dirpath + massp
    filelist = os.listdir(totalpath)
    # Read electric field only once
    if (i == 0):
        print("Reading electric field:")
        print(filelist[0], end='\n\n')

    # Polarizations
    print("Reading :", massp, filelist[1])
    np.load(totalpath + filelist[1])

    # Currents
    print("Reading :", massp, filelist[2])
    np.load(totalpath + filelist[2])
    # Emissions
    print("Reading :", massp, filelist[3], end='\n\n')
    np.load(totalpath + filelist[3])
