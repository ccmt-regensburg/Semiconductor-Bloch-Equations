import dill
import numpy as np
import os
import sympy as sp

from hfsbe.dipole import SymbolicDipole
from hfsbe.example import BiTe, TwoBandSystem

import params


orderpath = './data/order_sweep_complete_bz/E5MV_order4_dt0.01_C2off_Nk1-400/'
dirpath = 'M_dir/'
parampath = 'mw_0.00/'


def read_data():

    totalpath = orderpath + dirpath + parampath
    filelist = os.listdir(totalpath)

    for filename in filelist:
        if ('Full_' in filename):
            print("Reading full solution:")
            full = np.load(totalpath + filename, allow_pickle=True)


    breakpoint()
    print(type(full['system']))
    system = dill.loads(full['system'])
    dipole = dill.loads(full['dipole'])
    params = dill.loads(full['params'])
    paths = full['paths']
    time = full['time']
    solution = full['solution']
    driving_field = full['driving_field']

    return system, dipole, params, paths, time, solution, driving_field


if __name__ == "__main__":
    system, dipole, params, paths, time, solution, driving_field = read_data()
    print(params.C2)
