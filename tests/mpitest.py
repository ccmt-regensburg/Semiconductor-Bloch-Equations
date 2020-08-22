from hfsbe.utility import MpiHelpers
from mpi4py import MPI
import numpy as np


def main(phaselist):

    Multi = MpiHelpers()
    phlist, phlocal, ptuple, displace = Multi.listchop(phaselist)
    Multi.comm.Scatterv([phlist, ptuple, displace, MPI.DOUBLE], phlocal)
    print(Multi.rank)
    print("phlist: ", phlist)
    print("phlocal: ", phlocal)
    print("ptuble: ", ptuple)
    print("displace: ", displace)


if __name__ == "__main__":
    phaselist = np.linspace(0, np.pi, 20)
    main(phaselist)
