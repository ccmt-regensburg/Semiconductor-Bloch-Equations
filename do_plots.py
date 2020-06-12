import SBE as sbe
import numpy as np
import params

def main():
    dir	= "../generated_data/" + params.gauge + "_" + str(params.Nk_in_path) + "/"
    emis	= np.loadtxt(dir + "emission.txt")
    besetzung	= np.loadtxt(dir + "besetzung.txt")

    sbe.constructPlots(*emis, besetzung)

    return 0

if __name__ == "__main__":
    main()
