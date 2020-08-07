import params
import sys

def main():
    params.Nk_in_path       = int(sys.argv[1])
    params.gauge            = str(sys.argv[2])
    params.nir_mu           = int(sys.argv[3])
    params.transient_number = int(sys.argv[4])
    params.realistic_system = bool(int(sys.argv[5]) )
    order                   = int(sys.argv[6])
    params.tra_fac          = float(sys.argv[7])
    params.nir_fac          = float(sys.argv[8])

    if params.transient_number == -1:
        params.with_transient = False
        params.transient_number == 0

    if order  == 1:
        print("Attempting to plot configuration: ", params.gauge, params.Nk_in_path)
        print("Transient to be used: ", params.transient_number)
        print("Nir delay in fs: ", params.nir_mu)
        print("Using realistic Hamiltonian parameters: ", params.realistic_system)
        import do_plots
        do_plots.main()

    elif order == 2:
        print("Attempting to calculate the polarization rotation: ", params.gauge, params.Nk_in_path)
        print("Transient to be used: ", params.transient_number)
        print("Nir delay in fs: ", params.nir_mu)
        print("Using realistic Hamiltonian parameters: ", params.realistic_system)
        import polarization_rotation
        polarization_rotation.main()

    else:
        print("Calculating configuration: ", params.gauge, params.Nk_in_path)
        print("Transient to be used: ", params.transient_number)
        print("Nir delay in fs: ", params.nir_mu)
        print("Using realistic Hamiltonian parameters: ", params.realistic_system)
        from SBE import main as solver
        solver()

    return
    
if __name__ == "__main__":
    main()
