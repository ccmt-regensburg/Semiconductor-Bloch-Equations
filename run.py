import params
import sys

def main():
    params.Nk_in_path       = int(sys.argv[1])
    params.gauge            = str(sys.argv[2])
    params.with_transient   = bool(int(sys.argv[3]) )
    params.realistic_system = bool(int(sys.argv[4]) )

    if int(sys.argv[5]) > 0:
        print("Attempting to plot configuration: ", params.gauge, params.Nk_in_path)
        print("With transient: ", params.with_transient)
        print("Using realistic Hamiltonian parameters: ", params.realistic_system)
        import do_plots
        do_plots.main()
    else:
        print("Calculating configuration: ", params.gauge, params.Nk_in_path)
        print("With transient: ", params.with_transient)
        print("Using realistic Hamiltonian parameters: ", params.realistic_system)
        from SBE import main as solver
        solver()

    return
    
if __name__ == "__main__":
    main()
