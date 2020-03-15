import os
import subprocess
import fileinput

N_phases = 5
paramFile = 'params.py'

for i in range(N_phases+1):

    phase_div = '(' + str(i) + '/' + str(N_phases) + ')'
    print(phase_div)
    with fileinput.FileInput(paramFile, inplace=True) as file:
        for line in file:
            if 'phase               =' in line:
                print('phase               = '+phase_div+'*np.pi')
            else:
                print(line, end='')

    phase_div_prev = phase_div

    subprocess.Popen(["python3","SBE.py"]).communicate()

with fileinput.FileInput(paramFile, inplace=True) as file:
    for line in file:
         if 'phase               =' in line:
             print('phase               = 0            # Carrier envelope phase (edited by cep-scan.py)')
         else:
             print(line, end='')


# Call plotting script
call = 'python3 cep-plot.py ' + str(N_phases)
os.system(call)
