import os
import subprocess
import fileinput

N_phases = 20
paramFile = 'params.py'

phase_div_prev = '(0/1)'
for i in range(N_phases+1):

    phase_div = '(' + str(i) + '/' + str(N_phases) + ')'
    print(phase_div)
    with fileinput.FileInput(paramFile, inplace=True) as file:
        for line in file:
            print(line.replace(phase_div_prev,phase_div), end='')
    phase_div_prev = phase_div

    subprocess.Popen(["python3","SBE.py"]).communicate()

with fileinput.FileInput(paramFile, inplace=True) as file:
    for line in file:
        print(line.replace(phase_div_prev,'(0/1)'), end='')

# Call plotting script
call = 'python3 cep-plot.py ' + str(N_phases)
os.system(call)
