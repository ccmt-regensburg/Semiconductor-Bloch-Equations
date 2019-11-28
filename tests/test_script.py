import numpy as np
import os
import pytest

# THIS SCRIPT NEEDS TO BE EXECUTED IN THE MAIN GIT DIRECTORY BY CALLING python3 tests/test_script.py

def check_test(filename_reference):

   threshold_rel_error = 1.0E-10
   threshold_abs_error = 1.0E-24
   filename = "./test.dat"

   assert os.path.isfile(filename_reference), "Reference file is missing."

   print ("\n\n=====================================================\n\nStart with test:\
           \n\n"+filename_reference+\
          "\n\n=====================================================\n")

   # first line in filename_reference is the command to execute the code
   with open(filename_reference) as f:
       first_line = f.readline()
       os.system(first_line)

   assert os.path.isfile(filename), "Testfile is not printed from the code"

   print ("\n\n*****************************************************\n\nThe following numbers are tested:\n\n")
   print ('{:<15} {:<25} {:<25}'.format("Quantity", "Reference number", "Number obtained from current git version"))
   print ("")

   with open(filename) as f:
       count = 0
       for line in f:
           count += 1
           fields = line.split()
           value_test = float(fields[1])

           with open(filename_reference) as f_reference:

               count_reference = 0
               for line_reference in f_reference:
                   count_reference += 1
                   fields_reference = line_reference.split()

                   # we have the -1 because there is the header with executing command
                   # in the reference file
                   if count == count_reference-1:
                       value_reference = float(fields_reference[1])

                       abs_error = np.abs(value_test - value_reference)
                       rel_error = abs_error/np.abs(value_reference)

                       check_abs = abs_error < threshold_abs_error
                       check_rel = rel_error < threshold_rel_error

                       print('{:<15} {:>25} {:>25}'.format(fields_reference[0], value_reference, value_test))

                       assert check_abs or check_rel, \
                              "\n\nAbsolute and relative error of variable number "+str(count)+\
                              " compared to reference too big:"\
                              "\n\nRelative error: "+str(rel_error)+" and treshold: "+str(threshold_rel_error)+\
                              "\n\nAbsolute error: "+str(abs_error)+" and treshold: "+str(threshold_abs_error)

           f_reference.close()

   print("\n\nTest passed successfully.\n\n")

   f.close()

def main():

   is_dir = os.path.isdir("./tests")
   dirpath = os.getcwd()

   assert is_dir, "The directory ./tests does not exist inside the directory "+dirpath

   count = 0
   for filename_reference in os.listdir("./tests"):
       if filename_reference.endswith(".reference"): 
           count += 1
           check_test("./tests/"+filename_reference)
           continue
       else:
           continue

   assert count > 0, "There are no test files with ending .reference in directory "+dirpath+"/tests"

if __name__ == "__main__":
  main()
