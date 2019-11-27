import numpy as np
import os
import pytest

def main():

   for filename_reference in os.listdir("."):
       if filename_reference.endswith(".test"): 
           print(filename_reference)
           continue
       else:
           continue

   filename_reference = "01_1d_model_bandstructure_5_kpoints_and_default_E_field.test"
   threshold_rel_error = 1.0E-12
   threshold_abs_error = 1.0E-24
   filename = "test.dat"


   assert os.path.isfile(filename_reference), "Reference file is missing."

   print ("\n=========================\nStart with test number xyz\
           \n=========================\n")

   # first line in filename_reference is the command to execute the code
   with open(filename_reference) as f:
       first_line = f.readline()
       os.system(first_line)

   assert os.path.isfile(filename), "Testfile is not printed from the code"

   with open(filename) as f:
       count = 0
       for line in f:
           count += 1
           fields = line.split()
           print(fields[0])
           print(fields[1])
           value_test = float(fields[1])

           with open(filename_reference) as f_reference:

               count_reference = 0
               for line_reference in f_reference:
                   count_reference += 1
                   fields_reference = line_reference.split()

                   # we have the -1 because there is the header with executing command
                   # in the reference file
                   if count == count_reference-1:
                       print(fields_reference[1])
                       value_reference = float(fields_reference[1])

                       abs_error = np.abs(value_test - value_reference)
                       rel_error = abs_error/np.abs(value_reference)

                       print ("abs error, rel error =", abs_error, rel_error)

                       check_abs = abs_error < threshold_abs_error
                       check_rel = rel_error < threshold_rel_error

                       assert check_abs or check_rel, \
                              "\n\nAbsolute and relative error of variable number "+str(count)+\
                              " compared to reference too big:"\
                              "\n\nRelative error: "+str(rel_error)+" and treshold: "+str(threshold_rel_error)+\
                              "\n\nAbsolute error: "+str(abs_error)+" and treshold: "+str(threshold_abs_error)

                       print("abs_error =", abs_error)

           f_reference.close()

#               rel_error = 
   
   f.close()
   
if __name__ == "__main__":
  main()
