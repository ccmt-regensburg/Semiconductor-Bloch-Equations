import numpy as np
import os
import pytest

def main():

   filename = "../test.dat"
   filename_reference = "01_1d_model_bandstructure_5_kpoints_and_default_E_field.dat"
   threshold_rel_error = 1.0E-18
   threshold_abs_error = 1.0E-24

   exists = os.path.isfile(filename)
   assert exists

   exists_reference = os.path.isfile(filename_reference)
   assert exists_reference

   with open(filename) as f:
       with open(filename_reference) as f_reference:
           count = 1
           for line in f:
               fields = line.split()
               print(fields[0])
               print(fields[1])
               value_test = float(fields[1])

               count_reference = 1
               for line_reference in f_reference:
                   fields_reference = line_reference.split()

                   # we have the -1 because there is the header with executing command
                   # in the reference file
                   if count == count_reference-1:
                       print(fields_reference[1])
                       value_reference = float(fields_reference[1])
                       abs_error = np.abs(value_test - value_reference)
                       rel_error = abs_error/np.abs(value_reference)
                       print("abs_error =", abs_error)

                   count_reference += 1

               count += 1
#               rel_error = 
   
   f.close()
   
if __name__ == "__main__":
  main()
