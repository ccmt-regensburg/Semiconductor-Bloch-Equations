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
      count = 1
      for line in f:
          print("Line {}: {}".format(count, line.strip()))
          count += 1
#          rel_error = 
   
   f.close()
   
if __name__ == "__main__":
  main()
