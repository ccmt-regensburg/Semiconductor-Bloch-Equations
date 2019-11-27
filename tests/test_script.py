import numpy as np
import os
import pytest

def main():

   exists = os.path.isfile("../test.dat")

   assert exists

   if exists: 

      with open("../test.dat") as f:
         line = f.readline()
         cnt = 1
         while line:
             print("Line {}: {}".format(cnt, line.strip()))
             line = f.readline()
             cnt += 1
   
      f.close()
   
      print("check")

if __name__ == "__main__":
  main()
