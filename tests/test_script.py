import numpy as np
import os

def main():

   with open("../filename_has_to_be_set") as f:
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
