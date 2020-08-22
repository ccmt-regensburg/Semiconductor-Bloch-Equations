import multiprocessing
import numpy as np


def main(i):
    print("HELLO WORLD ", i)


if __name__ == "__main__":
    phaselist = np.linspace(0, np.pi, 20)
    for i in range(5):
        p = multiprocessing.Process(target=main, args=(i,))
        p.start()
