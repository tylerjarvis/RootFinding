import numpy as np
from random_tests import save_tests
np.random.seed(2)

degrees = {}
degrees[2] = np.arange(2,21)
degrees[3] = np.arange(2,21)
degrees[4] = np.arange(2,16)
degrees[5] = np.arange(2,11)
degrees[6] = [2,3,4,5]
degrees[7] = [2,3,4]
degrees[8] = [2,3]
degrees[9] = [2,3]
degrees[10] = [2,3]

N = {}
N[2] = 300
N[3] = 300
N[4] = 300
N[5] = 300
N[6] = 200
N[7] = 200
N[8] = 200
N[9] = 100
N[10] = 100

if __name__ == "__main__":
    from sys import argv
    kind = argv[1]

    for dim in np.arange(2,11):
        save_tests(dim,degrees[dim],N[dim],kind)
