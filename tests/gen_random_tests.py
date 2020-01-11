import numpy as np
from random_tests import save_tests

if __name__ == "__main__":
    # dimension 2
    degrees = np.arange(16)
    save_tests(2,degrees,20)
