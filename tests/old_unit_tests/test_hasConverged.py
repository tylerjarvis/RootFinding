# test_hasConverged.py

import numpy as np
from yroots.ChebyshevApproximator import hasConverged

def test_has_converged():
    # Test case 1: Large tolerance, converged
    coeff = np.array([0.1, 0.0000001, -0.3])
    coeff2 = np.array([0.11, -0.01, -0.31])
    tol = 0.01001
    assert hasConverged(coeff, coeff2, tol)

    # Test case 2: Small tolerance, not converged
    coeff = np.array([0.1, 0.2, 0.3])
    coeff2 = np.array([0.11, 0.21, 0.31])
    tol = 0.001
    assert not hasConverged(coeff, coeff2, tol)

    # Test case 3: Zero tolerance, not converged
    coeff = np.array([0.1, 0.2, 0.3])
    coeff2 = np.array([0.1000000001, 0.2, 0.3])
    tol = 0
    assert not hasConverged(coeff, coeff2, tol)

    # Test case 4: Large number of inputs, converged
    coeff = np.array([i * .1 for i in range(10)])
    coeff2 = np.array([(i * .1) + .0001 for i in range(10)])
    tol = .001
    assert hasConverged(coeff, coeff2, tol)

if __name__ == '__main__':
    test_has_converged()
    print("All tests passed!")
