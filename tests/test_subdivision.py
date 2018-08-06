import unittest
import numpy as np
from numalgsolve.polynomial import Polynomial, MultiCheb, MultiPower
from numalgsolve import subdivision as subdiv
from itertools import product

def getPoly(deg,dim,power):
    '''
    A helper function for testing. Returns a random upper triangular polynomial of the given dimension and degree.
    power is a boolean indicating whether or not the polynomial should be MultiPower.
    '''
    deg += 1
    ACoeff = np.random.random_sample(deg*np.ones(dim, dtype = int))
    for i,j in np.ndenumerate(ACoeff):
        if np.sum(i) >= deg:
            ACoeff[i] = 0
    if power:
        return MultiPower(ACoeff)
    else:
        return MultiCheb(ACoeff)

def correctZeros(polys, a, b):
    '''
    A helper function for test_subdivision_solve. Takes in polynomials, find their common zeros using subdivision, and calculates
    how many of the zeros are correct.
    In this function it ignores the number of zeros since it only searches a specific interval. It asserts that at least 95%
    of the zeros are correct (so it will pass even on bad random runs)
    '''
    zeros = subdiv.solve(polys, a, b)
    correct = 0
    outOfRange = 0
    for zero in zeros:
        good = True
        for poly in polys:
            if not np.isclose(0, poly(zero), atol = 1.e-3):
                good = False
                if (np.abs(zero) > 1).any():
                    outOfRange += 1
                break
        if good:
            correct += 1
    if len(zeros) == outOfRange:
        raise Exception("No zeros found")
    else:
        assert(100*correct/(len(zeros)-outOfRange) > 95)

def test_subdivision_solve_polys():
    '''
    The following tests will run subdivision.solve on relatively small random upper trianguler MultiPower.
    The assert statements will be inside of the correctZeros helper function.
    The fit occurs on [-1,1]X[-1,1]X..., so no transform is needed.
    '''
    #Case 1 - Two MultiPower 2D degree 10 polynomials.
    #choose a seed that has a zero like 1,3,7,8,12,20,21,22,22,27,38,41,42,43,46,51,54,55,57,60,65,67,68,69,73,74,78,80,81,84,86,90,95
    np.random.seed(1)
    a = -np.ones(2);b = np.ones(2)
    A = getPoly(10,2,True)
    B = getPoly(10,2,True)
    correctZeros([A,B], a, b)

    #Case 2 - Three MultiPower 3D degree 4 polynomials.
    #choose a seed that has a zero like 1,23,27,29,39,43,44,46,51,53,54,68,71,72,93
    np.random.seed(1)
    a = -np.ones(3);b = np.ones(3)
    A = getPoly(4,3,True)
    B = getPoly(4,3,True)
    C = getPoly(4,3,True)
    correctZeros([A,B,C], a, b)

    #Case 3 - Four MultiPower 4D degree 2 polynomials.
    #choose a seed that has a zero like 21,43,65,72,83
    np.random.seed(21)
    a = -np.ones(4);b = np.ones(4)
    A = getPoly(2,4,True)
    B = getPoly(2,4,True)
    C = getPoly(2,4,True)
    D = getPoly(2,4,True)
    correctZeros([A,B,C,D], a, b)

    #Case 4 - Two MultiPower 2D, one degree 20 and one degree 28
    #choose a seed that has a zero like 0,1,2,3,4,5,6,7,8,9,10
    np.random.seed(0)
    a = -np.ones(2);b = np.ones(2)
    A = getPoly(20,2,True)
    B = getPoly(28,2,True)
    correctZeros([A,B], a, b)

    #Case 5 - Three MultiPower 3D of degrees 3,4 and 5
    #choose a seed that has a zero like 1,3,5,11,13,16,24,28,31,32,33,41,42
    np.random.seed(1)
    a = -np.ones(3);b = np.ones(3)
    A = getPoly(3,3,True)
    B = getPoly(4,3,True)
    C = getPoly(5,3,True)
    correctZeros([A,B,C], a, b)

def test_subdivision_solve_1d():
    #Case 6 - One MultiPower 1D of degrees 10
    #choose a seed that has a zero like ?
    np.random.seed(1)
    a = -np.ones(1);b = np.ones(1)
    A = getPoly(20,1,False)
    correctZeros([A], a, b)

def test_subdivision_sine():
    '''
    Test case using basic sine function to put zeros on the coordinates.
    The expected zeros are
    (0,0), (0,1), (0,2), (0,3),
    (1,0), (1,1), (1,2), (1,3),
    (2,0), (2,1), (2,2), (2,3),
    (3,0), (3,1), (3,2), (3,3),

    '''
    f = lambda x: np.sin(np.pi*x[:,1])
    g = lambda x: np.sin(np.pi*(x[:,0]+x[:,1]))
    a = -0.511*np.ones(2)
    b = 3.511*np.ones(2)

    zeros = subdiv.solve([f, g], a, b)
    zeros = np.array(sorted(list(zeros), key=lambda x: 10*x[0] + x[1]))

    assert len(zeros) == 16

    X,Y = np.meshgrid(range(4),range(4),indexing='ij')
    expected_zeros = np.column_stack([X.flatten(), Y.flatten()])
    assert np.allclose(expected_zeros, zeros, atol=1e-4)


@unittest.skip("high degree polynomial explodes outside of the unit region")
def test_subdivision_solve_with_transform():
    '''
    The following tests will run subdivision.solve on relatively small random upper trianguler MultiPower.
    The assert statements will be inside of the correctZeros helper function.
    The fit occurs on [-2,2]X[-2,2]X..., so a transform is needed.
    '''
    #Case 1 - Two MultiPower 2D degree 10 polynomials.
    #choose a seed that has a zero like 1,3,7,8,12,20,21,22,22,27,38,41,42,43,46,51,54,55,57,60,65,67,68,69,73,74,78,80,81,84,86,90,95
    np.random.seed(1)
    a = -2*np.ones(2);b = 2*np.ones(2)
    A = getPoly(10,2,True)
    B = getPoly(10,2,True)
    correctZeros([A,B], a, b)

    #Case 2 - Three MultiPower 3D degree 4 polynomials.
    #choose a seed that has a zero like 1,23,27,29,39,43,44,46,51,53,54,68,71,72,93
    np.random.seed(1)
    a = -2*np.ones(3);b = 2*np.ones(3)
    A = getPoly(4,3,True)
    B = getPoly(4,3,True)
    C = getPoly(4,3,True)
    correctZeros([A,B,C], a, b)

    #Case 3 - Four MultiPower 4D degree 2 polynomials.
    #choose a seed that has a zero like 21,43,65,72,83
    np.random.seed(21)
    a = -2*np.ones(4);b = 2*np.ones(4)
    A = getPoly(2,4,True)
    B = getPoly(2,4,True)
    C = getPoly(2,4,True)
    D = getPoly(2,4,True)
    correctZeros([A,B,C,D], a, b)

    #Case 4 - Two MultiPower 2D, one degree 20 and one degree 28
    #choose a seed that has a zero like 0,1,2,3,4,5,6,7,8,9,10
    np.random.seed(0)
    a = -2*np.ones(2);b = 2*np.ones(2)
    A = getPoly(20,2,True)
    B = getPoly(28,2,True)
    correctZeros([A,B], a, b)

    #Case 5 - Three MultiPower 3D of degrees 3,4 and 5
    #choose a seed that has a zero like 1,3,5,11,13,16,24,28,31,32,33,41,42
    np.random.seed(1)
    a = -2*np.ones(3);b = 2*np.ones(3)
    A = getPoly(3,3,True)
    B = getPoly(4,3,True)
    C = getPoly(5,3,True)
    correctZeros([A,B,C], a, b)

@unittest.skip("high degree polynomial explodes outside of the unit region")
def test_subdivision_solve_with_transform_1d():
    #Case 6 - One MultiPower 1D of degrees 10
    #choose a seed that has a zero like ?
    np.random.seed(0)
    a = -2*np.ones(1);b = 2*np.ones(1)
    A = getPoly(10,1,False)
    correctZeros([A], a, b)

def test_good_zeros_nd():
    '''
    The good zeros function should remove zeros with imaginary part or outside
    the range [-1,1]X[-1,1]X...
    '''

    zeros = np.array([
    [0.9+0j, 0.9+0j],   #good
    [0+0j, 0.1+0j],     #good
    [-1.1+0j, 0.1+0j],  #out of range
    [0.1+0.1j, 0.1+0j], #imaginary
    ])

    assert np.all(subdiv.good_zeros_nd(zeros) == zeros[:2])

    zeros = np.array([
    [0.9+0j, 0.9+0j, -0.1+0j],    #good
    [0+0j, 0.1+0j, 0.2+1e-14j],   #good
    [-1.1+0j, 0.1+0j, 0+0j],      #out of range
    [0.1+0.1j, 0.1+0j, 0.8-0.1j], #imaginary
    ])

    assert np.all(subdiv.good_zeros_nd(zeros) == zeros[:2])
