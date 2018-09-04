import numpy as np
from numalgsolve.polynomial import Polynomial, MultiCheb, MultiPower
from numalgsolve.MacaulayReduce import find_degree, mon_combos
from numalgsolve import TVBMethod as tvb
from numalgsolve.utils import InstabilityWarning, arrays
from numalgsolve.Multiplication import create_matrix
from itertools import product
import unittest
import warnings

def test_TVB_paper_example():

    #Power form of the polys
    p1 = MultiPower(np.array([[1, -4, 0],[0, 3, 0],[1, 0, 0]])) #y^2 + 3xy - 4x +1
    p2 = MultiPower(np.array([[3, 0, -2],[6, -6, 0],[0, 0, 0]])) #-6xy -2x^2 + 6y +3

    #Cheb form of the polys
    c1 = MultiCheb(np.array([[2, 0, -1],[6, -6, 0], [0, 0, 0]])) #p1 in Cheb form
    c2 = MultiCheb(np.array([[1.5, -4, 0],[0, 3, 0], [.5, 0, 0]])) #p2 in Cheb form

    right_number_of_roots = 4

    power_roots = tvb.solve([p1, p2], verbose=True)
    assert len(power_roots) == right_number_of_roots
    for root in power_roots:
        assert np.isclose(0, p1(root), atol = 1.e-8)
        assert np.isclose(0, p2(root), atol = 1.e-8)

    cheb_roots = tvb.solve([c1, c2], verbose=True)
    assert len(cheb_roots) == right_number_of_roots
    for root in cheb_roots:
        assert np.isclose(0, c1(root), atol = 1.e-8)
        assert np.isclose(0, c1(root), atol = 1.e-8)

def getPoly(deg,dim,power):
    '''
    A helper function for testing. Returns a random upper triangular polynomial
    of the given dimension and degree.
    power is a boolean indicating whether or not the polynomial should be
    MultiPower.
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

def correctZeros(polys):
    '''
    A helper function for polyroots tests. Takes in polynomials, find their common zeros using polyroots, and calculates
    how many of the zeros are correct.
    In this function it asserts that the number of zeros is equal to the product of the degrees, which is only valid if
    the polynomials are random and upper triangular, and that at least 95% of the zeros are correct (so it will pass even
    on bad random runs)
    '''
    zeros = tvb.solve(polys, verbose=False)
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
    assert(100*correct/(len(zeros)-outOfRange) > 95)

def test_TVB_power_roots():
    '''
    The following tests will run polyroots on relatively small random upper trianguler MultiPower.
    The assert statements will be inside of the correctZeros helper function.
    '''

    np.random.seed(423)

    #Case 1 - Two MultiPower 2D degree 10 polynomials.
    A = getPoly(10,2,True)
    B = getPoly(10,2,True)
    correctZeros([A,B])
    print('Case2')
    #Case 2 - Three MultiPower 3D degree 4 polynomials.
    A = getPoly(4,3,True)
    B = getPoly(4,3,True)
    C = getPoly(4,3,True)
    correctZeros([A,B,C])

    #Case 3 - Four MultiPower 4D degree 2 polynomials.
    A = getPoly(2,4,True)
    B = getPoly(2,4,True)
    C = getPoly(2,4,True)
    D = getPoly(2,4,True)
    correctZeros([A,B,C,D])

    #Case 4 - Two MultiPower 2D, one degree 5 and one degree 7
    A = getPoly(5,2,True)
    B = getPoly(7,2,True)
    correctZeros([A,B])

    #Case 5 - Three MultiPower 3D of degrees 3,4 and 5
    A = getPoly(3,3,True)
    B = getPoly(4,3,True)
    C = getPoly(5,3,True)
    correctZeros([A,B,C])

def test_TVB_cheb_roots():
    '''
    The following tests will run polyroots on relatively small random upper trianguler MultiCheb.
    The assert statements will be inside of the correctZeros helper function.
    '''

    np.random.seed(59)

    #Case 1 - Two MultiCheb 2D degree 10 polynomials.
    A = getPoly(10,2,False)
    B = getPoly(10,2,False)
    correctZeros([A,B])

    #Case 2 - Three MultiCheb 3D degree 4 polynomials.
    A = getPoly(4,3,False)
    B = getPoly(4,3,False)
    C = getPoly(4,3,False)
    correctZeros([A,B,C])

    #Case 3 - Four MultiCheb 4D degree 2 polynomials.
    A = getPoly(2,4,False)
    B = getPoly(2,4,False)
    C = getPoly(2,4,False)
    D = getPoly(2,4,False)
    correctZeros([A,B,C,D])

    #Case 4 - Two MultiCheb 2D, one degree 5 and one degree 7
    A = getPoly(5,2,False)
    B = getPoly(7,2,False)
    correctZeros([A,B])

    #Case 5 - Three MultiCheb 3D of degrees 3,4 and 5
    A = getPoly(3,3,False)
    B = getPoly(4,3,False)
    C = getPoly(5,3,False)
    correctZeros([A,B,C])
