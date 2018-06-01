import numpy as np
from numalgsolve.polynomial import Polynomial, MultiCheb, MultiPower
from numalgsolve.TVBCore import find_degree, mon_combos
from numalgsolve import polyroots as pr
from numalgsolve.utils import InstabilityWarning, arrays
from itertools import product
import warnings

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

def correctZeros(polys, method):
    '''
    A helper function for test_TVB. Takes in polynomials, find their common zeros using TVB, and calculates
    how many of the zeros are correct.
    In this function it asserts that the number of zeros is equal to the product of the degrees, which is only valid if
    the polynomials are random and upper triangular, and that at least 95% of the zeros are correct (so it will pass even
    on bad random runs)
    '''
    zeros = pr.solve(polys, method = method)
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
    The following tests will run TVB on relatively small random upper trianguler MultiPower.
    The assert statements will be inside of the correctZeros helper function.
    '''
    #Case 1 - Two MultiPower 2D degree 10 polynomials.
    A = getPoly(10,2,True)
    B = getPoly(10,2,True)
    correctZeros([A,B], 'mult')

    #Case 2 - Three MultiPower 3D degree 4 polynomials.
    A = getPoly(4,3,True)
    B = getPoly(4,3,True)
    C = getPoly(4,3,True)
    correctZeros([A,B,C], 'mult')

    #Case 3 - Four MultiPower 4D degree 2 polynomials.
    A = getPoly(2,4,True)
    B = getPoly(2,4,True)
    C = getPoly(2,4,True)
    D = getPoly(2,4,True)
    correctZeros([A,B,C,D], 'mult')

    #Case 4 - Two MultiPower 2D, one degree 5 and one degree 7
    A = getPoly(5,2,True)
    B = getPoly(7,2,True)
    correctZeros([A,B], 'mult')

    #Case 5 - Three MultiPower 3D of degrees 3,4 and 5
    A = getPoly(3,3,True)
    B = getPoly(4,3,True)
    C = getPoly(5,3,True)
    correctZeros([A,B,C], 'mult')

def test_TVB_cheb_roots():
    '''
    The following tests will run TVB on relatively small random upper trianguler MultiCheb.
    The assert statements will be inside of the correctZeros helper function.
    '''
    #Case 1 - Two MultiCheb 2D degree 10 polynomials.
    A = getPoly(10,2,False)
    B = getPoly(10,2,False)
    correctZeros([A,B], 'mult')

    #Case 2 - Three MultiCheb 3D degree 4 polynomials.
    A = getPoly(4,3,False)
    B = getPoly(4,3,False)
    C = getPoly(4,3,False)
    correctZeros([A,B,C], 'mult')

    #Case 3 - Four MultiCheb 4D degree 2 polynomials.
    A = getPoly(2,4,False)
    B = getPoly(2,4,False)
    C = getPoly(2,4,False)
    D = getPoly(2,4,False)
    correctZeros([A,B,C,D], 'mult')

    #Case 4 - Two MultiCheb 2D, one degree 5 and one degree 7
    A = getPoly(5,2,False)
    B = getPoly(7,2,False)
    correctZeros([A,B], 'mult')

    #Case 5 - Three MultiCheb 3D of degrees 3,4 and 5
    A = getPoly(3,3,False)
    B = getPoly(4,3,False)
    C = getPoly(5,3,False)
    correctZeros([A,B,C], 'mult')

def test_div_power_roots():
    '''
    The following tests will run TVB on relatively small random upper trianguler MultiPower.
    The assert statements will be inside of the correctZeros helper function.
    '''
    #Case 1 - Two MultiPower 2D degree 10 polynomials.
    A = getPoly(10,2,True)
    B = getPoly(10,2,True)
    correctZeros([A,B], 'div')

    #Case 2 - Three MultiPower 3D degree 4 polynomials.
    A = getPoly(4,3,True)
    B = getPoly(4,3,True)
    C = getPoly(4,3,True)
    correctZeros([A,B,C], 'div')

    #Case 3 - Four MultiPower 4D degree 2 polynomials.
    A = getPoly(2,4,True)
    B = getPoly(2,4,True)
    C = getPoly(2,4,True)
    D = getPoly(2,4,True)
    correctZeros([A,B,C,D], 'div')

    #Case 4 - Two MultiPower 2D, one degree 5 and one degree 7
    A = getPoly(5,2,True)
    B = getPoly(7,2,True)
    correctZeros([A,B], 'div')

    #Case 5 - Three MultiPower 3D of degrees 3,4 and 5
    A = getPoly(3,3,True)
    B = getPoly(4,3,True)
    C = getPoly(5,3,True)
    correctZeros([A,B,C], 'div')

def test_div_cheb_roots():
    '''
    The following tests will run TVB on relatively small random upper trianguler MultiCheb.
    The assert statements will be inside of the correctZeros helper function.
    '''
    #Case 1 - Two MultiCheb 2D degree 10 polynomials.
    A = getPoly(10,2,False)
    B = getPoly(10,2,False)
    correctZeros([A,B], 'div')

    #Case 2 - Three MultiCheb 3D degree 4 polynomials.
    A = getPoly(4,3,False)
    B = getPoly(4,3,False)
    C = getPoly(4,3,False)
    correctZeros([A,B,C], 'div')

    #Case 3 - Four MultiCheb 4D degree 2 polynomials.
    A = getPoly(2,4,False)
    B = getPoly(2,4,False)
    C = getPoly(2,4,False)
    D = getPoly(2,4,False)
    correctZeros([A,B,C,D], 'div')

    #Case 4 - Two MultiCheb 2D, one degree 5 and one degree 7
    A = getPoly(5,2,False)
    B = getPoly(7,2,False)
    correctZeros([A,B], 'div')

    #Case 5 - Three MultiCheb 3D of degrees 3,4 and 5
    A = getPoly(3,3,False)
    B = getPoly(4,3,False)
    C = getPoly(5,3,False)
    correctZeros([A,B,C], 'div')
