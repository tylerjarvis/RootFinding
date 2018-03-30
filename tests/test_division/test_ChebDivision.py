import numpy as np
from groebner.polynomial import Polynomial, MultiCheb, MultiPower
from groebner.DivisionMatrixes.ChebyshevDivision import division_cheb

def getPoly(deg,dim):
    '''
    A helper function for testing. Returns a random upper triangular polynomial of the given dimension and degree.
    power is a boolean indicating whether or not the polynomial should be MultiPower.
    '''
    deg += 1
    ACoeff = np.random.random_sample(deg*np.ones(dim, dtype = int))
    for i,j in np.ndenumerate(ACoeff):
        if np.sum(i) >= deg:
            ACoeff[i] = 0
    return MultiCheb(ACoeff)

def correctZeros(polys, divisor_var, checkNumber = False):
    '''
    A helper function. Takes in polynomials, find their common zeros, and calculates how many of the zeros are correct.
    In this function it asserts that the number of zeros is equal to the product of the degrees, which is only valid if
    the polynomials are random and upper triangular, and that at least 95% of the zeros are correct (so it will pass even
    on bad random runs)
    '''
    zeros = division_cheb(polys, divisor_var = divisor_var)
    assert(zeros != -1)
    if checkNumber:
        expectedNum = np.product([poly.degree for poly in polys])
        assert(len(zeros) == expectedNum)
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

def test_Division_Cheb():
    '''
    The following tests will run division_cheb on relatively small random upper trianguler MultiCheb polynomials.
    The assert statements will be inside of the correctZeros helper function.
    '''
    #Case 1 - Two 2D degree 10 polynomials.
    A = getPoly(10,2)
    B = getPoly(10,2)
    correctZeros([A,B], 0)
    correctZeros([A,B], 1)

    #Case 2 - Two 2D, one degree 5, one degree 7.
    A = getPoly(5,2)
    B = getPoly(7,2)
    correctZeros([A,B], 0)
    correctZeros([A,B], 1)
    
    #Case 3 - Three 3D degree 4 polynomials.
    A = getPoly(4,3)
    B = getPoly(4,3)
    C = getPoly(4,3)
    correctZeros([A,B,C], 0)
    correctZeros([A,B,C], 1)
    correctZeros([A,B,C], 2)
    
    #Case 4 - Three 3D of degrees 3,4 and 5
    A = getPoly(3,3)
    B = getPoly(4,3)
    C = getPoly(5,3)
    correctZeros([A,B,C], 0)
    correctZeros([A,B,C], 1)
    correctZeros([A,B,C], 2)

    #Case 5 - Four 4D degree 2 polynomials.
    A = getPoly(2,4)
    B = getPoly(2,4)
    C = getPoly(2,4)
    D = getPoly(2,4)
    correctZeros([A,B,C,D], 0)
    correctZeros([A,B,C,D], 1)
    correctZeros([A,B,C,D], 2)
    correctZeros([A,B,C,D], 3)