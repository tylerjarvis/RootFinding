import numpy as np
from groebner.polynomial import Polynomial, MultiCheb, MultiPower
from groebner.DivisionMatrixes.PowerDivision import division_power

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

def correctZeros(polys, checkNumber = True):
    '''
    A helper function. Takes in polynomials, find their common zeros, and calculates how many of the zeros are correct.
    In this function it asserts that the number of zeros is equal to the product of the degrees, which is only valid if
    the polynomials are random and upper triangular, and that at least 95% of the zeros are correct (so it will pass even
    on bad random runs)
    '''
    zeros = division_power(polys)
    assert(zeros != -1)
    if checkNumber:
        expectedNum = np.product([poly.degree for poly in polys])
        assert(len(zeros) == expectedNum)
    correct = 0
    outOfRange = 0
    for zero in zeros:
        good = True
        for poly in polys:
            if not np.isclose(0, poly.evaluate_at(zero), atol = 1.e-3):
                good = False
                if (np.abs(zero) > 1).any():
                    outOfRange += 1
                break
        if good:
            correct += 1
    assert(100*correct/(len(zeros)-outOfRange) > 95)

def test_Division_Power():
    '''
    The following tests will run division_power on relatively small random upper trianguler MultiPower polynomials.
    The assert statements will be inside of the correctZeros helper function.
    '''
    #Case 1 - Two MultiPower 2D degree 10 polynomials.
    A = getPoly(10,2,True)
    B = getPoly(10,2,True)
    correctZeros([A,B])

    #Case 2 - Two MultiPower, one degree 5, one degree 7.
    A = getPoly(5,2,True)
    B = getPoly(7,2,True)
    correctZeros([A,B])