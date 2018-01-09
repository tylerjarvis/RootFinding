import numpy as np
from groebner.polynomial import Polynomial, MultiCheb, MultiPower
from groebner.DivisionMatrixes.OneDimension import one_dimensional_solve

def getPoly(deg, power):
    '''
    A helper function for testing. Returns a random 1D polynomial of the given degree.
    power is a boolean indicating whether or not the polynomial should be MultiPower.
    '''
    coeff = np.random.random_sample(deg+1)
    if power:
        return MultiPower(coeff)
    else:
        return MultiCheb(coeff)

def correctZeros(poly, method, checkNumber = True):
    '''
    A helper function. Takes in a polynomial, find the zeros, and calculates how many of the zeros are correct.
    In this function it asserts that the number of zeros is equal to the product of the degrees, which is only valid if
    the polynomial is random, and that at least 95% of the zeros are correct (so it will pass even
    on bad random runs)
    '''
    zeros = one_dimensional_solve(poly, method = method)
    if checkNumber:
        assert(len(zeros) == poly.degree)
    correct = 0
    outOfRange = 0
    for zero in zeros:
        good = True
        if not np.isclose(0, poly.evaluate_at([zero]), atol = 1.e-3):
            good = False
        if good:
            correct += 1
    assert(100*correct/(len(zeros)) > 95)

def test_Division_Cheb():
    '''
    The following tests will run division_cheb on relatively small random upper trianguler MultiCheb polynomials.
    The assert statements will be inside of the correctZeros helper function.
    '''
    #Case 1 - MultiPower degree 10. Multiplication and Division Matrixes.
    poly = getPoly(10,True)
    correctZeros(poly, 'M')
    correctZeros(poly, 'D')

    #Case 2 - MultiCheb degree 10. Multiplication and Division Matrixes.
    poly = getPoly(10,False)
    correctZeros(poly, 'M')
    correctZeros(poly, 'D')

    #Case 3 - MultiPower degree 50. Multiplication and Division Matrixes.
    poly = getPoly(50,True)
    correctZeros(poly, 'M')
    correctZeros(poly, 'D')

    #Case 4 - MultiCheb degree 50. Multiplication and Division Matrixes.
    poly = getPoly(50,False)
    correctZeros(poly, 'M')
    correctZeros(poly, 'D')

    #Case 5 - MultiPower degree 100. Multiplication and Division Matrixes.
    poly = getPoly(100,True)
    correctZeros(poly, 'M')
    correctZeros(poly, 'D')

    #Case 6 - MultiCheb degree 100. Multiplication and Division Matrixes.
    poly = getPoly(100,False)
    correctZeros(poly, 'M')
    correctZeros(poly, 'D')

    
    