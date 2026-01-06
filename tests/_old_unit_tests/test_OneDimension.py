import numpy as np
from yroots.polynomial import Polynomial, MultiCheb, MultiPower
from yroots.OneDimension import solve

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

def correctZeros(poly,  MSmatrix, eigvals = True, checkNumber = True):
    '''
    A helper function. Takes in a polynomial, find the zeros, and calculates how many of the zeros are correct.
    In this function it asserts that the number of zeros is equal to the product of the degrees, which is only valid if
    the polynomial is random, and that at least 95% of the zeros are correct (so it will pass even
    on bad random runs)
    '''
    zeros = solve(poly,  MSmatrix=MSmatrix, eigvals=eigvals)
    if checkNumber:
        assert(len(zeros) == poly.degree)
    correct = 0
    outOfRange = 0
    for zero in zeros:
        good = True
        if not np.isclose(0, poly([zero]), atol = 1.e-3):
            good = False
        if good:
            correct += 1
    assert(100*correct/(len(zeros)) > 95)

def test_OneD_power_eigenvalues():
    '''
    The following tests will solve 1D polynomials using eigenvalues of
    multiplication, multiplication rotated, and division matrices.
    The assert statements will be inside of the correctZeros helper function.
    '''
    np.random.seed(3902)

    #Case 1 - MultiPower degree 10
    poly = getPoly(10,True)
    correctZeros(poly, 0)
    correctZeros(poly, -1)

    #Case 2 - MultiPower degree 50
    poly = getPoly(50,True)
    correctZeros(poly, 0)
    correctZeros(poly, -1)

    #Case 3 - MultiPower degree 100
    poly = getPoly(100,True)
    correctZeros(poly, 0)
    correctZeros(poly, -1)

def test_OneD_power_eigenvectors():
    '''
    The following tests will solve 1D polynomials using eigenvectors of
    multiplication, multiplication rotated, and division matrices.
    The assert statements will be inside of the correctZeros helper function.
    '''
    np.random.seed(3902)

    #Case 1 - MultiPower degree 10
    poly = getPoly(10,True)
    correctZeros(poly, 0, eigvals=False)
    correctZeros(poly, -1, eigvals=False)

    #Case 2 - MultiPower degree 50
    poly = getPoly(50,True)
    correctZeros(poly, 0, eigvals=False)
    correctZeros(poly, -1, eigvals=False)

    #Case 3 - MultiPower degree 100
    poly = getPoly(100,True)
    correctZeros(poly, 0, eigvals=False)
    correctZeros(poly, -1, eigvals=False)

def test_OneD_cheb_eigenvalues():
    '''
    The following tests will solve 1D cheb polynomials using eigenvalues of
    multiplication, multiplication rotated, and division matrices.
    The assert statements will be inside of the correctZeros helper function.
    '''
    np.random.seed(45)

    #Case 1 - MultiCheb degree 10
    poly = getPoly(10,False)
    correctZeros(poly, 0)
    correctZeros(poly, -1)

    #Case 2 - MultiCheb degree 50
    poly = getPoly(50,False)
    correctZeros(poly, 0)
    correctZeros(poly, -1)

    #Case 3 - MultiCheb degree 100
    poly = getPoly(100,False)
    correctZeros(poly, 0)
    correctZeros(poly, -1)

def test_OneD_cheb_eigenvectors():
    '''
    The following tests will solve 1D cheb polynomials using eigenvectors of
    multiplication, multiplication rotated, and division matrices.
    The assert statements will be inside of the correctZeros helper function.
    '''
    np.random.seed(45)

    #Case 1 - MultiCheb degree 10
    poly = getPoly(10,False)
    correctZeros(poly, 0, eigvals=False)
    correctZeros(poly, -1, eigvals=False)

    #Case 2 - MultiCheb degree 50
    poly = getPoly(50,False)
    correctZeros(poly, 0, eigvals=False)
    correctZeros(poly, -1, eigvals=False)

    #Case 3 - MultiCheb degree 100
    poly = getPoly(100,False)
    correctZeros(poly, 0, eigvals=False)
    correctZeros(poly, -1, eigvals=False)
