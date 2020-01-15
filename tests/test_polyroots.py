import numpy as np
from yroots.polynomial import Polynomial, MultiCheb, MultiPower, getPoly
from yroots.MacaulayReduce import find_degree, mon_combos
from yroots import polyroots as pr
from yroots.utils import InstabilityWarning, arrays
from yroots.Multiplication import create_matrix
from itertools import product
import unittest
import warnings
import yroots.subdivision as sbd

def test_paper_example():

    #Power form of the polys
    p1 = MultiPower(np.array([[1, -4, 0],[0, 3, 0],[1, 0, 0]])) #y^2 + 3xy - 4x +1
    p2 = MultiPower(np.array([[3, 0, -2],[6, -6, 0],[0, 0, 0]])) #-6xy -2x^2 + 6y +3

    #Cheb form of the polys
    c1 = MultiCheb(np.array([[2, 0, -1],[6, -6, 0], [0, 0, 0]])) #p1 in Cheb form
    c2 = MultiCheb(np.array([[1.5, -4, 0],[0, 3, 0], [.5, 0, 0]])) #p2 in Cheb form

    right_number_of_roots = 4

    #~ ~ ~ Power Form, Mx Matrix ~ ~ ~
    power_mult_roots = pr.solve([p1, p2], MSmatrix = 1)
    assert len(power_mult_roots) == right_number_of_roots
    for root in power_mult_roots:
        assert np.isclose(0, p1(root), atol = 1.e-8)
        assert np.isclose(0, p2(root), atol = 1.e-8)

    #~ ~ ~ Cheb Form, Mx Matrix ~ ~ ~
    cheb_mult_roots = pr.solve([c1, c2], MSmatrix = 1)
    assert len(cheb_mult_roots) == right_number_of_roots
    for root in cheb_mult_roots:
        assert np.isclose(0, c1(root), atol = 1.e-8)
        assert np.isclose(0, c1(root), atol = 1.e-8)

    #~ ~ ~ Power Form, My Matrix ~ ~ ~
    power_multR_roots = pr.solve([p1, p2], MSmatrix = 2)
    assert len(power_multR_roots) == right_number_of_roots
    for root in power_multR_roots:
        assert np.isclose(0, p1(root), atol = 1.e-8)
        assert np.isclose(0, p2(root), atol = 1.e-8)

    #~ ~ ~ Cheb Form, My Matrix ~ ~ ~
    cheb_multR_roots = pr.solve([c1, c2], MSmatrix = 2)
    assert len(cheb_multR_roots) == right_number_of_roots
    for root in cheb_multR_roots:
        assert np.isclose(0, c1(root), atol = 1.e-8)
        assert np.isclose(0, c1(root), atol = 1.e-8)

    #~ ~ ~ Power Form, Pseudorandom Multiplication Matrix ~ ~ ~
    power_multrand_roots = pr.solve([p1, p2],MSmatrix = 0)
    assert len(power_multrand_roots) == right_number_of_roots
    for root in power_multrand_roots:
        assert np.isclose(0, p1(root), atol = 1.e-8)
        assert np.isclose(0, p2(root), atol = 1.e-8)

    #~ ~ ~ Cheb Form, Pseudorandom Multiplication Matrix ~ ~ ~
    cheb_multrand_roots = pr.solve([c1, c2], MSmatrix = 0)
    assert len(cheb_multrand_roots) == right_number_of_roots
    for root in cheb_multrand_roots:
        assert np.isclose(0, c1(root), atol = 1.e-8)
        assert np.isclose(0, c1(root), atol = 1.e-8)

    #~ ~ ~ Power Form, Division Matrix ~ ~ ~
    power_div_roots = pr.solve([p1, p2], MSmatrix = -1)
    assert len(power_div_roots)== right_number_of_roots
    for root in power_div_roots:
        assert np.isclose(0, p1(root), atol = 1.e-8)
        assert np.isclose(0, p2(root), atol = 1.e-8)

    #~ ~ ~ Cheb Form, Division Matrix ~ ~ ~
    cheb_div_roots = pr.solve([c1, c2], MSmatrix = -1)
    assert len(cheb_div_roots) == right_number_of_roots
    for root in cheb_div_roots:
        assert np.isclose(0, c1(root), atol = 1.e-8)
        assert np.isclose(0, c2(root), atol = 1.e-8)

def correctZeros(polys, MSmatrix):
    '''
    A helper function for polyroots tests. Takes in polynomials, find their common zeros using polyroots, and calculates
    how many of the zeros are correct.
    In this function it asserts that the number of zeros is equal to the product of the degrees, which is only valid if
    the polynomials are random and upper triangular, and that at least 95% of the zeros are correct (so it will pass even
    on bad random runs)
    '''
    zeros = pr.solve(polys, MSmatrix = MSmatrix)
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

def test_power_roots_mult():
    '''
    The following tests will run polyroots on relatively small random upper trianguler MultiPower.
    The assert statements will be inside of the correctZeros helper function.
    '''

    np.random.seed(423)

    #Case 1 - Two MultiPower 2D degree 10 polynomials.
    A = getPoly(10,2,True)
    B = getPoly(10,2,True)
    correctZeros([A,B], 1)
    correctZeros([A,B], 2)

    #Case 2 - Three MultiPower 3D degree 4 polynomials.
    A = getPoly(4,3,True)
    B = getPoly(4,3,True)
    C = getPoly(4,3,True)
    correctZeros([A,B,C], 1)
    correctZeros([A,B,C], 2)
    correctZeros([A,B,C], 3)

    #Case 3 - Four MultiPower 4D degree 2 polynomials.
    A = getPoly(2,4,True)
    B = getPoly(2,4,True)
    C = getPoly(2,4,True)
    D = getPoly(2,4,True)
    correctZeros([A,B,C,D], 1)
    correctZeros([A,B,C,D], 2)
    correctZeros([A,B,C,D], 3)
    correctZeros([A,B,C,D], 4)

    #Case 4 - Two MultiPower 2D, one degree 5 and one degree 7
    A = getPoly(5,2,True)
    B = getPoly(7,2,True)
    correctZeros([A,B], 1)
    correctZeros([A,B], 2)

    #Case 5 - Three MultiPower 3D of degrees 3,4 and 5
    A = getPoly(3,3,True)
    B = getPoly(4,3,True)
    C = getPoly(5,3,True)
    correctZeros([A,B,C], 1)
    correctZeros([A,B,C], 2)
    correctZeros([A,B,C], 3)

def test_cheb_roots_mult():
    '''
    The following tests will run polyroots on relatively small random upper trianguler MultiCheb.
    The assert statements will be inside of the correctZeros helper function.
    '''

    np.random.seed(59)

    #Case 1 - Two MultiCheb 2D degree 10 polynomials.
    A = getPoly(10,2,False)
    B = getPoly(10,2,False)
    correctZeros([A,B], 1)
    correctZeros([A,B], 2)

    #Case 2 - Three MultiCheb 3D degree 4 polynomials.
    A = getPoly(4,3,False)
    B = getPoly(4,3,False)
    C = getPoly(4,3,False)
    correctZeros([A,B,C], 1)
    correctZeros([A,B,C], 2)
    correctZeros([A,B,C], 3)

    #Case 3 - Four MultiCheb 4D degree 2 polynomials.
    A = getPoly(2,4,False)
    B = getPoly(2,4,False)
    C = getPoly(2,4,False)
    D = getPoly(2,4,False)
    correctZeros([A,B,C,D], 1)
    correctZeros([A,B,C,D], 2)
    correctZeros([A,B,C,D], 3)
    correctZeros([A,B,C,D], 4)

    #Case 4 - Two MultiCheb 2D, one degree 5 and one degree 7
    A = getPoly(5,2,False)
    B = getPoly(7,2,False)
    correctZeros([A,B], 1)
    correctZeros([A,B], 2)

    #Case 5 - Three MultiCheb 3D of degrees 3,4 and 5
    A = getPoly(3,3,False)
    B = getPoly(4,3,False)
    C = getPoly(5,3,False)
    correctZeros([A,B,C], 1)
    correctZeros([A,B,C], 2)
    correctZeros([A,B,C], 3)

def test_power_roots_multrand():
    '''
    The following tests will run polyroots on relatively small random upper trianguler MultiPower.
    The assert statements will be inside of the correctZeros helper function.
    '''

    np.random.seed(423)

    #Case 1 - Two MultiPower 2D degree 10 polynomials.
    A = getPoly(10,2,True)
    B = getPoly(10,2,True)
    correctZeros([A,B], 0)

    #Case 2 - Three MultiPower 3D degree 4 polynomials.
    A = getPoly(4,3,True)
    B = getPoly(4,3,True)
    C = getPoly(4,3,True)
    correctZeros([A,B,C], 0)

    #Case 3 - Four MultiPower 4D degree 2 polynomials.
    A = getPoly(2,4,True)
    B = getPoly(2,4,True)
    C = getPoly(2,4,True)
    D = getPoly(2,4,True)
    correctZeros([A,B,C,D], 0)

    #Case 4 - Two MultiPower 2D, one degree 5 and one degree 7
    A = getPoly(5,2,True)
    B = getPoly(7,2,True)
    correctZeros([A,B], 0)

    #Case 5 - Three MultiPower 3D of degrees 3,4 and 5
    A = getPoly(3,3,True)
    B = getPoly(4,3,True)
    C = getPoly(5,3,True)
    correctZeros([A,B,C], 0)

def test_cheb_roots_multrand():
    '''
    The following tests will run polyroots on relatively small random upper trianguler MultiCheb.
    The assert statements will be inside of the correctZeros helper function.
    '''

    np.random.seed(590)

    #Case 1 - Two MultiCheb 2D degree 10 polynomials.
    A = getPoly(10,2,False)
    B = getPoly(10,2,False)
    correctZeros([A,B], 0)

    #Case 2 - Three MultiCheb 3D degree 4 polynomials.
    A = getPoly(4,3,False)
    B = getPoly(4,3,False)
    C = getPoly(4,3,False)
    correctZeros([A,B,C], 0)

    #Case 3 - Four MultiCheb 4D degree 2 polynomials.
    A = getPoly(2,4,False)
    B = getPoly(2,4,False)
    C = getPoly(2,4,False)
    D = getPoly(2,4,False)
    correctZeros([A,B,C,D], 0)

    #Case 4 - Two MultiCheb 2D, one degree 5 and one degree 7
    A = getPoly(5,2,False)
    B = getPoly(7,2,False)
    correctZeros([A,B], 0)

    #Case 5 - Three MultiCheb 3D of degrees 3,4 and 5
    A = getPoly(3,3,False)
    B = getPoly(4,3,False)
    C = getPoly(5,3,False)
    correctZeros([A,B,C], 0)

def test_div_power_roots():
    '''
    The following tests will run polyroots on relatively small random upper trianguler MultiPower.
    The assert statements will be inside of the correctZeros helper function.
    '''
    #Case 1 - Two MultiPower 2D degree 10 polynomials.
    A = getPoly(10,2,True)
    B = getPoly(10,2,True)
    correctZeros([A,B], -1)

    #Case 2 - Three MultiPower 3D degree 4 polynomials.
    A = getPoly(4,3,True)
    B = getPoly(4,3,True)
    C = getPoly(4,3,True)
    correctZeros([A,B,C], -1)

    #Case 3 - Four MultiPower 4D degree 2 polynomials.
    A = getPoly(2,4,True)
    B = getPoly(2,4,True)
    C = getPoly(2,4,True)
    D = getPoly(2,4,True)
    correctZeros([A,B,C,D], -1)

    #Case 4 - Two MultiPower 2D, one degree 5 and one degree 7
    A = getPoly(5,2,True)
    B = getPoly(7,2,True)
    correctZeros([A,B], -1)

    #Case 5 - Three MultiPower 3D of degrees 3,4 and 5
    A = getPoly(3,3,True)
    B = getPoly(4,3,True)
    C = getPoly(5,3,True)
    correctZeros([A,B,C], -1)

def test_div_cheb_roots():
    '''
    The following tests will run polyroots on relatively small random upper trianguler MultiCheb.
    The assert statements will be inside of the correctZeros helper function.
    '''
    #Case 1 - Two MultiCheb 2D degree 10 polynomials.
    A = getPoly(10,2,False)
    B = getPoly(10,2,False)
    correctZeros([A,B], -1)

    #Case 2 - Three MultiCheb 3D degree 4 polynomials.
    A = getPoly(4,3,False)
    B = getPoly(4,3,False)
    C = getPoly(4,3,False)
    correctZeros([A,B,C], -1)

    #Case 3 - Four MultiCheb 4D degree 2 polynomials.
    A = getPoly(2,4,False)
    B = getPoly(2,4,False)
    C = getPoly(2,4,False)
    D = getPoly(2,4,False)
    correctZeros([A,B,C,D], -1)

    #Case 4 - Two MultiCheb 2D, one degree 5 and one degree 7
    A = getPoly(5,2,False)
    B = getPoly(7,2,False)
    correctZeros([A,B], -1)

    #Case 5 - Three MultiCheb 3D of degrees 3,4 and 5
    A = getPoly(3,3,False)
    B = getPoly(4,3,False)
    C = getPoly(5,3,False)
    correctZeros([A,B,C], -1)

if __name__ == "__main__":
    test_div_power_roots()
