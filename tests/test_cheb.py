import numpy as np
from groebner.multi_cheb import MultiCheb
import pytest
import pdb
import random

def test_add():
    """Test Multivariate Chebyshev polynomial addition."""
    t = np.arange(27).reshape((3,3,3))
    poly1 = MultiCheb(t)
    poly2 = MultiCheb(np.ones((3,3,3)))
    S = poly1 + poly2 # the sum of the polynomials
    result = (S.coeff == (poly1.coeff + poly2.coeff))
    assert result.all()

def test_mult():
    """Test Multivariate Chebyshev polynomial multiplication."""
    test1 = np.array([[0,1],[2,1]])
    test2 = np.array([[2,2],[3,0]])
    cheb1 = MultiCheb(test1)
    cheb2 = MultiCheb(test2)
    new_cheb = cheb1*cheb2
    truth = MultiCheb(np.array([[4, 3.5, 1],[5,9,1],[3,1.5,0]]))
    assert np.allclose(new_cheb.coeff, truth.coeff)

def test_mult_diff():
    '''
    Test Multivariate Chebyshev polynomial multiplication.
    Test implementation with different shape sizes
    '''
    c1 = MultiCheb(np.arange(0,4).reshape(2,2))
    c2 = MultiCheb(np.ones((2,1)))
    p = c1*c2
    truth = MultiCheb(np.array([[1,2.5,0],[2,4,0],[1,1.5,0]]))
    assert np.allclose(p.coeff,truth.coeff)

def test_mon_mult():
    """
    Tests monomial multiplication using normal polynomial multiplication.
    """

    mon = (1,2)
    Poly = MultiCheb(np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]))
    mon_matr = MultiCheb(np.array([[0,0,0,0],[0,0,1,0],[0,0,0,0],[0,0,0,0]]))
    P1 = mon_matr*Poly
    P2 = MultiCheb.mon_mult(Poly, mon)

    mon2 = (0,1,1)
    Poly2 = MultiCheb(np.arange(1,9).reshape(2,2,2))
    mon_matr2 = MultiCheb(np.array([[[0,0],[0,1]],[[0,0],[0,0]]]))
    T1 = mon_matr2*Poly2
    T2 = MultiCheb.mon_mult(Poly2, mon2)


    assert np.allclose(P1.coeff, P2.coeff)
    assert np.allclose(T1.coeff, T2.coeff)

    #test with random matrices
    possible_dim = np.random.randint(1,4, (1,10))
    dim = possible_dim[0, random.randint(1,9)]
    shape = list()
    for i in range(dim):
        shape.append(random.randint(2,10))
    matrix1 = np.random.randint(1,101,(shape))
    M1 = MultiCheb(matrix1)

    #dim2 = possible_dim[0, random.randint(1,9)]
    shape2 = list()
    for i in range(dim):
        shape2.append(random.randint(2,10))
    matrix2 = np.ones(shape2)
    M2 = MultiCheb(matrix2)

    M3 = M1*M2
    M3 = MultiCheb(M3.coeff)

    for index, i in np.ndenumerate(M2.coeff):
        if sum(index) == 0:
            M4 = MultiCheb.mon_mult(M1, index)
        else:
            M4 = M4 + MultiCheb.mon_mult(M1, index)
    
    M4 = MultiCheb(M4.coeff)
    #print(M3.coeff.shape,M4.coeff.shape)
    #print(M3.coeff)
    #print(M4.coeff)
    
    assert np.allclose(M3.coeff, M4.coeff)

def test_evaluate_at():
    cheb = MultiCheb(np.array([[0,0,0,1],[0,0,0,0],[0,0,1,0]]))
    value = cheb.evaluate_at((2,5))
    assert(value == 828)

    value = cheb.evaluate_at((.25,.5))
    assert(np.isclose(value, -.5625))

def test_evaluate_at2():
    cheb = MultiCheb(np.array([[0,0,0,1],[0,0,0,0],[0,0,.5,0]]))
    value = cheb.evaluate_at((2,5))
    assert(np.isclose(value, 656.5))

