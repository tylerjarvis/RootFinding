import numpy as np
from groebner.Macaulay import Macaulay, find_degree, add_polys, create_matrix, mon_combos
from groebner.polynomial import MultiCheb, MultiPower
from groebner.root_finder import roots
import pytest
import random
from itertools import product

def test_Macaulay():

    #test 1 - compare Groebner results and Macaulay results in Chebyshev and Power.
    for i in range(5):
        ACoeff = np.random.rand(3,3)
        BCoeff = np.random.rand(3,3)
        for i,j in np.ndenumerate(ACoeff):
            if np.sum(i) > 3:
                ACoeff[i] = 0
                BCoeff[i] = 0
        A = MultiCheb(ACoeff)
        B = MultiCheb(BCoeff)

        A1 = MultiPower(ACoeff)
        B1 = MultiPower(BCoeff)

        zeros_from_Macaulay = roots([A,B], 'Macaulay')
        zeros_from_Groebner = roots([A,B], 'Groebner')

        zeros_from_Macaulay1 = roots([A1,B1], 'Macaulay')
        zeros_from_Groebner1 = roots([A1,B1], 'Groebner')

        sorted_from_Macaulay = np.sort(zeros_from_Macaulay, axis = 0)
        sorted_from_Groebner = np.sort(zeros_from_Groebner, axis = 0)

        sorted_from_Macaulay1 = np.sort(zeros_from_Macaulay1, axis = 0)
        sorted_from_Groebner1 = np.sort(zeros_from_Groebner1, axis = 0)

        assert np.allclose(sorted_from_Macaulay, sorted_from_Groebner)
        assert np.allclose(sorted_from_Macaulay1, sorted_from_Groebner1)

        # Test 2 - Hand calculated example in the Power basis

        poly1 = MultiPower(np.array([[3,0,1],[0,0,0],[0,0,1]]))
        poly2 = MultiPower(np.array([[3,0],[1,1],[0,1]]))

def test_get_poly_from_matrix():
    raise NotImplementedError

def test_get_good_rows():
    raise NotImplementedError

def test_find_degree():
    '''Test Case #1 - 2,3,4, and 5 2D Polynomials of degree 3'''
    degree3Coeff = np.array([
                    [1,1,1,1],
                    [1,1,1,0],
                    [1,1,0,0],
                    [1,0,0,0]])
    A = MultiPower(degree3Coeff)
    B = MultiPower(degree3Coeff)
    C = MultiPower(degree3Coeff)
    D = MultiPower(degree3Coeff)
    E = MultiPower(degree3Coeff)
    assert(find_degree([A,B]) == 5)
    assert(find_degree([A,B,C]) == 7)
    assert(find_degree([A,B,C,D]) == 9)
    assert(find_degree([A,B,C,D,E]) == 11)

    '''Test Case #2 - A 2D polynomials of degree 3 and one of degree 5'''
    degree5Coeff = np.array([
                    [1,1,1,1,1,1],
                    [1,1,1,1,1,0],
                    [1,1,1,1,0,0],
                    [1,1,1,0,0,0],
                    [1,1,0,0,0,0],
                    [1,0,0,0,0,0]])
    F = MultiPower(degree5Coeff)
    assert(find_degree([A,F]) == 7)

    ''' Test Case #3 - Two 3D polynomials of degree 15'''
    G = MultiPower(np.random.rand(6,6,6))
    H = MultiPower(np.random.rand(6,6,6))
    assert(find_degree([G,H]) == 29)

    #Test 3 - Simple Example in 2D
    poly1 = MultiPower(np.array([[3,0,1],[0,0,0],[0,0,1]]))
    poly2 = MultiPower(np.array([[3,0],[1,1],[0,1]]))
    found_degree = find_degree([poly1,poly2])
    correct_degree = 6
    assert found_degree == correct_degree

    #Test 4 - Simple Example in 3D
    a = np.zeros((4,4,4))
    a[3,3,3] = 1
    poly1 = MultiCheb(a)
    poly2 = MultiCheb(np.ones((3,5,4)))
    poly3 = MultiCheb(np.ones((2,4,5)))
    found_degree1 = find_degree([poly1,poly2,poly3])
    correct_degree1 = 24
    assert found_degree1 == correct_degree1

def test_mon_combos():
    '''
    Tests the mon_combos function against the simpler itertools product.
    '''
    #Test Case #1 - degree 5, dimension 2
    deg = 5
    dim = 2
    mons = mon_combos(np.zeros(dim, dtype = int),deg)
    mons2 = list()
    for i in product(np.arange(deg+1), repeat=dim):
        if np.sum(i) <= deg:
            mons2.append(i)
    for i in range(len(mons)):
        assert((mons[i] == mons2[i]).all())

    #Test Case #2 - degree 25, dimension 2
    deg = 25
    dim = 2
    mons = mon_combos(np.zeros(dim, dtype = int),deg)
    mons2 = list()
    for i in product(np.arange(deg+1), repeat=dim):
        if np.sum(i) <= deg:
            mons2.append(i)
    for i in range(len(mons)):
        assert((mons[i] == mons2[i]).all())

    #Test Case #3 - degree 5, dimension 3
    deg = 5
    dim = 3
    mons = mon_combos(np.zeros(dim, dtype = int),deg)
    mons2 = list()
    for i in product(np.arange(deg+1), repeat=dim):
        if np.sum(i) <= deg:
            mons2.append(i)
    for i in range(len(mons)):
        assert((mons[i] == mons2[i]).all())

    #Test Case #4 - degree 5, dimension 5
    deg = 5
    dim = 5
    mons = mon_combos(np.zeros(dim, dtype = int),deg)
    mons2 = list()
    for i in product(np.arange(deg+1), repeat=dim):
        if np.sum(i) <= deg:
            mons2.append(i)
    for i in range(len(mons)):
        assert((mons[i] == mons2[i]).all())

def test_add_polys():
    raise NotImplementedError

def test_sort_matrix():
    raise NotImplementedError

def test_clean_matrix():
    raise NotImplementedError

def test_create_matrix():
    raise NotImplementedError

def test_create_matrix2():
    raise NotImplementedError

def test_rrqr_reduce():
    raise NotImplementedError

def test_rrqr_reduce2():
    raise NotImplementedError
