import numpy as np
from groebner.polynomial import Polynomial, MultiCheb, MultiPower
from groebner.TelenVanBarel import find_degree, mon_combos, sort_matrix
from itertools import product

def test_makeBasisDict():
    
    
    
    
    pass

def test_find_degree():
    #Test Case #1 - 2,3,4, and 5 2D Polynomials of degree 3
    
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
    
    #Test Case #2 - A 2D polynomials of degree 3 and one of degree 5
    degree5Coeff = np.array([
                    [1,1,1,1,1,1],
                    [1,1,1,1,1,0],
                    [1,1,1,1,0,0],
                    [1,1,1,0,0,0],
                    [1,1,0,0,0,0],
                    [1,0,0,0,0,0]])
    F = MultiPower(degree5Coeff)
    assert(find_degree([A,F]) == 7)
    
    #Test Case #3 - Two 3D polynomials of degree 15
    G = MultiPower(np.random.rand(6,6,6))
    H = MultiPower(np.random.rand(6,6,6))
    assert(find_degree([G,H]) == 29)

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
    
    #Test Case #1 - degree 25, dimension 2
    deg = 25
    dim = 2
    mons = mon_combos(np.zeros(dim, dtype = int),deg)
    mons2 = list()
    for i in product(np.arange(deg+1), repeat=dim):
        if np.sum(i) <= deg:
            mons2.append(i)
    for i in range(len(mons)):
        assert((mons[i] == mons2[i]).all())
    
    #Test Case #1 - degree 5, dimension 3
    deg = 5
    dim = 3
    mons = mon_combos(np.zeros(dim, dtype = int),deg)
    mons2 = list()
    for i in product(np.arange(deg+1), repeat=dim):
        if np.sum(i) <= deg:
            mons2.append(i)
    for i in range(len(mons)):
        assert((mons[i] == mons2[i]).all())
    
    #Test Case #1 - degree 5, dimension 5
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
    
    pass

def test_sort_matrix():
    
    
    pass

def test_rrqr_reduceTelenVanBarel():
    
    
    pass