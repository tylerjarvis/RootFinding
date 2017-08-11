import numpy as np
from groebner.polynomial import Polynomial, MultiCheb, MultiPower
from groebner.TelenVanBarel import find_degree, mon_combos, sort_matrix_terms
from groebner.root_finder import roots
from itertools import product

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
    return polys

def correctZeros(polys):
    '''
    A helper function for test_TVB. Takes in polynomials, find their common zeros using TVB, and calculates
    how many of the zeros are correct.
    In this function it asserts that the number of zeros is equal to the product of the degrees, which is only valid if
    the polynomials are random and upper triangular, and that at least 95% of the zeros are correct (so it will pass even
    on bad random runs)
    '''
    zeros = roots(polys, method = 'TVB')
    assert(zeros != -1)
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
                else:
                    #print("ZERO OFF")
                    #print(zero)
                    #print(cheb.chebval2d(zero[0],zero[1],poly.coeff))
                    pass
                break
        if good:
            correct += 1
    assert(100*correct/(len(zeros)-outOfRange) > 95)

def test_TVB_roots():
    '''
    The following tests will run TVB on relatively small random upper trianguler MultiPower and MultiCheb polynomials.
    The assert statements will be inside of the correctZeros helper function.
    '''
    #Case 1 - Two MultiPower 2D degree 10 polynomials.
    A = getPoly(10,2,True)
    B = getPoly(10,2,True)
    correctZeros([A,B])
    
    #Case 2 - Two MultiCheb 2D degree 10 polynomials.
    A = getPoly(10,2,False)
    B = getPoly(10,2,False)
    correctZeros([A,B])
    
    #Case 3 - Three MultiPower 3D degree 4 polynomials.
    A = getPoly(4,3,True)
    B = getPoly(4,3,True)
    C = getPoly(4,3,True)
    correctZeros([A,B,C])
    
    #Case 4 - Three MultiCheb 3D degree 4 polynomials.
    A = getPoly(4,3,False)
    B = getPoly(4,3,False)
    C = getPoly(4,3,False)
    correctZeros([A,B,C])

    #Case 5 - Four MultiPower 4D degree 2 polynomials.
    A = getPoly(2,4,True)
    B = getPoly(2,4,True)
    C = getPoly(2,4,True)
    D = getPoly(2,4,True)
    correctZeros([A,B,C,D])
    
    #Case 6 - Four MultiCheb 4D degree 2 polynomials.
    A = getPoly(2,4,False)
    B = getPoly(2,4,False)
    C = getPoly(2,4,False)
    D = getPoly(2,4,False)
    correctZeros([A,B,C,D])

    #Case 7 - Two MultiPower 2D, one degree 5 and one degree 7
    A = getPoly(5,2,True)
    B = getPoly(7,2,True)
    correctZeros([A,B])

    #Case 8 - Two MultiCheb 2D, one degree 5 and one degree 7
    A = getPoly(5,2,False)
    B = getPoly(7,2,False)
    correctZeros([A,B])
    
    #Case 9 - Three MultiPower 3D of degrees 3,4 and 5
    A = getPoly(3,3,True)
    B = getPoly(4,3,True)
    C = getPoly(5,3,True)
    correctZeros([A,B,C])
    
    #Case 10 - Three MultiCheb 3D of degrees 3,4 and 5
    A = getPoly(3,3,False)
    B = getPoly(4,3,False)
    C = getPoly(5,3,False)
    correctZeros([A,B,C])
        
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