import pytest
import numpy as np
from itertools import permutations
from numalgsolve.root_finder import roots


# groebner module imports
from numalgsolve.gsolve import F4
from numalgsolve.polynomial import MultiPower, MultiCheb

def test_sorted_polys_monomial():
    #raise NotImplementedError
    pass

def test_sorted_polys_coeff():
    #raise NotImplementedError
    pass

def test_reduce_matrix():
    #raise NotImplementedError
    pass

def test_solve():
    #raise NotImplementedError
    pass

def testF4():
    #First Test
    A = MultiPower(np.array([[-10,0],[0,1],[1,0]]))
    B = MultiPower(np.array([[-26,0,0],[0,0,1],[0,0,0],[1,0,0]]))
    C = MultiPower(np.array([[-70,0,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,0]]))
    x1, y1 = F4([A,B,C])
    X = MultiPower(np.array([[-2.],[ 1.]]))
    Y = MultiPower(np.array([[-3.,1.]]))
    assert(np.any([X==i and Y==j for i,j in permutations((x1,y1),2)]))

    #Second Test
    A = MultiPower(np.array([
                         [[[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]],
                         [[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]
                        ]
                         ))
    B = MultiPower(np.array([
                         [[[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]],
                         [[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]
                        ]
                         ))
    C = MultiPower(np.array([
                         [[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]],
                         [[[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]
                        ]
                         ))
    w1, x1, y1, z1 = F4([A,B,C])

    W = MultiPower(np.array([[[[ 0.],[ 0.]],[[ 0.],[ 0.]],[[ 0.],[ 0.]],[[ 1.],[ 0.]]],
                             [[[ 0.],[-1.]],[[ 0.],[ 0.]],[[ 0.],[ 0.]],[[ 0.],[ 0.]]]]))
    X = MultiPower(np.array([[[[ 0.,0.,0.,0.,0.,1.],[-1.,0.,0.,0.,0.,0.]]]]))
    Y = MultiPower(np.array([[[[ 0.],[ 0.],[ 1.]],[[-1.],[ 0.],[ 0.]]]]))
    Z = MultiPower(np.array([[[[ 0.],[ 0.]],[[ 0.],[ 0.]],[[ 0.],[ 1.]]],
                             [[[-1.],[ 0.]],[[ 0.],[ 0.]],[[ 0.],[ 0.]]]]))

    assert(np.any([W==i and X==j and Y==k and Z==l for i,j,k,l in permutations((w1,x1,y1,z1),4)]))

    #Third Test
    A = MultiPower(np.array([[-1,0,1],[0,0,0]]))
    B = MultiPower(np.array([[-1,0,0],[0,1,0],[1,0,0]]))
    x1, y1 = F4([A,B])
    assert(np.any([A==i and B==j for i,j in permutations((x1,y1),2)]))

    #Fourth Test
    A = MultiPower(np.array([[-10,0],[0,1],[1,0]]))
    B = MultiPower(np.array([[-25,0,0],[0,0,1],[0,0,0],[1,0,0]]))
    C = MultiPower(np.array([[-70,0,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,0]]))
    X = MultiPower(np.array([[1.]]))
    x1 = F4([A,B,C])
    assert(X == x1[0])

    #Fifth Test
    A = MultiPower(np.array([[1,1],[0,0]]))
    B = MultiPower(np.array([[1,0],[1,0]]))
    C = MultiPower(np.array([[1,0],[1,0],[0,1]]))
    X = MultiPower(np.array([[1.]]))
    x1 = F4([A,B,C])
    assert(X == x1[0])

def test_phi_criterion():    
    # Simple Test Case (Nothing gets added )
    A = MultiPower(np.array([[-1,0,1],[0,0,0]]))
    B = MultiPower(np.array([[-1,0,0],[0,1,0],[1,0,0]]))
    
    x1,y1 = F4([A,B], phi = True)
    x2,y2 = F4([A,B], phi = False)

    assert(np.any([x2==i and y2==j for i,j in permutations((x1,y1),2)])), "Not the same basis!"
    
    #Second Test
    A = MultiPower(np.array([
                         [[[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]],
                         [[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]
                        ]
                         ))
    B = MultiPower(np.array([
                         [[[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]],
                         [[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]
                        ]
                         ))
    C = MultiPower(np.array([
                         [[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]],
                         [[[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]
                        ]
                         ))

    w1, x1, y1, z1 = F4([A,B,C], phi = True)
    w2, x2, y2, z2 = F4([A,B,C], phi = False)

    assert(np.any([w2==i and x2==j and y2==k and z2==l for i,j,k,l in permutations((w1,x1,y1,z1),4)]))

     #Third Test
    A = MultiPower(np.array([[-1,0,1],[0,0,0]]))
    B = MultiPower(np.array([[-1,0,0],[0,1,0],[1,0,0]]))
    x1, y1 = F4([A,B], phi = True)
    x2, y2 = F4([A,B], phi = False)

    assert(np.any([A==i and B==j for i,j in permutations((x1,y1),2)]))

    #Fourth Test
    A = MultiPower(np.array([[-10,0],[0,1],[1,0]]))
    B = MultiPower(np.array([[-25,0,0],[0,0,1],[0,0,0],[1,0,0]]))
    C = MultiPower(np.array([[-70,0,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,0]]))

    x1 = F4([A,B,C], phi = True)
    x2 = F4([A,B,C], phi = False)

    assert(x2[0] == x1[0])

    #Fifth Test
    A = MultiPower(np.array([[1,1],[0,0]]))
    B = MultiPower(np.array([[1,0],[1,0]]))
    C = MultiPower(np.array([[1,0],[1,0],[0,1]]))
    x1 = F4([A,B,C], phi = True)
    x2 = F4([A,B,C], phi = False)
    assert(x2[0]== x1[0])
    
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
    A helper function for test_TVB. Takes in polynomials, find their common zeros using TVB, and calculates
    how many of the zeros are correct.
    In this function it asserts that the number of zeros is equal to the product of the degrees, which is only valid if
    the polynomials are random and upper triangular, and that at least 95% of the zeros are correct (so it will pass even
    on bad random runs)
    '''
    zeros = roots(polys, method = 'Groebner')
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
    assert(100*correct/(len(zeros)-outOfRange) > 80)

def test_TVB_roots():
    '''
    The following tests will run TVB on relatively small random upper trianguler MultiPower and MultiCheb polynomials.
    The assert statements will be inside of the correctZeros helper function.
    '''
    #Case 1 - Two MultiPower 2D degree 4 polynomials.
    A = getPoly(4,2,True)
    B = getPoly(4,2,True)
    correctZeros([A,B])

    #Case 2 - Two MultiCheb 2D degree 4 polynomials.
    A = getPoly(4,2,False)
    B = getPoly(4,2,False)
    correctZeros([A,B])

    #Case 3 - Three MultiPower 3D degree 2 polynomials.
    A = getPoly(2,3,True)
    B = getPoly(2,3,True)
    C = getPoly(2,3,True)
    correctZeros([A,B,C])

    #Case 4 - Three MultiCheb 3D degree 2 polynomials.
    A = getPoly(2,3,False)
    B = getPoly(2,3,False)
    C = getPoly(2,3,False)
    correctZeros([A,B,C])

    #Case 5 - Four MultiPower 4D degree, three degree 2 and a degree 1.
    A = getPoly(2,4,True)
    B = getPoly(2,4,True)
    C = getPoly(2,4,True)
    D = getPoly(1,4,True)
    correctZeros([A,B,C,D])

    #Case 6 - Four MultiCheb 4D degree three degree 2 and a degree 1.
    A = getPoly(2,4,False)
    B = getPoly(2,4,False)
    C = getPoly(2,4,False)
    D = getPoly(1,4,False)
    correctZeros([A,B,C,D])

    #Case 7 - Two MultiPower 2D, one degree 3 and one degree 5
    A = getPoly(3,2,True)
    B = getPoly(5,2,True)
    correctZeros([A,B])

    #Case 8 - Two MultiCheb 2D, one degree 3 and one degree 5
    A = getPoly(3,2,False)
    B = getPoly(5,2,False)
    correctZeros([A,B])

    #Case 9 - Three MultiPower 3D of degrees 2,3 and 4
    A = getPoly(2,3,True)
    B = getPoly(3,3,True)
    C = getPoly(4,3,True)
    correctZeros([A,B,C])

    #Case 10 - Three MultiCheb 3D of degrees 2,3 and 4
    A = getPoly(2,3,False)
    B = getPoly(3,3,False)
    C = getPoly(4,3,False)
    correctZeros([A,B,C])