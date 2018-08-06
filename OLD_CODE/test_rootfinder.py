"""TESTS REMOVED FOR NOW BECAUSE WE DON'T USE THIS CODE"""

"""
import numpy as np
from numalgsolve import root_finder as rf
from numalgsolve.polynomial import MultiPower, MultiCheb
from numalgsolve.gsolve import F4
import pytest
import pdb

def test_vectorSpaceBasis():
    f1 = MultiPower(np.array([[0,-1.5,.5],[-1.5,1.5,0],[1,0,0]]))
    f2 = MultiPower(np.array([[0,0,0],[-1,0,1],[0,0,0]]))
    f3 = MultiPower(np.array([[0,-1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]]))
    G = [f1, f2, f3]
    basis = rf.vectorSpaceBasis(G)[0]
    trueBasis = [(0,0), (1,0), (0,1), (1,1), (0,2)]

    assert ((len(basis) == len(trueBasis)) and (m in basis for m in trueBasis))
    #Failed on MultiPower in 2 vars."

def test_vectorSpaceBasis_2():
    f1 = MultiPower(np.array([[[0,0,1],[0,3/20,0],[0,0,0]],
                              [[0,0,0],[-3/40,1,0],[0,0,0]],
                              [[0,0,0],[0,0,0],[0,0,0]]]))

    f2 = MultiPower(np.array([[[3/16,-5/2,0],[0,3/16,0],[0,0,0]],
                              [[0,0,1],[0,0,0],[0,0,0]],
                              [[0,0,0],[0,0,0],[0,0,0]]]))

    f3 = MultiPower(np.array([[[0,1,1/2],[0,3/40,1],[0,0,0]],
                              [[-1/2,20/3,0],[-3/80,0,0],[0,0,0]],
                              [[0,0,0],[0,0,0],[0,0,0]]]))

    f4 = MultiPower(np.array([[[3/32,-7/5,0,1],[-3/16,83/32,0,0],[0,0,0,0],[0,0,0,0]],
                              [[3/40,-1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                              [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                              [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]]))

    f5 = MultiPower(np.array([[[5,0,0],[0,0,0],[0,0,0]],
                              [[0,-2,0],[0,0,0],[0,0,0]],
                              [[1,0,0],[0,0,0],[0,0,0]]]))

    f6 = MultiPower(np.array([[[0,0,0],[0,0,0],[1,0,0]],
                              [[0,-8/3,0],[0,0,0],[0,0,0]],
                              [[0,0,0],[0,0,0],[0,0,0]]]))

    G = [f1, f2, f3, f4, f5, f6]
    basis = rf.vectorSpaceBasis(G)[0]
    trueBasis = [(0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(0,0,2),(1,0,1),(0,1,1)]

    assert (len(basis) == len(trueBasis)) and (m in basis for m in trueBasis), \
            "Failed on MultiPower in 3 vars."

def testReducePoly():
    poly = MultiPower(np.array([[-3],[2],[-4],[1]]))
    g = MultiPower(np.array([[2],[1]]))
    basisSet = set()
    basisSet.add((0,0))

    reduced = rf.reduce_poly(poly, [g], basisSet)
    assert(MultiPower(reduced).coeff == np.array([[-31.]]))

def testReducePoly_2():
    poly = MultiPower(np.array([[-7],[2],[-13],[4]]))
    g = MultiPower(np.array([[-2],[3],[1]]))
    basisSet = set()
    basisSet.add((0,0))
    basisSet.add((1,0))

    reduced = rf.reduce_poly(poly, [g], basisSet)
    assert(np.all(MultiPower(reduced).coeff == np.array([[-57.],[85.]])))

def testReducePoly_3():
    poly = MultiPower(np.array([[0,-1,0,1],
                            [0,2,0,0],
                            [0,0,1,0],
                            [1,0,0,0]]))

    g1 = MultiPower(np.array([[0,0,0],
                          [-2,0,0],
                          [1,0,0]]))

    g2 = MultiPower(np.array([[0,-1,0,1],
                         [3,0,0,0],
                         [0,0,0,0],
                         [0,0,0,0]]))
    basisSet = set()
    basisSet.add((0,0))
    basisSet.add((0,1))
    basisSet.add((0,2))
    basisSet.add((1,0))
    basisSet.add((1,1))
    basisSet.add((1,2))

    reduced = rf.reduce_poly(poly, [g1, g2], basisSet)
    assert(np.all(MultiPower(reduced).coeff == np.array([[0,0,0],[1,2,2]])))

def testReducePoly_4():
    poly = MultiPower(np.array([[[-1,2,0],[0,0,0],[-3,0,0]],
                           [[0,0,0],[2,0,0],[0,0,0]],
                           [[0,0,0],[0,0,1],[0,0,0]]]))
    d1 = MultiPower(np.array([[[0,-3,0],
                          [0,0,0],
                          [1,0,0]]]))
    d2 = MultiPower(np.array([[[0,0,0,1],
                         [4,0,0,0]]]))
    d3 = MultiPower(np.array([[[-1]],[[1]]]))

    basisSet = set()
    for i in range(2):
        for j in range(3):
            for k in range(1):
                basisSet.add((k,i,j))

    reduced = rf.reduce_poly(poly, [d1, d2, d3], basisSet)

    assert(np.all(MultiPower(reduced).coeff == np.array([[[-1,-7,0],[2,0,1]]])))

def testCoordinateVector():
    poly = MultiCheb(np.array([[0,1,0],[0,0,1],[1,0,0]]))
    VB = [(2,0),(1,2),(0,1),(1,0)]
    GB = [MultiCheb(np.array([[0,0,0],[0,0,0],[0,0,1]]))] # LT is big so nothing gets reduced
    
    slices = ([2,1,0,1],[0,2,1,0])

    cv = rf.coordinateVector(poly, GB, set(VB), slices)
    print(cv)
    assert((cv == np.array([1,1,1,0])).all())

def testMultMatrix():
    f1 = MultiPower(np.array([[[5,0,0],[0,0,0],[0,0,0]],
                          [[0,-2,0],[0,0,0],[0,0,0]],
                          [[1,0,0],[0,0,0],[0,0,0]]]))

    f2 = MultiPower(np.array([[[1,0,0],[0,1,0],[0,0,0]],
                          [[0,0,0],[0,0,0],[1,0,0]],
                          [[0,0,0],[0,0,0],[0,0,0]]]))

    f3 = MultiPower(np.array([[[0,0,0],[0,0,0],[3,0,0]],
                          [[0,-8,0],[0,0,0],[0,0,0]],
                          [[0,0,0],[0,0,0],[0,0,0]]]))

    F = [f1, f2, f3]

    GB = F4(F)
    VB = rf.vectorSpaceBasis(GB)[0]

    x = MultiPower(np.array([[[0,1]]]))
    y = MultiPower(np.array([[[0],[1]]]))
    z = MultiPower(np.array([[[0]],[[1]]]))

    mx_RealEig = [eig.real for eig in \
        np.linalg.eigvals(rf.multMatrix(x, GB, VB)) if (eig.imag == 0)]

    my_RealEig = [eig.real for eig in \
        np.linalg.eigvals(rf.multMatrix(y, GB, VB)) if (eig.imag==0)]

    mz_RealEig = [eig.real for eig in \
        np.linalg.eigvals(rf.multMatrix(z, GB, VB)) if (eig.imag==0)]

    assert(len(mx_RealEig) == 2)
    assert(len(my_RealEig) == 2)
    assert(len(mz_RealEig) == 2)
    assert(np.allclose(mx_RealEig, [3.071618528, -2.821182227], atol=1.e-8))
    assert(np.allclose(my_RealEig, [-2.878002536, -2.81249605], atol=1.e-8))
    assert(np.allclose(mz_RealEig, [-1.100987715, .9657124563], atol=1.e-8))

def testMultMatrix_2():
    f1 = MultiPower(np.array([[0,-1.5,.5],[-1.5,1.5,0],[1,0,0]]))
    f2 = MultiPower(np.array([[0,0,0],[-1,0,1],[0,0,0]]))
    f3 = MultiPower(np.array([[0,-1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]]))

    GB = [f1, f2, f3]
    VB = rf.vectorSpaceBasis(GB)[0]

    x = MultiPower(np.array([[0],[1]]))
    y = MultiPower(np.array([[0,1]]))

    mx_Eig = np.linalg.eigvals(rf.multMatrix(x, GB, VB))
    my_Eig = np.linalg.eigvals(rf.multMatrix(y, GB, VB))

    assert(len(mx_Eig) == 5)
    assert(len(my_Eig) == 5)
    assert(np.allclose(mx_Eig, [-1., 2., 1., 1., 0.]))
    assert(np.allclose(my_Eig, [1., -1., 1., -1., 0.]))

def testRoots():
    f1 = MultiPower(np.array([[0,-1.5,.5],[-1.5,1.5,0],[1,0,0]]), clean_zeros=False)
    f2 = MultiPower(np.array([[0,0,0],[-1,0,1],[0,0,0]]), clean_zeros=False)
    f3 = MultiPower(np.array([[0,-1,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]]), clean_zeros=False)

    roots = rf.roots([f1, f2, f3], method='Groebner')
    values_at_roots = np.array([[f1(root) for root in roots],
                   [f2(root) for root in roots],
                   [f3(root) for root in roots]])

    assert(np.all(np.isclose(values_at_roots,0)))

def testRoots_2():
    f1 = MultiPower(np.array([[[5,0,0],[0,0,0],[0,0,0]],
                          [[0,-2,0],[0,0,0],[0,0,0]],
                          [[1,0,0],[0,0,0],[0,0,0]]]))

    f2 = MultiPower(np.array([[[1,0,0],[0,1,0],[0,0,0]],
                          [[0,0,0],[0,0,0],[1,0,0]],
                          [[0,0,0],[0,0,0],[0,0,0]]]))

    f3 = MultiPower(np.array([[[0,0,0],[0,0,0],[3,0,0]],
                          [[0,-8,0],[0,0,0],[0,0,0]],
                          [[0,0,0],[0,0,0],[0,0,0]]]))

    roots = rf.roots([f1, f2, f3], method='Groebner')

    values_at_roots = np.array([[f1(root) for root in roots],
                    [f2(root) for root in roots],
                    [f3(root) for root in roots]])

    assert(np.all(np.isclose(values_at_roots,0)))

def testRoots_3():
    # roots of [x^2-y, x^3-y+1]
    f1 = MultiPower(np.array([[0,-1],[0,0],[1,0]]))
    f2 = MultiPower(np.array([[1,-1],[0,0],[0,0],[1,0]]))

    roots = rf.roots([f1, f2], method='Groebner')

    values_at_roots = np.array([[f1(root) for root in roots],
                                [f2(root) for root in roots]])

    assert(np.all(np.isclose(values_at_roots,0)))

def testRoots_4():
    f1 = MultiPower(np.array([[5,-1],[1,0]]))
    f2 = MultiPower(np.array([[1,-1],[-1,0]]))

    root = rf.roots([f1, f2], method='Macaulay')[0]

    assert(all(np.isclose(root, [-2,3])))

def testRoots_5():
    f1 = MultiPower(np.array([[0,-1],[0,0],[1,0]]))
    f2 = MultiPower(np.array([[1,-1],[1,0]]))

    roots = rf.roots([f1, f2], method='Macaulay')

    assert(all(np.isclose(roots[0], [-0.61803399,  0.38196601])))
    assert(all(np.isclose(roots[1], [1.61803399,  2.61803399])))

def testRoots_6(): # test when ideal is not zero-dimensional
    f1 = MultiPower(np.array([[-12,-12],[1,1],[1,1]]))
    f2 = MultiPower(np.array([[6,3,-3],[-2,-1,1]]))

    roots = rf.roots([f1, f2], method='Macaulay')
    assert(roots == -1)
"""