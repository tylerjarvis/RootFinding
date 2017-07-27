import numpy as np
import os,sys
from groebner.multi_power import MultiPower
import pytest
import random


def test_add():
    a1 = np.arange(27).reshape((3,3,3))
    Test2 = MultiPower(a1)
    a2 = np.ones((3,3,3))
    Test3 = MultiPower(a2)
    addTest = Test2 + Test3
    assert (addTest.coeff == (Test2.coeff + Test3.coeff)).all()

def test_mult():
    test1 = np.array([[0,1],[2,1]])
    test2 = np.array([[2,2],[3,0]])
    p1 = MultiPower(test1)
    p2 = MultiPower(test2)
    new_poly = p1*p2
    truth = MultiPower(np.array([[0, 2, 2],[4,9,2],[6,3,0]]))
    assert np.allclose(new_poly.coeff, truth.coeff)

""" THE GENERATOR IS CURRENTLY OUT OF COMMISION.
def test_generator():
    poly = MultiPower(np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]))
    gen = poly.degrevlex_gen()
    i = 0
    for idx in gen:
        i += 1
        if(i == 1):
            assert (idx == [4., 4.]).all()
        elif(i == 2):
            assert (idx == [4., 3.]).all()
        elif(i == 3):
            assert (idx == [3., 4.]).all()
        elif(i == 4):
            assert (idx == [4., 2.]).all()
        elif(i == 5):
            assert (idx == [3., 3.]).all()
        elif(i == 6):
            assert (idx == [2., 4.]).all()
        elif(i == 7):
            assert (idx == [4., 1.]).all()
        elif(i == 8):
            assert (idx == [3., 2.]).all()
        elif(i == 9):
            assert (idx == [2., 3.]).all()
        elif(i == 10):
            assert (idx == [1., 4.]).all()
        elif(i == 11):
            assert (idx == [4., 0.]).all()
        elif(i == 12):
            assert (idx == [3., 1.]).all()
        elif(i == 13):
            assert (idx == [2., 2.]).all()
        elif(i == 14):
            assert (idx == [1., 3.]).all()
        elif(i == 15):
            assert (idx == [0., 4.]).all()
        elif(i == 16):
            assert (idx == [3., 0.]).all()
        elif(i == 17):
            assert (idx == [2., 1.]).all()
        elif(i == 18):
            assert (idx == [1., 2.]).all()
        elif(i == 19):
            assert (idx == [0., 3.]).all()
        elif(i == 20):
            assert (idx == [2., 0.]).all()
        elif(i == 21):
            assert (idx == [1., 1.]).all()
        elif(i == 22):
            assert (idx == [0., 2.]).all()
        elif(i == 23):
            assert (idx == [1., 0.]).all()
        elif(i == 24):
            assert (idx == [0., 1.]).all()
        elif(i == 25):
            assert (idx == [0., 0.]).all()


    poly = MultiPower(np.array([[[[0.,0.,0.],[0.,0.,0.]], [[0.,0.,0.],[0.,0.,0.]]], [[[0.,0.,0.],[0.,0.,0.]], [[0.,0.,0.],[0.,0.,0.]]]]),
                      lead_term = (0,0,0,0))
    gen = poly.degrevlex_gen()
    i = 0
    for idx in gen:
        i += 1
        if(i == 1):
            assert (idx == [ 1.,  1.,  1.,  2.]).all()
        elif(i == 2):
            assert (idx == [ 1.,  1.,  1.,  1.]).all()
        elif(i == 3):
            assert (idx == [ 1.,  1.,  0.,  2.]).all()
        elif(i == 4):
            assert (idx == [ 1.,  0.,  1.,  2.]).all()
        elif(i == 5):
            assert (idx == [ 0.,  1.,  1.,  2.]).all()
        elif(i == 6):
            assert (idx == [ 1.,  1.,  1.,  0.]).all()
        elif(i == 7):
            assert (idx == [ 1.,  1.,  0.,  1.]).all()
        elif(i == 8):
            assert (idx == [ 1.,  0.,  1.,  1.]).all()
        elif(i == 9):
             assert (idx == [ 0.,  1.,  1.,  1.]).all()
        elif(i == 10):
            assert (idx == [ 1.,  0.,  0.,  2.]).all()
        elif(i == 11):
            assert (idx == [ 0.,  1.,  0.,  2.]).all()
        elif(i == 12):
            assert (idx == [ 0.,  0.,  1.,  2.]).all()
        elif(i == 13):
            assert (idx == [ 1.,  1.,  0.,  0.]).all()
        elif(i == 14):
            assert (idx == [ 1.,  0.,  1.,  0.]).all()
        elif(i == 15):
            assert (idx == [ 0.,  1.,  1.,  0.]).all()
        elif(i == 16):
            assert (idx == [ 1.,  0.,  0.,  1.]).all()
        elif(i == 17):
            assert (idx == [ 0.,  1.,  0.,  1.]).all()
        elif(i == 18):
            assert (idx == [ 0.,  0.,  1.,  1.]).all()
        elif(i == 19):
            assert (idx == [ 0.,  0.,  0.,  2.]).all()
        elif(i == 20):
            assert (idx == [ 1.,  0.,  0.,  0.]).all()
        elif(i == 21):
            assert (idx == [ 0.,  1.,  0.,  0.]).all()
        elif(i == 22):
            assert (idx == [ 0.,  0.,  1.,  0.]).all()
        elif(i == 23):
            assert (idx == [ 0.,  0.,  0.,  1.]).all()
        elif(i == 24):
            assert (idx == [ 0.,  0.,  0.,  0.]).all()
"""
def test_mon_mult():
    """
    Tests monomial multiplication using normal polynomial multiplication.
    """

    mon = (1,2)
    Poly = MultiPower(np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]))
    mon_matr = MultiPower(np.array([[0,0,0,0],[0,0,1,0],[0,0,0,0],[0,0,0,0]]))
    P1 = mon_matr*Poly
    P2 = MultiPower.mon_mult(Poly, mon)

    mon2 = (0,1,1)
    Poly2 = MultiPower(np.arange(1,9).reshape(2,2,2))
    mon_matr2 = MultiPower(np.array([[[0,0],[0,1]],[[0,0],[0,0]]]))
    T1 = mon_matr2*Poly2
    T2 = MultiPower.mon_mult(Poly2, mon2)


    assert np.allclose(P1.coeff, P2.coeff, atol = 1.0e-10)
    assert np.allclose(T1.coeff, T2.coeff, atol = 1.0e-10)

def test_mon_mult_random():
    #test with random matrices
    possible_dim = np.random.randint(1,5, (1,10))
    dim = possible_dim[0, random.randint(1,9)]

    shape = list()
    for i in range(dim):
        shape.append(random.randint(2,4))
    matrix1 = np.random.randint(1,11,(shape))
    M1 = MultiPower(matrix1)

    shape2 = list()
    for i in range(dim):
        shape2.append(random.randint(2,4))
    matrix2 = np.ones(shape2)
    M2 = MultiPower(matrix2)

    M3 = M1*M2

    for index, i in np.ndenumerate(M2.coeff):
        if sum(index) == 0:
            M4 = MultiPower.mon_mult(M1, index)
        else:
            M4 = M4 + MultiPower.mon_mult(M1, index)

    if M3.shape != M4.shape:
        new_M3, new_M4 = MultiPower.match_size(M3,M3,M4)
    else:
        new_M3, new_M4 = M3, M4

    assert np.allclose(new_M3.coeff, new_M4.coeff)

def test_evaluate_at():
    # Evaluate .5xyz + 2x + y + z at (4,2,1)
    poly = MultiPower(np.array([[[0,1,0],
                                [1,0,0],
                                [0,0,0]],
                                [[2,0,0],
                                [0,.5,0],
                                [0,0,0]]]))

    assert(poly.evaluate_at((4,2,1)) == 15)

def test_evaluate_at2():
    # Evaluate -.5x^2y + 2xy^2 - 3z^2 + yz at (7.4,2.33,.25)
    poly = MultiPower(np.array([[[0,0,-3],
                                [0,1,0],
                                [0,0,0]],
                                [[0,0,0],
                                [0,0,0],
                                [2,0,0]],
                                [[0,0,0],
                                [-.5,0,0],
                                [0,0,0]]]))

    assert(np.isclose(poly.evaluate_at((7.4, 2.33, .25)), 16.94732))
