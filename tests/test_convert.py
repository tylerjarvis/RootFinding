import numpy as np
import os, sys
from groebner.multi_cheb import MultiCheb
from groebner.multi_power import MultiPower
from groebner.convert_poly import cheb2poly, poly2cheb
import pytest
import pdb
import random
import time

def test_cheb2poly():
    c1 = MultiCheb(np.array([[0,0,0],[0,0,0],[0,1,1]]))
    c2 = cheb2poly(c1)
    truth = np.array([[1,-1,-2],[0,0,0],[-2,2,4]])
    assert np.allclose(truth,c2.coeff)

    #test2
    c3 = MultiCheb(np.array([[3,1,4,7],[8,3,1,2],[0,5,6,2]]))
    c4 = cheb2poly(c3)
    truth2 = np.array([[5,-19,-4,20],[7,-3,2,8],[-12,-2,24,16]])
    assert np.allclose(truth2,c4.coeff)

    c4_1 = poly2cheb(c4)
    assert np.allclose(c3.coeff, c4_1.coeff)

    #Test3
    c5 = MultiCheb(np.array([[3,1,3,4,6],[2,0,0,2,0],[3,1,5,1,8]]))
    c6 = cheb2poly(c5)
    truth3 = np.array([[0,-9,12,12,-16],[2,-6,0,8,0],[12,-4,-108,8,128]])
    assert np.allclose(truth3, c6.coeff)

    #test4 - Random 1D
    matrix2 = np.random.randint(1,100, random.randint(1,30))
    c7 = MultiCheb(matrix2)
    c8 = cheb2poly(c7)
    c9 = poly2cheb(c8)
    assert np.allclose(c7.coeff, c9.coeff)

    #Test5 - Random 2D
    shape = list()
    for i in range(2):
        shape.append(random.randint(2,30))
    matrix1 = np.random.randint(1,50,(shape))
    c10 = MultiCheb(matrix1)
    c11 = cheb2poly(c10)
    c12 = poly2cheb(c11)
    assert np.allclose(c10.coeff, c12.coeff)

    #Test6 - Random 4D
    shape = list()
    for i in range(4):
        shape.append(random.randint(2,15))
    matrix1 = np.random.randint(1,50,(shape))
    c13 = MultiCheb(matrix1)
    c14 = cheb2poly(c13)
    c15 = poly2cheb(c14)
    assert np.allclose(c13.coeff, c15.coeff)

def test_poly2cheb():
    P = MultiPower(np.array([[1,-1,-2],[0,0,0],[-2,2,4]]))
    c_new = poly2cheb(P)
    truth = np.array(np.array([[0,0,0],[0,0,0],[0,1,1]]))
    assert np.allclose(truth, c_new.coeff)

    #test2
    p3 = MultiPower(np.array([[5,-19,-4,20],[7,-3,2,8],[-12,-2,24,16]]))
    p4 = poly2cheb(p3)
    truth2 = np.array([[3,1,4,7],[8,3,1,2],[0,5,6,2]])
    assert np.allclose(truth2,p4.coeff)

    #Test3
    p5 = MultiPower(np.array([[0,-9,12,12,-16],[2,-6,0,8,0],[12,-4,-108,8,128]]))
    p6 = poly2cheb(p5)
    truth3 = np.array([[3,1,3,4,6],[2,0,0,2,0],[3,1,5,1,8]])
    assert np.allclose(truth3, p6.coeff)

    #test4 Random 1-D
    matrix2 = np.random.randint(1,100, random.randint(1,30))
    p7 = MultiPower(matrix2)
    p8 = poly2cheb(p7)
    p9 = cheb2poly(p8)
    assert np.allclose(p7.coeff, p9.coeff)

    #test5 Random 2D
    shape = list()
    for i in range(2):
        shape.append(random.randint(2,30))
    matrix1 = np.random.randint(1,50,(shape))
    p10 = MultiPower(matrix1)
    p11 = poly2cheb(p10)
    p12 = cheb2poly(p11)
    assert np.allclose(p10.coeff, p12.coeff)

    #test6 Random 4D
    shape = list()
    for i in range(4):
        shape.append(random.randint(2,15))
    matrix1 = np.random.randint(1,50,(shape))
    p13 = MultiPower(matrix1)
    p14 = poly2cheb(p13)
    p15 = cheb2poly(p14)
    assert np.allclose(p13.coeff, p15.coeff)
