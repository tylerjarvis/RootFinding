"""
A solid 2 dimensional check before I hit the pull request.
"""
import numpy as np
import yroots.M_maker as M_maker
import yroots.ChebyshevSubdivisionSolver as ChebyshevSubdivisionSolver
import pytest
from yroots.polynomial import MultiCheb
from yroots.utils import transform
from yroots.Combined_Solver import solver

f = lambda x,y: (x-1)*(np.cos(x*y**2)+2)
g = lambda x,y: np.sin(8*np.pi*y)*(np.cos(x*y)+2)
f_deg,g_deg = 20,20

def solver_check(funcs,a,b):
    """
    raw functions on [-1,1]^n
    """

    f,g = funcs
    guess_degs = [f_deg,g_deg]
    yroots_1 = solver(funcs,a,b,guess_degs)

    arr_neg1 = np.array([-1]*len(a)) #what if a>b
    arr_1 = np.ones(len(a))

    f_approx = M_maker.M_maker(f,arr_neg1,arr_1,f_deg)
    g_approx = M_maker.M_maker(g,arr_neg1,arr_1,g_deg)
  
    yroots_2 = np.array(ChebyshevSubdivisionSolver.solveChebyshevSubdivision([f_approx.M,g_approx.M],np.array([f_approx.err,g_approx.err])))
    if len(yroots_2) > 0: #transform doesn't work on empty arrays
        yroots_2 = transform(yroots_2,a,b)

    return np.allclose(yroots_1,yroots_2)

def test_solver():
    a = -1*np.random.random(2)
    b = np.random.random(2)
    assert solver_check([f,g],a,b) == True
    b = np.ones(2).astype(float)
    a = -1*b
    assert solver_check([f,g],a,b) == True

    a,b = np.array([-0.5,-0.75]), np.array([0.25,0.7])
    g_approx = M_maker.M_maker(g,a,b,g_deg)
    h = MultiCheb(g_approx.M)
    f_approx = M_maker.M_maker(f,a,b,g_deg)
    k = MultiCheb(f_approx.M)

    assert solver_check([h,k],a,b) == True

    a,b = np.array([-0.9,-0.9]), np.array([0.9,0.9])
    assert solver_check([h,k],a,b) == True

def test_bad_intervals():
    a,b = np.array([1,-1]),np.array([1,1])
    funcs = [f,g]
    with pytest.raises(ValueError) as excinfo:
        solver([f,g],a,b,[f_deg,g_deg])
    print(excinfo)
    assert excinfo.value.args[0] == "At least one lower bound is >= an upper bound."

    a = [a[0]]
    with pytest.raises(ValueError) as excinfo:
        solver([f,g],a,b,[f_deg,g_deg])
    print(excinfo)
    assert excinfo.value.args[0] == "Dimension mismatch in intervals."

def test_exact_option():
    f = lambda x,y: np.sin(4*(x + y/10 + np.pi/10))
    g = lambda x,y: np.cos(2*(x-2*y+ np.pi/7))
    a,b = np.array([-1,-1]),np.array([1,1])

    funcs = [f,g]
    f_deg, g_deg = 16,32
    guess_degs = [f_deg,g_deg]
    yroots_non_exact = solver(funcs,a,b,guess_degs,exact=False)
    yroots_exact = solver(funcs,a,b,guess_degs,exact=True)

    actual_roots_2_3 = np.array([[-0.35797059,  0.43811326],
    [-0.28317077, -0.30988499],
    [ 0.39002766,  0.81211239],
    [ 0.46482748,  0.06411414],
    [ 0.53962731, -0.68388412]])

    print(len(yroots_exact))
    print(len(yroots_non_exact))
    print(len(actual_roots_2_3))

    assert len(yroots_non_exact) == len(actual_roots_2_3)
    assert len(yroots_exact) == len(actual_roots_2_3)

    norm_yroots_non_exact = np.linalg.norm(yroots_non_exact-actual_roots_2_3)
    norm_yroots_exact = np.linalg.norm(yroots_exact-actual_roots_2_3)

    assert norm_yroots_exact <= norm_yroots_non_exact

def testreturnBoundingBoxes():
    f = lambda x,y: np.sin(4*(x + y/10 + np.pi/10))
    g = lambda x,y: np.cos(2*(x-2*y+ np.pi/7))
    a,b = np.array([-1,-1]),np.array([1,1])

    funcs = [f,g]
    f_deg, g_deg = 16,32
    guess_degs = [f_deg,g_deg]

    yroots, boxes = solver(funcs,a,b,guess_degs,returnBoundingBoxes=True)

    for root, box in zip(yroots,boxes):
        box = ChebyshevSubdivisionSolver.TrackedInterval(box)
        assert box.__contains__(root) == True

#TODO: can I delete all the stuff below?

#WHAT CAN WE TEST ABOUT THIS CODE
#WE CAN CHECK THAT IT PRESERVES WHAT ERIKs solver does when it is given the approximations
    #CASES
    #not neg1_1 and not all multicheb
    #neg1_1 and not all multicheb
    #not neg1_1 and all multicheb
    #both neg1_1 and all multicheb

    #value error check for a and b

