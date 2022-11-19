"""
A solid 2 dimensional check before I hit the pull request.
"""
import numpy as np
import yroots.M_maker as M_maker
import yroots.ChebyshevSubdivisionSolver as ChebyshevSubdivisionSolver
import pytest
from yroots.polynomial import MultiCheb
from yroots.utils import transform
from yroots.Combined_Solver import solve

f = lambda x,y: (x-1)*(np.cos(x*y**2)+2)
g = lambda x,y: np.sin(8*np.pi*y)*(np.cos(x*y)+2)
f_deg,g_deg = 20,20

def solver_check(funcs,a,b):
    """
    raw functions on [-1,1]^n
    """

    f,g = funcs
    guess_degs = [f_deg,g_deg]
    yroots_1 = solve(funcs,a,b,guess_degs)

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
        solve([f,g],a,b,[f_deg,g_deg])
    assert excinfo.value.args[0] == "At least one lower bound is >= an upper bound."

    a = [a[0]]
    with pytest.raises(ValueError) as excinfo:
        solve([f,g],a,b,[f_deg,g_deg])
    assert excinfo.value.args[0] == "Dimension mismatch in intervals."

def test_exact_option():
    f = lambda x,y: np.sin(4*(x + y/10 + np.pi/10))
    g = lambda x,y: np.cos(2*(x-2*y+ np.pi/7))
    a,b = np.array([-1,-1]),np.array([1,1])

    funcs = [f,g]
    f_deg, g_deg = 16,32
    guess_degs = [f_deg,g_deg]
    yroots_non_exact = solve(funcs,a,b,guess_degs,exact=False) #FALSE --> non_exact
    yroots_exact = solve(funcs,a,b,guess_degs,exact=True) #TRUE --> exact

    actual_roots = np.load('Polished_results/polished_2.3.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_2.3.csv', delimiter=',')

    assert len(yroots_non_exact) == len(actual_roots)
    assert len(yroots_exact) == len(actual_roots)
    assert len(yroots_exact) == len(chebfun_roots)

    actual_roots = ChebyshevSubdivisionSolver.sortRoots(actual_roots)
    yroots_non_exact = ChebyshevSubdivisionSolver.sortRoots(yroots_non_exact)
    yroots_exact = ChebyshevSubdivisionSolver.sortRoots(yroots_exact) 
    chebfun_roots = ChebyshevSubdivisionSolver.sortRoots(chebfun_roots) #sort the Roots

    assert np.allclose(yroots_exact,actual_roots)
    assert np.allclose(yroots_exact,chebfun_roots)
    assert np.allclose(yroots_non_exact,actual_roots)
    assert np.allclose(yroots_non_exact,chebfun_roots)

def testreturnBoundingBoxes():
    f = lambda x,y: np.sin(4*(x + y/10 + np.pi/10))
    g = lambda x,y: np.cos(2*(x-2*y+ np.pi/7))
    a,b = np.array([-1,-1]),np.array([1,1])

    funcs = [f,g]
    f_deg, g_deg = 16,32
    guess_degs = [f_deg,g_deg]

    yroots, boxes = solve(funcs,a,b,guess_degs,returnBoundingBoxes=True)

    for root, box in zip(yroots,boxes):
        box = ChebyshevSubdivisionSolver.TrackedInterval(box)
        assert box.__contains__(root) == True

def testoutside_neg1_pos1():
    f = lambda x,y: 2*x*y*np.cos(y**2)*np.cos(2*x)-np.cos(x*y)
    g = lambda x,y: 2*np.sin(x*y**2)*np.sin(3*x*y)-np.sin(x*y)
    a,b = np.array([-2,-2]), np.array([2,2])
    funcs = [f,g]
    f_deg,g_deg = 16,16
    guess_degs = [f_deg,g_deg]
    
    yroots, boxes = solve(funcs,a,b,guess_degs,returnBoundingBoxes=True)
    for root, box in zip(yroots,boxes):
        box = ChebyshevSubdivisionSolver.TrackedInterval(box)
        assert box.__contains__(root) == True

def test_default_nodeg():
    f = lambda x,y: np.sin(4*(x + y/10 + np.pi/10))
    g = lambda x,y: np.cos(2*(x-2*y+ np.pi/7))
    a,b = np.array([-1,-1]),np.array([1,1])

    funcs = [f,g]

    yroots = solve(funcs,a,b)

    actual_roots = np.load('Polished_results/polished_2.3.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_2.3.csv', delimiter=',')

    actual_roots = ChebyshevSubdivisionSolver.sortRoots(actual_roots)
    chebfun_roots = ChebyshevSubdivisionSolver.sortRoots(chebfun_roots) #sort the Roots
    yroots = ChebyshevSubdivisionSolver.sortRoots(yroots) 

    assert np.allclose(yroots,actual_roots)
    assert np.allclose(yroots,chebfun_roots)





#TODO: can I delete all the stuff below?

#WHAT CAN WE TEST ABOUT THIS CODE
#WE CAN CHECK THAT IT PRESERVES WHAT ERIKs solver does when it is given the approximations
    #CASES
    #not neg1_1 and not all multicheb
    #neg1_1 and not all multicheb
    #not neg1_1 and all multicheb
    #both neg1_1 and all multicheb

    #value error check for a and b

