"""
I run all these checks in two dimensions before hitting a pull request.
TODO: add checks in higher dimensions or make sure that these checks are
representative of higher dimensions
"""
import numpy as np
import yroots.M_maker as M_maker
import yroots.ChebyshevSubdivisionSolver as ChebyshevSubdivisionSolver
import pytest
from yroots.polynomial import MultiCheb, MultiPower
from yroots.utils import transform
from yroots.Combined_Solver import solve #, degree_guesser
import inspect
import sympy as sy

f = lambda x,y: (x-1)*(np.cos(x*y**2)+2)
g = lambda x,y: np.sin(8*np.pi*y)*(np.cos(x*y)+2)
f_deg,g_deg = 20,20

def solver_check(funcs,a,b):
    """
    parameters
    ----------
    funcs: list 
    collection of callable functions on [-1,1]^n
    a,b: ndarray
    lower and upper bounds

    returns
    -------
    bool: whether or not the solver accomplishes the same result as plugging the output of M_maker to solveChebyshevSubdivision
    """

    f,g = funcs
    yroots_1 = solve(funcs,a,b)

    arr_neg1 = np.array([-1]*len(a))
    arr_1 = np.ones(len(a))

    f_approx = M_maker.M_maker(f,arr_neg1,arr_1,f_deg)
    g_approx = M_maker.M_maker(g,arr_neg1,arr_1,g_deg)
    
    #TODO: make sure plugging in multicheb objects for [f_approx.M,g_approx.M]
    yroots_2 = np.array(ChebyshevSubdivisionSolver.solveChebyshevSubdivision([f_approx.M,g_approx.M],np.array([f_approx.err,g_approx.err])))
    if len(yroots_2) > 0: #because transform doesn't work on empty arrays
        yroots_2 = transform(yroots_2,a,b)
    else: #case where no roots are found
        return len(yroots_1) == 0 

    return np.allclose(yroots_1,yroots_2)

def test_solver():
    """
    runs solver_check() on the six following cases:
    (a) non-[-1,1]^n region of the space
        (i) non-MultiCheb objects
        (ii) some MultiCheb, some non-MultiCheb objects
        (iii) MultiCheb objects
    (b) ^same as above, but on [-1,1]^n region of the space
    """
    a = -1*np.random.random(2)
    b = np.random.random(2)
    arr_neg1 = np.array([-1]*len(a))
    arr_1 = np.ones(len(a))

    g_approx = M_maker.M_maker(g,arr_neg1,arr_1,g_deg)
    h = MultiCheb(g_approx.M)
    f_approx = M_maker.M_maker(f,arr_neg1,arr_1,f_deg)
    k = MultiCheb(f_approx.M)

    assert solver_check([f,g],a,b) == True #none multicheb and not neg1_1
    assert solver_check([f,h],a,b) == True #some multicheb and not neg1_1
    assert solver_check([h,k],a,b) == True #all multicheb and not neg1_1
    b = np.ones(2).astype(float)
    a = -1*b
    assert solver_check([f,g],a,b) == True #none multicheb and neg1_1
    assert solver_check([k,g],a,b) == True #some multicheb and neg1_1
    assert solver_check([h,k],a,b) == True #all multicheb and neg1_1

def test_bad_intervals():
    """
    tests to make sure bad intervals get rejected by solve:
    (a) a lower bound is greater than an upper bound
    (b) the bounding arrays are unequal in length
    """
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
    """
    Solve has an "exact" option. 
    This tests that option on test case 2.3 from chebfun2_suite.
    We find the roots using the exact method and non-exact method.
    Then we make sure we got the same roots between the two, and that those roots are correct.
    """
    f = lambda x,y: np.sin(4*(x + y/10 + np.pi/10))
    g = lambda x,y: np.cos(2*(x-2*y+ np.pi/7))
    a,b = np.array([-1,-1]),np.array([1,1])

    funcs = [f,g]
    f_deg, g_deg = 16,32
    guess_degs = [f_deg,g_deg]
    yroots_non_exact = solve(funcs,a,b,guess_degs,exact=False)
    yroots_exact = solve(funcs,a,b,guess_degs,exact=True)

    actual_roots = np.load('Polished_results/polished_2.3.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_2.3.csv', delimiter=',')

    assert len(yroots_non_exact) == len(actual_roots)
    assert len(yroots_exact) == len(actual_roots)
    assert len(yroots_exact) == len(chebfun_roots)

    actual_roots = np.sort(actual_roots, axis=0)
    yroots_non_exact = np.sort(yroots_non_exact, axis=0)
    yroots_exact = np.sort(yroots_exact, axis=0) 
    chebfun_roots = np.sort(chebfun_roots, axis=0) #sort the Roots

    assert np.allclose(yroots_exact,actual_roots)
    assert np.allclose(yroots_exact,chebfun_roots)
    assert np.allclose(yroots_non_exact,actual_roots)
    assert np.allclose(yroots_non_exact,chebfun_roots)

def testreturnBoundingBoxes():
    """
    Solve has an option to return the bounding boxes on the roots. 
    This test makes sure each root lies within their respective box.
    This uses test case "" from chebfun2_suite
    """
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
    """
    Let the search interval be larger than [-1,1]^n.
    Assert that each root is in its respective box.
    This uses test case "" from chebfun2_suite
    """
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
    """
    Checks that the solver gets the correct solver when no guess degree is specified.
    Using test case "" from chebfun2_suite.
    """
    f = lambda x,y: np.sin(4*(x + y/10 + np.pi/10))
    g = lambda x,y: np.cos(2*(x-2*y+ np.pi/7))
    a,b = np.array([-1,-1]),np.array([1,1])

    funcs = [f,g]

    yroots = solve(funcs,a,b)

    actual_roots = np.load('Polished_results/polished_2.3.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_2.3.csv', delimiter=',')

    actual_roots = np.sort(actual_roots, axis=0)
    chebfun_roots = np.sort(chebfun_roots, axis=0) #sort the Roots
    yroots = np.sort(yroots, axis=0) 

    assert np.allclose(yroots,actual_roots)
    assert np.allclose(yroots,chebfun_roots)

# def test_deg_inf():
#     """
#     Tests the logic in Combined_Solver.py that detects which functions are MultiCheb, non-MultiCheb
#     and which functions can be treated like polynomials. This information is used to make smart degree
#     guesses, and the logic used to make the guesses is tested as well.
#     """
#     f = lambda x,y: y**2-x**3
#     g = lambda x,y: (y+.1)**3-(x-.1)**2
#     h = lambda x,y: np.cos(2*(x-2*y+ np.pi/7))
#     a,b = np.array([-1,-1]), np.array([1,1])
#     g_deg = 3
#     g = MultiCheb(M_maker.M_maker(g,a,b,g_deg).M)
#     funcs = [f,g,h]
#     guess_degs = None

#     default_deg = 2
#     is_lambda_poly, is_routine, is_lambda, guess_degs = degree_guesser(funcs,guess_degs,default_deg)

#     assert (is_lambda_poly == np.array([True, False, False])).all()
#     assert (is_routine == np.array([True,False,True])).all()
#     assert (is_lambda == np.array([True,False,True])).all() #TODO:need a test case for python functions with lambda not in the function definition, so is_routine is not is_lambda
#     assert (guess_degs == np.array([3,3,2])).all()

#     #now, a standard test for finding all the roots
