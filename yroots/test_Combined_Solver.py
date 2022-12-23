"""
A solid 2 dimensional check before I hit the pull request.
"""
import numpy as np
import yroots.M_maker as M_maker
import yroots.ChebyshevSubdivisionSolver as ChebyshevSubdivisionSolver
import pytest
from yroots.polynomial import MultiCheb, MultiPower
from yroots.utils import transform
from yroots.Combined_Solver import solve
import inspect
import sympy as sy

f = lambda x,y: (x-1)*(np.cos(x*y**2)+2)
g = lambda x,y: np.sin(8*np.pi*y)*(np.cos(x*y)+2)
f_deg,g_deg = 20,20

def solver_check(funcs,a,b):
    """
    raw functions on [-1,1]^n
    """

    f,g = funcs
    yroots_1 = solve(funcs,a,b)
    print("great")

    arr_neg1 = np.array([-1]*len(a)) #what if a>b
    arr_1 = np.ones(len(a))

    #hate to do this
    f_approx = M_maker.M_maker(f,arr_neg1,arr_1,f_deg)
    g_approx = M_maker.M_maker(g,arr_neg1,arr_1,g_deg)
    
    #bug with multicheb objects
    print("trying this guy")
    yroots_2 = np.array(ChebyshevSubdivisionSolver.solveChebyshevSubdivision([f_approx.M,g_approx.M],np.array([f_approx.err,g_approx.err])))
    if len(yroots_2) > 0: #transform doesn't work on empty arrays
        yroots_2 = transform(yroots_2,a,b)

    return np.allclose(yroots_1,yroots_2)

#TODO: can I delete all the stuff below? find out where it is taken care of
#TODO: need docstrings for tests

#CASES
    #not neg1_1 and not all multicheb
    #neg1_1 and not all multicheb
    #not neg1_1 and all multicheb
    #both neg1_1 and all multicheb

def test_solver():
    a = -1*np.random.random(2)
    b = np.random.random(2)
    arr_neg1 = np.array([-1]*len(a)) #what if a>b
    arr_1 = np.ones(len(a))

    #TODO: multicheb object on the correct domain
    print("now approximating the first one")
    g_approx = M_maker.M_maker(g,arr_neg1,arr_1,g_deg)
    print("now approximating the second one")
    h = MultiCheb(g_approx.M)
    f_approx = M_maker.M_maker(f,arr_neg1,arr_1,f_deg)
    k = MultiCheb(f_approx.M)

    print("first 3 tests")
    # assert solver_check([f,g],a,b) == True #none multicheb and neg1_1
    # print("test 1")
    assert solver_check([f,k],a,b) == True #some multicheb and neg1_1
    print("test 2")
    # assert solver_check([h,k],a,b) == True #all multicheb and neg1_1
    # print("test 3")
    b = np.ones(2).astype(float)
    a = -1*b
    print("second 3 tests")
    # assert solver_check([f,g],a,b) == True #none multicheb and not neg1_1
    # print("test 1")
    assert solver_check([h,g],a,b) == True #some multicheb and not neg1_1
    print("test 2")
    # assert solver_check([h,k],a,b) == True #all multicheb and not neg1_1
    # print("test 3")

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
    yroots_non_exact = solve(funcs,a,b,guess_degs,exact=False)
    yroots_exact = solve(funcs,a,b,guess_degs,exact=True)

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

def test_deg_inf():
    f = lambda x,y: y**2-x**3
    g = lambda x,y: (y+.1)**3-(x-.1)**2
    h = lambda x,y: np.cos(2*(x-2*y+ np.pi/7))
    a,b = np.array([-1,-1]), np.array([1,1])
    g_deg = 3
    g = MultiCheb(M_maker.M_maker(g,a,b,g_deg).M)
    funcs = [f,g,h]
    guess_degs = None

    ###THIS IS A DIRECT COPY PASTE FROM Combined_Solver.py. 
    ### START: ###
    default_deg = 2
    if guess_degs == None:
        guess_degs = np.array([default_deg]*len(funcs))

    is_lambda_poly = np.array([True]*len(funcs)) #keeps track of code
    is_routine = np.array([inspect.isroutine(func) for func in funcs]) #is it a python routine?
    is_lambda = is_routine #assumption: all routines are lambdas

    if sum(is_routine) > 0: #if at least one func is a routine
        routine_mask = is_routine == 1 #evaluate assumption
        routine_true_idxs = np.where(routine_mask == True)[0]
        funcs_routines = np.array([funcs[idx] for idx in routine_true_idxs]) #funcs that are routines
        #idxs of funcs that are lamba routines
        lambda_mask = np.array([True if "lambda" in inspect.getsource(func) else False for func in funcs_routines])
        is_lambda[routine_mask][~lambda_mask] = 0 #update assumption where necessary

    for i,func in enumerate(funcs):
        if isinstance(func,MultiCheb) or isinstance(func,MultiPower):
            guess_degs[i] = max(func.shape) - 1 #funcs[i].shape[0] might suffice
            is_lambda_poly[i] = False
        elif is_lambda[i]:
            f_str_lst = inspect.getsource(func).strip().split(":")
            vars, expr = f_str_lst[0].strip().split('lambda')[1].strip(), f_str_lst[1].strip()
            vars = sy.symbols(vars)
            if "np." in expr:
                is_lambda_poly[i] = False #not a great check, since polynomials can be expressed with np.array(), but good start
                #problem: this includes rational polynomials, maybe include "/" in the string search
                #don't update guess_degs[i]
            else:
                expr = sy.sympify(expr)
                guess_degs[i] = max(sy.degree_list(expr))
    ### END. ###

    assert (is_lambda_poly == np.array([True, False, False])).all()
    assert (is_routine == np.array([True,False,True])).all()
    assert (is_lambda == np.array([True,False,True])).all() #TODO:need a test case for python functions with lambda not in the function definition, so is_routine is not is_lambda
    assert (guess_degs == np.array([3,3,2])).all()

    #now, a standard test for finding all the roots
