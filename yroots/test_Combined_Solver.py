"""
A solid 2 dimensional check before I hit the pull request.
"""
import numpy as np
import M_maker
import ChebyshevSubdivisionSolver
import pytest
from polynomial import MultiCheb
from utils import transform

f = lambda x,y: (x-1)*(np.cos(x*y**2)+2)
g = lambda x,y: np.sin(8*np.pi*y)*(np.cos(x*y)+2)
f_deg,g_deg = 20,20

def solver(funcs,a,b,guess_degs,rescale=False,rel_approx_tol=1.e-15, abs_approx_tol=1.e-12,exact=False):
    """
    Finds the roots of the system of functions

    parameters
    ----------
    funcs: list
    list of the vectorized functions (R^n --> R)
    a: ndarray
    lower bound on the search interval
    b: ndarray
    upper bound on the search interval
    guess_degs: list
    guess of the best approximation degree for each function
    rescale: bool
    whether to rescale the approximation by inf_norm or not
    rel_approx_tol: float
    relative approximation tolerance
    abs_approx_tol: float
    absolute approximation tolerance

    returns
    -------
    ndarray:
    the yroots of the system of functions
    """
    #TODO: allow for a,b to deafult to ones and negative ones
    #TODO: handle case for when input degree is less than the approximation degree that was used
    #TODO: decide whether to have the guess_deg input default, and what it would be (could the approximation degree used work), maybe they need to know their degree
    #TODO: handle for case that input degree is above max_deg (provide a warning)
    #TODO: maybe next we can include an option to return the bounding boxes

    if len(a) != len(b):
        raise ValueError("Dimension mismatch in intervals.")
    
    if (b<=a).any():
        raise ValueError("At least one lower bound is >= an upper bound.")
    
    is_neg1_1 = True
    arr_neg1 = np.array([-1]*len(a)) #what if a>b
    arr_1 = np.ones(len(a))

    if np.allclose(arr_neg1,a,rtol=1e-08) and np.allclose(arr_1,b,rtol=1e-08):
        pass
    else:
        is_neg1_1 = False
    
    is_multi_cheb_arr = []

    for func in funcs: #USE
        if isinstance(func,MultiCheb):
            is_multi_cheb_arr.append(True)
            pass
        else:
            is_multi_cheb_arr.append(False)

    is_multi_cheb_arr = np.array(is_multi_cheb_arr)
    funcs = np.array(funcs)

    MultiCheb_idxs = list(np.where(is_multi_cheb_arr==1)[0])
    non_MultiCheb_idxs = list(np.where(is_multi_cheb_arr==0)[0])

    errs = np.array([0]*len(funcs))

    for idx in non_MultiCheb_idxs:
        approx = M_maker.M_maker(funcs[idx],arr_neg1,arr_1,guess_degs[idx],rel_approx_tol,abs_approx_tol)
        if rescale:
            funcs[idx] = MultiCheb(approx.M_rescaled)
        else:
            funcs[idx] = MultiCheb(approx.M)
        errs[idx] = approx.err

    for idx in MultiCheb_idxs:
        approx = M_maker.M_maker(funcs[idx],arr_neg1,arr_1,guess_degs[idx],rel_approx_tol,abs_approx_tol)
        if rescale:
            funcs[idx] = MultiCheb(approx.M_rescaled)
        else:
            funcs[idx] = MultiCheb(approx.M)

    funcs = [func.coeff for func in funcs]
    yroots = np.array(ChebyshevSubdivisionSolver.solveChebyshevSubdivision(funcs,errs,exact))

    #transform doesn't work on empty arrays
    if is_neg1_1 == False and len(yroots) > 0:
        yroots = transform(yroots,a,b)

    return yroots

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


    # gotta sort these
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


#WHAT CAN WE TEST ABOUT THIS CODE
#WE CAN CHECK THAT IT PRESERVES WHAT ERIKs solver does when it is given the approximations
    #CASES
    #not neg1_1 and not all multicheb
    #neg1_1 and not all multicheb
    #not neg1_1 and all multicheb
    #both neg1_1 and all multicheb

    #value error check for a and b

