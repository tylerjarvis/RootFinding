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

def solver(funcs,a,b,guess_degs,rescale=False,rel_approx_tol=1.e-15, abs_approx_tol=1.e-12):
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
    #TODO: allow for a,b to default to neg1_1, require input dim? it's tedious (-), it's a good sanity check (+)
    #handle for when input deg is less than the approximation degree used to build that Multicheb object
    #guess deg input default
    #maybe the SHOULD know what degree to input
    #handle for when the input deg is too high
    #handle for when there is no input deg
    if len(a) != len(b):
        raise ValueError("Dimension mismatch in intervals.")
    
    if (a>=b).any():
        raise ValueError("At least one lower bound is >= an upper bound.")
    
    is_neg1_1 = True
    arr_neg1 = np.array([-1]*len(a))
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
        errs[idx] = approx.err

    funcs = [func.coeff for func in funcs]
    yroots = np.array(ChebyshevSubdivisionSolver.solveChebyshevSubdivision(funcs,errs))

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

#WHAT CAN WE TEST ABOUT THIS CODE
#WE CAN CHECK THAT IT PRESERVES WHAT ERIKs solver does when it is given the approximations
    #CASES
    #not neg1_1 and not all multicheb
    #neg1_1 and not all multicheb
    #not neg1_1 and all multicheb
    #both neg1_1 and all multicheb

    #value error check for a and b