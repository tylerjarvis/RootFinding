import numpy as np
import ChebyshevSubdivisionSolver, M_maker
from utils import transform
from polynomial import MultiCheb

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
    
    if (b>=a).any():
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
    yroots = np.array(ChebyshevSubdivisionSolver.solveChebyshevSubdivision(funcs,errs))

    #transform doesn't work on empty arrays
    if is_neg1_1 == False and len(yroots) > 0:
        yroots = transform(yroots,a,b)

    return yroots