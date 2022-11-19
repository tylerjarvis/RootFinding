import numpy as np
import yroots.ChebyshevSubdivisionSolver as ChebyshevSubdivisionSolver 
import yroots.M_maker as M_maker
from yroots.utils import transform
from yroots.polynomial import MultiCheb

def solve(funcs,a,b,guess_degs=None,rescale=False,rel_approx_tol=1.e-15, abs_approx_tol=1.e-12, returnBoundingBoxes = False, exact=False):
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

    if guess_degs == None:
        guess_degs = [20]*len(funcs)

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

    if returnBoundingBoxes:
        yroots, boundingBoxes = np.array(ChebyshevSubdivisionSolver.solveChebyshevSubdivision(funcs,errs,returnBoundingBoxes,exact))
        boundingBoxes = np.array([boundingBox.interval for boundingBox in boundingBoxes])
        if is_neg1_1 == False and len(yroots) > 0:
            yroots = transform(yroots,a,b)
            boundingBoxes = np.array([boundingBox.interval for boundingBox in boundingBoxes])
            boundingBoxes = np.array([transform(boundingBox.T,a,b).T for boundingBox in boundingBoxes]) #xx yy, roots are xy xy each row
        return yroots, boundingBoxes

    else:
        yroots = np.array(ChebyshevSubdivisionSolver.solveChebyshevSubdivision(funcs,errs,returnBoundingBoxes,exact))
        if is_neg1_1 == False and len(yroots) > 0:
            yroots = transform(yroots,a,b)
        return yroots