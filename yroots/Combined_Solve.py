import numpy as np
from yroots import ChebyshevSubdivisionSolver, M_maker

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
    the yroots of hthe system of functions
    """
    approximations = []
    errs = []
    for f,deg in zip(funcs,guess_degs):
        approx = M_maker.M_maker(f,a,b,deg,rel_approx_tol,abs_approx_tol)
        if rescale == True:
            approximations.append(approx.M_rescaled)
        else:
            approximations.append(approx.M)
        errs.append(approx.err)
    errs = np.array(errs)
    yroots = np.array(ChebyshevSubdivisionSolver.solveChebyshevSubdivision(approximations,np.array([a,b]).T,errs))
    return yroots