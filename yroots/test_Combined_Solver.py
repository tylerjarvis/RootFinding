"""
A solid 2 dimensional check before I hit the pull request.
"""

#NEED NEW TESTS

import numpy as np
from yroots import ChebyshevSubdivisionSolver, M_maker
#from Combined_Solver import solver

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

f = lambda x,y: (x-1)*(np.cos(x*y**2)+2)
g = lambda x,y: np.sin(8*np.pi*y)*(np.cos(x*y)+2)
f_deg,g_deg = 20,20

def solver_check(a,b):
    funcs = [f,g]
    f_approx = M_maker.M_maker(f,a,b,f_deg)
    print(f_approx.err) #DELETE THIS LINE
    g_approx = M_maker.M_maker(g,a,b,g_deg)
    print(g_approx.err) #DELETE THIS LINE
    guess_degs = [f_deg,g_deg]
    yroots_1 = solver(funcs,a,b,guess_degs)
    yroots_2 = np.array(ChebyshevSubdivisionSolver.solveChebyshevSubdivision([f_approx.M,g_approx.M],np.array([a,b]).T,np.array([f_approx.err,g_approx.err])))
    return np.allclose(yroots_1,yroots_2)

def test_solver():
    a_list = []
    b_list = []
    for i in range(5):
        ax, ay= -1*np.random.random(), -1*np.random.random()
        bx, by = np.random.random(), np.random.random()
        a_list.append(np.array([ax,ay]))
        b_list.append(np.array([bx,by]))

    for a,b in zip(a_list,b_list):
        assert solver_check(a,b) == True