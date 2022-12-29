import numpy as np
import inspect
import sympy as sy
import yroots.ChebyshevSubdivisionSolver as ChebyshevSubdivisionSolver 
import yroots.M_maker as M_maker
from yroots.utils import transform
from yroots.polynomial import MultiCheb, MultiPower

def degree_guesser(funcs,guess_degs,default_deg):
    """
    parameters
    ----------
    funcs: list
    list of the vectorized functions (R^n --> R)
    guess_degs: list
    guess of the best approximation degree for each function

    returns
    -------
    (list) smarter guesses about the degree to approximate with
    """
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
    return [is_lambda_poly, is_routine, is_lambda, guess_degs]

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
    #TODO: allow for a,b to default to ones and negative ones
    #TODO: handle case for when input degree is less than the approximation degree that was used
    #TODO: optimize guess_deg DOING
    #TODO: handle for case that input degree is above max_deg (provide a warning)
    #TODO: decide whether to move degree_guesser --> utils.py
    default_deg = 2 #the default for the guess degrees
    guess_degs = degree_guesser(funcs,guess_degs,default_deg)[3]

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

    for idx in MultiCheb_idxs: #future: could we skip M_Maker process if we knew that it matched the [-1,1]^n interval?
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