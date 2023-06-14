import numpy as np
import inspect
import sympy as sy
import yroots.ChebyshevSubdivisionSolver as ChebyshevSubdivisionSolver
# import ChebyshevSubdivisionSolver
import yroots.M_maker as M_maker
# import M_maker
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

    # Set guess_degs to the default if not provided
    if guess_degs is None:
        guess_degs = np.array([default_deg]*len(funcs))

    # Find and mark all lambda functions
    is_lambda_poly = np.array([True]*len(funcs))
    is_routine = np.array([inspect.isroutine(func) for func in funcs]) #is it a python routine?
    is_lambda = is_routine #assumption: all routines are lambdas. Next loop checks this assumption.
    if sum(is_routine) > 0: #if at least one func is a routine
        routine_mask = is_routine == 1
        routine_true_idxs = np.where(routine_mask == True)[0]
        funcs_routines = np.array([funcs[idx] for idx in routine_true_idxs]) #indices of funcs that are routines
        lambda_mask = np.array([True if "lambda" in inspect.getsource(func) else False for func in funcs_routines])
        is_lambda[routine_mask][~lambda_mask] = 0 # Set is_lambda false for all funcs that are routines but do not contain "lambda"

    # Keeps track of how many lambda functions are typed directly as parameters into the call to solve.
    lambda_counter = 0

    for i,func in enumerate(funcs):
        if isinstance(func,MultiCheb) or isinstance(func,MultiPower):
            guess_degs[i] = max(func.shape) - 1 #funcs[i].shape[0] might suffice
            is_lambda_poly[i] = False
        elif is_lambda[i]:
            f_str_lst = inspect.getsource(func).strip().split(":")
            # If the source is the call to the solve function, the source will have 'solve' in it
            # and will be one long line. Thus the splitting functions differently.
            if "solve" in f_str_lst[0]:
                vars, expr = f_str_lst[lambda_counter].strip().split('lambda')[1].strip(), f_str_lst[lambda_counter+1].strip().split(',')[0]
                lambda_counter += 1
                # The last func will have the rest of the code used to call the solve function tacked on;
                # delete this by searching for the closing ']'
                if ']' in expr:
                    expr = expr.split(']')[0]
            else: # the function was defined outside of the call to solve and can be split as normal  
                vars, expr = f_str_lst[0].strip().split('lambda')[1].strip(), f_str_lst[1].strip().split(',')[0]
            vars = sy.symbols(vars)
            if "np." in expr:
                is_lambda_poly[i] = False #not a great check, since polynomials can be expressed with np.array(), but good start
                #problem: this includes rational polynomials, maybe include "/" in the string search
                #don't update guess_degs[i]
            else:
                expr = sy.sympify(expr)
                guess_degs[i] = max(sy.degree_list(expr))
    return [is_lambda_poly, is_routine, is_lambda, guess_degs]

def solve(funcs,a=[],b=[],guess_degs=None,max_deg_edit=None,rescale=False,rel_approx_tol=1.e-15, abs_approx_tol=1.e-12, 
          returnBoundingBoxes = False, exact=False, constant_check = True, low_dim_quadratic_check = True,
          all_dim_quadratic_check = False):
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
    constant_check : bool
        Defaults to True. Whether or not to run constant term check after each subdivision. Testing indicates
        that this saves time in all dimensions.
    low_dim_quadratic_check : bool
        Defaults to True. Whether or not to run quadratic check in dimensions two and three. Testing indicates
        That this saves a lot of time compared to not running it in low dimensions.
    all_dim_quadratic_check : bool
        Defaults to False. Whether or not to run quadratic check in every dimension. Testing indicates it loses
        time in 4 or higher dimensions.

    returns
    -------
    ndarray:
    the yroots of the system of functions
    """

    #TODO: handle case for when input degree is less than the approximation degree that was used
    #TODO: optimize guess_deg DOING
    #TODO: handle for case that input degree is above max_deg (provide a warning)
    #TODO: decide whether to move degree_guesser --> utils.py

    #Initialize a, b to negative ones and ones if no argument passed in
    if a == []:
        a = -np.ones(len(funcs))
    if b == []:
        b = np.ones(len(funcs))
    # Convert any given bounds a, b into np.array format
    if type(a) == list:
        a = np.array(a)
    if type(b) == list:
        b = np.array(b)
    if type(a) != np.array:
        a = np.full(len(funcs),a)
    if type(b) != np.array:
        b = np.full(len(funcs),b)
    #Ensure inputs a and b are valid for the problem.
    if len(a) != len(b):
        raise ValueError("Dimension mismatch in intervals.")
    if (b<=a).any():
        raise ValueError("At least one lower bound is >= an upper bound.")
    
    funcs = np.array(funcs) # change format of funcs to np.array
    errs = np.array([0.]*len(funcs)) # Initialize array to keep track of errors
    default_deg = 2 #the default for the guess degrees
    guess_degs = degree_guesser(funcs,guess_degs,default_deg)[3]
    
    #Check to see if the bounds are [-1,1]^n
    is_neg1_1 = True
    arr_neg1 = -np.ones(len(a))
    arr_1 = np.ones(len(a))
    if not np.allclose(arr_neg1,a,rtol=1e-08) or not np.allclose(arr_1,b,rtol=1e-08):
        is_neg1_1 = False

    # Get the indices of any MultiCheb functions and those that are not MultCheb
    is_multi_cheb_arr = []
    for func in funcs: #Mark True if MultiCheb, False otherwise
        if isinstance(func,MultiCheb):
            is_multi_cheb_arr.append(True)
            pass
        else:
            is_multi_cheb_arr.append(False)
    is_multi_cheb_arr = np.array(is_multi_cheb_arr)
    MultiCheb_idxs = list(np.where(is_multi_cheb_arr==1)[0])
    non_MultiCheb_idxs = list(np.where(is_multi_cheb_arr==0)[0])

    # Get a coefficient matrix for each function in funcs from M_maker.py
    for idx in non_MultiCheb_idxs:
        approx = M_maker.M_maker(funcs[idx],a,b,guess_degs[idx],max_deg_edit,rel_approx_tol,abs_approx_tol)
        if rescale:
            funcs[idx] = MultiCheb(approx.M_rescaled)
        else:
            funcs[idx] = MultiCheb(approx.M)
        errs[idx] = approx.err
    for idx in MultiCheb_idxs: #future: could we skip M_Maker process if we knew that it matched the [-1,1]^n interval?
        approx = M_maker.M_maker(funcs[idx],a,b,guess_degs[idx],max_deg_edit,rel_approx_tol,abs_approx_tol)
        if rescale:
            funcs[idx] = MultiCheb(approx.M_rescaled)
        else:
            funcs[idx] = MultiCheb(approx.M)

    funcs = [func.coeff for func in funcs] #Each func in funcs is now a coefficient matrix

    #Solve the problem using ChebyshevSubdivisionSolver.py
    # If there are roots and the interval given is not [-1,1]^n,
    # transform roots and bounding boxes back to the a,b interval
    # Return the roots (and bounding boxes if desired)
    if returnBoundingBoxes:
        yroots, boundingBoxes = ChebyshevSubdivisionSolver.solveChebyshevSubdivision(funcs,errs,returnBoundingBoxes,exact,
                                         constant_check=constant_check, low_dim_quadratic_check=low_dim_quadratic_check,
                                         all_dim_quadratic_check=all_dim_quadratic_check)
        boundingBoxes = np.array([boundingBox.interval for boundingBox in boundingBoxes])
        if is_neg1_1 == False and len(yroots) > 0: 
            yroots = transform(yroots,a,b)
            boundingBoxes = np.array([boundingBox.interval for boundingBox in boundingBoxes])
            boundingBoxes = np.array([transform(boundingBox.T,a,b).T for boundingBox in boundingBoxes]) #xx yy, roots are xy xy each row
        return yroots, boundingBoxes
    else:
        yroots = ChebyshevSubdivisionSolver.solveChebyshevSubdivision(funcs,errs,returnBoundingBoxes,exact,constant_check=constant_check,                                                                        low_dim_quadratic_check=low_dim_quadratic_check,                                                                    all_dim_quadratic_check=all_dim_quadratic_check)
        if is_neg1_1 == False and len(yroots) > 0:
            yroots = transform(yroots,a,b)
        return yroots
        