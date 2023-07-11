import numpy as np
import yroots.ChebyshevSubdivisionSolver as ChebyshevSubdivisionSolver
import yroots.ChebyshevApproximator as ChebyshevApproximator
from yroots.utils import transform
from yroots.polynomial import MultiCheb, MultiPower

def solve(funcs,a=[],b=[],guess_degs=None,max_n_coeffs_edit=None,rescale=False,rel_approx_tol=1.e-15, abs_approx_tol=1.e-12, 
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
    if type(a) != np.ndarray:
        a = np.full(len(funcs),a)
    if type(b) != np.ndarray:
        b = np.full(len(funcs),b)
    #Ensure inputs a and b are valid for the problem.
    if len(a) != len(b):
        raise ValueError("Dimension mismatch in intervals.")
    if (b<=a).any():
        raise ValueError("At least one lower bound is >= an upper bound.")
    
    funcs = np.array(funcs) # change format of funcs to np.array
    errs = np.array([0.]*len(funcs)) # Initialize array to keep track of errors
    
    #Check to see if the bounds are [-1,1]^n
    is_neg1_1 = True
    arr_neg1 = -np.ones(len(a))
    arr_1 = np.ones(len(a))
    if not np.allclose(arr_neg1,a,rtol=1e-08) or not np.allclose(arr_1,b,rtol=1e-08):
        is_neg1_1 = False

    # Get an approximation for each function.
    for i in range(len(funcs)):
        funcs[i], errs[i] = ChebyshevApproximator.chebApproximate(funcs[i],a,b)

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
        