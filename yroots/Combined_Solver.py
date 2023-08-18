import numpy as np
from numba import njit
import yroots.ChebyshevSubdivisionSolver as ChebyshevSubdivisionSolver
import yroots.ChebyshevApproximator as ChebyshevApproximator

def solve(funcs,a=-1,b=1, verbose = False, returnBoundingBoxes = False, exact=False, constant_check = True,
          low_dim_quadratic_check = True, all_dim_quadratic_check = False):

    """Finds and returns the roots of a system of functions on the search interval [a,b].

    Generates an approximation for each function using Chebyshev polynomials on the interval given,
    then uses properties of the approximations to shrink the search interval. When the information
    contained in the approximation is insufficient to shrink the interval further, the interval is
    subdivided into subregions, and the searching function is recursively called until it zeros in
    on each root. A specific point (and, optionally, a bounding box) is returned for each root found.

    NOTE: The solve function is only guaranteed to work well on systems of equations where each function
    is continuous and smooth and each root in the interval is a simple root. If a function is not
    continuous and smooth on an interval or an infinite number of roots exist in the interval, the
    solver may get stuck in recursion or the kernel may crash.

    Examples
    --------

    >>> f = lambda x,y,z: 2*x**2 / (x**4-4) - 2*x**2 + .5
    >>> g = lambda x,y,z: 2*x**2*y / (y**2+4) - 2*y + 2*x*z
    >>> h = lambda x,y,z: 2*z / (z**2-4) - 2*z
    >>> roots = yroots.solve([f, g, h], np.array([-0.5,0,-2**-2.44]), np.array([0.5,np.exp(1.1376),.8]))
    >>> print(roots)
    [[-4.46764373e-01  4.44089210e-16 -5.55111512e-17]
     [ 4.46764373e-01  4.44089210e-16 -5.55111512e-17]]
    


    >>> M1 = yroots.MultiPower(np.array([[0,3,0,2],[1.5,0,7,0],[0,0,4,-2],[0,0,0,1]]))
    >>> M2 = yroots.MultiCheb(np.array([[0.02,0.31],[-0.43,0.19],[0.06,0]]))
    >>> roots = yroots.solve([M1,M2],-5,5)
    >>> print(roots)
    [[-0.98956615 -4.12372817]
     [-0.06810064  0.03420242]]

    Parameters
    ----------
    funcs: list
        List of functions for searching. NOTE: Valid input is restricted to callable Python functions
        (including user-created functions) and yroots Polynomial (MultiCheb and MultiPower) objects.
        String representations of functions are not valid input. See examples below.
    a: list or numpy array
        An array containing the lower bound of the search interval in each dimension, listed in
        dimension order. If the lower bound is to be the same in each dimension, a single float input
        is also accepted. Defaults to -1 in each dimension if no input is given.
    b: list or numpy array
        An array containing the upper bound of the search interval in each dimension, listed in
        dimension order. If the upper bound is to be the same in each dimension, a single float input
        is also accepted. Defaults to 1 in each dimension if no input is given.
    verbose : bool
        Defaults to False. Tracks progress of the approximation and rootfinding by outputting progress to
        the terminal. Useful in tracking progress of systems of equations that take a long time to solve.
    returnBoundingBoxes : bool
        Defaults to False. Whether or not to return a precise bounding box for each root.
    exact: bool
        Defaults to False. Whether transformations performed on the approximation should be performed
        with higher precision to minimize error.
    constant_check : bool
        Defaults to True. Whether or not to run the constant term check to possibly eliminate an
         entire subdivision interval after each subdivision.
    low_dim_quadratic_check : bool
        Defaults to True. Whether or not to run the quadratic term check to possibly eliminate an
         entire subdivision interval after each subdivision in dimensions 2 and 3.
    all_dim_quadratic_check : bool
        Defaults to False. Whether or not to run the quadratic term check to possibly eliminate an
         entire subdivision interval after each subdivision in dimensions greater than 3.

    Returns
    -------
    yroots : numpy array
        A list of the roots of the system of functions on the interval.
    boundingBoxes : numpy array (optional)
        The exact intervals (boxes) in which each root is bound to lie.
    """

    # Set up problem and ensure input functions and upper/lower bounds are valid
    if type(funcs) != list:
        funcs = [funcs]
    for i in range(len(funcs)):
        if not hasattr(funcs[i], '__call__'):
            raise ValueError(f"Invalid input: input function {i} is not callable")
    if type(a) == list:
        a = np.array(a)
    if type(b) == list:
        b = np.array(b)
    if type(a) != np.ndarray:
        a = np.full(len(funcs),a)
    if type(b) != np.ndarray:
        b = np.full(len(funcs),b)
    if len(a) != len(b):
        raise ValueError(f"Invalid input: {len(a)} lower bounds were given but {len(b)} upper bounds were given")
    if (b<a).any():
        raise ValueError(f"Invalid input: at least one lower bound is greater than the corresponding upper bound.")
    funcs = np.array(funcs)
    errs = np.array([0.]*len(funcs))
    
    # Check to see if the bounds are [-1,1]^n (if so, no final transformation will be needed)
    is_neg1_1 = True
    arr_neg1 = -np.ones(len(a))
    arr_1 = np.ones(len(a))
    if not np.allclose(arr_neg1,a,rtol=1e-08) or not np.allclose(arr_1,b,rtol=1e-08):
        is_neg1_1 = False

    # Get an approximation for each function.
    if verbose:
        print("Approximation shapes:", end=" ")
    for i in range(len(funcs)):
        funcs[i], errs[i] = ChebyshevApproximator.chebApproximate(funcs[i],a,b)
        if verbose:
            print(f"{i}: {funcs[i].shape}", end = " " if i != len(funcs)-1 else '\n')
    if verbose:
        print(f"Searching on interval {[[a[i],b[i]] for i in range(len(a))]}")

       
    # Find and return the roots (and, optionally, the bounding boxes)
    if returnBoundingBoxes:
        yroots, boundingBoxes = ChebyshevSubdivisionSolver.solveChebyshevSubdivision(funcs,errs,verbose,returnBoundingBoxes,exact,
                constant_check=constant_check, low_dim_quadratic_check=low_dim_quadratic_check,
                all_dim_quadratic_check=all_dim_quadratic_check)
        boundingBoxes = np.array([boundingBox.interval for boundingBox in boundingBoxes])
        if is_neg1_1 == False and len(yroots) > 0: 
            yroots = ChebyshevApproximator.transform(yroots,a,b)
            boundingBoxes = np.array([ChebyshevApproximator.transform(boundingBox.T,a,b).T for boundingBox in boundingBoxes]) #xx yy, roots are xy xy each row
        return yroots, boundingBoxes
    else:
        yroots = ChebyshevSubdivisionSolver.solveChebyshevSubdivision(funcs,errs,verbose,returnBoundingBoxes,exact,
                constant_check=constant_check, low_dim_quadratic_check=low_dim_quadratic_check,
                all_dim_quadratic_check=all_dim_quadratic_check)
        if is_neg1_1 == False and len(yroots) > 0:
            yroots = ChebyshevApproximator.transform(yroots,a,b)
        return yroots