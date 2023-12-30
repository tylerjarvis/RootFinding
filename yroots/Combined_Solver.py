import numpy as np
from numba import njit
import itertools
import functools
import yroots.ChebyshevSubdivisionSolver as ChebyshevSubdivisionSolver
import yroots.ChebyshevApproximator as ChebyshevApproximator
from yroots.polynomial import MultiCheb

def solve(funcs,a=-1,b=1, verbose = False, returnBoundingBoxes = False, exact=False, minBoundingIntervalSize=1e-5):
    """Finds and returns the roots of a system of functions on the search interval [a,b].

    Generates an approximation for each function using Chebyshev polynomials on the interval given,
    then uses properties of the approximations to shrink the search interval. When the information
    contained in the approximation is insufficient to shrink the interval further, the interval is
    subdivided into subregions, and the searching function is recursively called until it zeros in
    on each root. A specific point (and, optionally, a bounding box) is returned for each root found.

    NOTE: YRoots uses just in time compiling, which means that part of the code will not be compiled until
    a system of functions to solve is given (rather than compiling all the code upon importing the module).
    As a result, the very first time the solver is given any system of equations of a particular dimension,
    the module will take several seconds longer to solve due to compiling time. Once the first system of a
    particular dimension has run, however, other systems of that dimension (or even the same system run
    again) will be solved at the normal (faster) speed thereafter.

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
        String representations of functions are not valid input.
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
    minBoundingIntervalSize : double
        Defaults to 1e-5. If a root is found with a bounding interval of size > minBoundingIntervalSize in
        each dimension, the functions are solved again on the smaller interval. Setting too small could cause
        issues if the functions can't be evaluated accurately on points close together, and will increase solve
        times. Should give more accurate roots when smaller. This number is absolute when the boudning interval in
        question is in [-1,1], and relative otherwise. So if an interval has an endpoint of magnitude > 1, then
        minBoundingIntervalSize is multipled by that value for that dimension.

    Returns
    -------
    yroots : numpy array
        A list of the roots of the system of functions on the interval.
    boundingBoxes : numpy array (optional)
        The exact intervals (boxes) in which each root is bound to lie.
    """
    # Ensure input functions and upper/lower bounds are valid
    if type(funcs) != list and type(funcs) != np.ndarray:
        funcs = [funcs]
    for i in range(len(funcs)):
        if not hasattr(funcs[i], '__call__'):
            raise ValueError(f"Invalid input: input function {i} is not callable")
    dim = len(funcs)
    if type(a) == list:
        a = np.array(a)
    if type(b) == list:
        b = np.array(b)
    if type(a) != np.ndarray:
        a = np.full(dim,a)
    if type(b) != np.ndarray:
        b = np.full(dim,b)
    if len(a) != len(b):
        raise ValueError(f"Invalid input: {len(a)} lower bounds were given but {len(b)} upper bounds were given")
    if (b<a).any():
        raise ValueError(f"Invalid input: at least one lower bound is greater than the corresponding upper bound.")
    polys = np.array(funcs)
    errs = np.array([0.]*dim)

    # Get an approximation for each function.
    if verbose:
        print("Approximation shapes:", end=" ")
    for i in range(dim):
        polys[i], errs[i] = ChebyshevApproximator.chebApproximate(funcs[i],a,b)
        if verbose:
            print(f"{i}: {polys[i].shape}", end = " " if i != dim-1 else '\n')
    if verbose:
        print(f"Searching on interval {[[a[i],b[i]] for i in range(dim)]}")
        
    #Solve the Chebyshev polynomial system
    yroots, boundingBoxes = ChebyshevSubdivisionSolver.solveChebyshevSubdivision(polys,errs,verbose,True,exact,
                constant_check=True, low_dim_quadratic_check=True, all_dim_quadratic_check=False)
    
    #If the bounding box is the entire interval, subdivide it!
    usingSubdivision = np.all(b-a > minBoundingIntervalSize)
    if len(boundingBoxes) == 1 and np.all(boundingBoxes[0].finalDimSize() == 2) and usingSubdivision:
        #Subdivide the interval and resolve to get better resolution across different parts of the interval
        yroots, boundingBoxes = [], []
        for val in itertools.product([False, True], repeat=len(a)):
            #Split almost in half
            #TODO: Do we need to combine bounding boxes in this step of the recursion as well?
            #      For now it seems safe enough to assume we won't have any roots on the midpoints.
            midPoint = (a + b) * 0.51234912839471234
            newA = np.where(val, midPoint, a)
            newB = np.where(val, b, midPoint)
            #Solve recursively
            if verbose:
                print("Re-solving on:", newA, newB)
            roots, boxes = solve(funcs, a=newA, b=newB, verbose=verbose, returnBoundingBoxes=True, exact=exact, minBoundingIntervalSize = minBoundingIntervalSize)
            if len(roots) != 0:
                boundingBoxes.append(boxes)
                yroots.append(roots)
        if len(yroots) > 0:
            yroots = np.vstack(yroots)
            boundingBoxes = np.vstack(boundingBoxes)
        if returnBoundingBoxes:
            return yroots, boundingBoxes
        else:
            return yroots
    
    #TODO: Handle if we have duplicate roots or extra roots at the top level. Easiest if we actually return the bounding boxes!
    #Maybe return the bounding boxes in the recursive steps?
    
    #If any of the bounding boxes is too large, re-solve that box.
    finalBoxes = []
    finalRoots = []
    for box in boundingBoxes:
        #Get the relative max size in each dimension. If a or b > 1 in magnitude, minBoundingIntervalSize is a relative number.
        #If they are < 1 in magnitude, it is an absolute number.
        newA, newB = ChebyshevApproximator.transform(box.finalInterval.T,a,b)
        relMaxSize = minBoundingIntervalSize * functools.reduce(np.maximum, [np.abs(a),np.abs(b), 1])
        if np.all(newB - newA > relMaxSize):
            #Re-solve this box
            if verbose:
                print("Re-solving on:", newA, newB)
            roots, boxes = solve(funcs, a=newA, b=newB, verbose=verbose, returnBoundingBoxes=True, exact=exact, minBoundingIntervalSize = minBoundingIntervalSize)
            if len(roots) > 0:
                finalRoots.append(roots)
                finalBoxes.append(boxes)
        else:
            #Transform back
            finalBoxes.append([ChebyshevApproximator.transform(box.finalInterval.T,a,b).T])
            #Get the roots from this box
            if len(box.possibleDuplicateRoots) > 0:
                finalRoots.append(ChebyshevApproximator.transform(np.array(box.possibleDuplicateRoots),a,b))
            else:
                finalRoots.append(ChebyshevApproximator.transform(box.getFinalPoint(),a,b))
    if len(finalBoxes) != 0:
        finalBoxes = np.vstack(finalBoxes)
    if len(finalRoots) != 0:
        finalRoots = np.vstack(finalRoots)
    
    # Find and return the roots (and, optionally, the bounding boxes)
    if returnBoundingBoxes:
        return finalRoots, finalBoxes
    else:
        return finalRoots
