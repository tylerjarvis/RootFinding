import numpy as np
from numba import njit
import itertools
import functools
import yroots.ChebyshevSubdivisionSolver as ChebyshevSubdivisionSolver
import yroots.ChebyshevApproximator as ChebyshevApproximator
from yroots.polynomial import MultiCheb,MultiPower

def _printRootCount(numRoots):
    """Prints how many roots are being returned, closing out the solver's progress marks."""
    finish_string = '\n' + f"Found {numRoots} roots"
    print((finish_string if numRoots != 1 else finish_string[:-1]),end='\n\n')

def solve(funcs,a=-1,b=1, verbose = False, returnBoundingBoxes = False, exact=False, minBoundingIntervalSize=1e-5, max_cpu=1,
          parallel_depth=1):
    """Finds and returns the roots of a system of functions on the search interval [a,b].

    Generates an approximation for each function using Chebyshev polynomials on the interval given,
    then uses properties of the approximations to shrink the search interval. When the information
    contained in the approximation is insufficient to shrink the interval further, the interval is
    subdivided into subregions, and the searching function is recursively called until it zeros in
    on each root. A specific point (and, optionally, a bounding box) is returned for each root found.

    NOTE: YRoots uses just-in-time compiling with an on-disk cache. The first time the solver is called at
    a given dimension on any given install, numba compiles the required specializations (which takes several
    seconds or minutes) and writes them to ``yroots/__pycache__/`` as ``.nbi``/``.nbc`` files. Every later Python
    process that solves a system of that same dimension loads the compiled code from disk on first call
    instead of recompiling. The cache is invalidated automatically when the source file or the numba
    version changes, and it is rebuilt lazily on the next call. Because the cache is keyed by the dimension
    of the system (a new type signature per dimension), the very first solve at each new dimension on a
    fresh install still pays a one-time compile cost.

    After the cache is keyed for a certain dimension, a fresh Python process still pays roughly one second or less of
    import overhead the first time it does ``import yroots`` (for numpy, numba, and the yroots modules
    themselves), and each new dimension adds roughly 30-50 ms to its first solve call for reading the
    cached binaries from disk, linking them, and populating numba's dispatch table. However, both costs are
    per-process rather than per-call, essentially replacing a multi-second/minute warmup with a couple seconds of warmup.

    NOTE: The solve function is only guaranteed to work well on systems of equations where each function
    is continuous and smooth and each root in the interval is a simple root. If a function is not
    continuous and smooth on an interval or an infinite number of roots exist in the interval, the
    solver may get stuck in recursion or the kernel may crash.

    Examples
    --------

    >>> f = lambda x,y,z: 2*x**2 / (x**4-4) - 2*y**2 + .5*z
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
    funcs : list
        List of functions for searching. NOTE: Valid input is restricted to callable Python functions
        (including user-created functions) and yroots Polynomial (MultiCheb and MultiPower) objects.
        String representations of functions are not valid input.
    a : list or numpy array
        An array containing the lower bound of the search interval in each dimension, listed in
        dimension order. If the lower bound is to be the same in each dimension, a single float input
        is also accepted. Defaults to -1 in each dimension if no input is given.
    b : list or numpy array
        An array containing the upper bound of the search interval in each dimension, listed in
        dimension order. If the upper bound is to be the same in each dimension, a single float input
        is also accepted. Defaults to 1 in each dimension if no input is given.
    verbose : bool
        Defaults to False. When True, prints progress of approximation and rootfinding to the terminal.
        Useful for long-running systems.
    returnBoundingBoxes : bool
        Defaults to False. Whether or not to return a precise bounding box for each root.
    exact : bool
        Defaults to False. Whether transformations performed on the approximation should be performed
        with higher precision to minimize error.
    minBoundingIntervalSize : float
        Defaults to 1e-5. If a root is found with a bounding interval of size > minBoundingIntervalSize in
        each dimension, the functions are solved again on the smaller interval. Setting too small could cause
        issues if the functions can't be evaluated accurately on points close together, and will increase solve
        times. Should give more accurate roots when smaller. This number is absolute when the bounding interval in
        question is in [-1,1], and relative otherwise. So if an interval has an endpoint of magnitude > 1, then
        minBoundingIntervalSize is multiplied by that value for that dimension.
    max_cpu : int
        Defaults to 1. Max number of allowed cpus when solving subdivided regions.
    parallel_depth : int
        Defaults to 1. Subdivision depth at which child tasks start being pushed to the worker
        pool. Higher values keep work serial for longer before parallelizing.

    Returns
    -------
    roots : numpy array
        The roots of the system of functions on the interval.
    boundingBoxes : numpy array, optional
        Only returned when ``returnBoundingBoxes`` is True. The exact intervals (boxes) in
        which each root is bound to lie.
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
    macheps = 2**-52
    unit_box = True
    # Check if original region is in the unit box
    if not np.allclose(a,-np.ones_like(a)) or not np.allclose(b,np.ones_like(b)):
        unit_box = False
    # Get an approximation for each function.
    if verbose:
        print("Approximation shapes:", end=" ")

    if not unit_box:
        alphas = (b - a) / 2
        betas = (b + a) / 2

    for i in range(dim):
        if isinstance(funcs[i], MultiPower):
            polys[i] = funcs[i].to_cheb()
            errs[i] = macheps
            if not unit_box:
                polys[i], errs[i] = ChebyshevSubdivisionSolver.transformCheb(polys[i], alphas, betas, errs[i], exact)
        elif isinstance(funcs[i], MultiCheb):
            polys[i] = funcs[i].coeff
            errs[i] = macheps
            if not unit_box:
                polys[i], errs[i] = ChebyshevSubdivisionSolver.transformCheb(polys[i], alphas, betas, errs[i], exact)
        else:
            polys[i], errs[i] = ChebyshevApproximator.chebApproximate(funcs[i],a,b)
        if verbose:
            print(f"{i}: {polys[i].shape}", end = " " if i != dim-1 else '\n')
    if verbose:
        print(f"Searching on interval {[[a[i],b[i]] for i in range(dim)]}")
    #Every point is a root of a function that is identically zero, so there is nothing to isolate.
    for i in range(dim):
        if not np.any(polys[i]):
            raise ValueError(f"Invalid input: function {i} is identically zero on the search interval, "
                             "so every point of the interval solves it.")

    #Solve the Chebyshev polynomial system
    boundingBoxes = ChebyshevSubdivisionSolver.solveChebyshevSubdivision(polys,errs,verbose,exact, constant_check=True,
                low_dim_quadratic_check=True, all_dim_quadratic_check=False, max_cpu=max_cpu, parallel_depth=parallel_depth)
    
    #If the bounding box is the entire interval, subdivide it!
    usingSubdivision = np.all(b-a > minBoundingIntervalSize)
    if len(boundingBoxes) == 1 and np.all(boundingBoxes[0].finalDimSize() == 2) and usingSubdivision:
        #Subdivide the interval and resolve to get better resolution across different parts of the interval
        yroots, boundingBoxes = [], []
        for val in itertools.product([False, True], repeat=len(a)):
            #Split almost in half
            #TODO: Do we need to combine bounding boxes in this step of the recursion as well?
            #      For now it seems safe enough to assume we won't have any roots on the midpoints.
            midPoint = a + (b - a) * 0.51234912839471234
            newA = np.where(val, midPoint, a)
            newB = np.where(val, b, midPoint)
            #Solve recursively
            if verbose:
                print("Re-solving on:", newA, newB)
            roots, boxes = solve(funcs, a=newA, b=newB, verbose=verbose, returnBoundingBoxes=True, exact=exact, minBoundingIntervalSize = minBoundingIntervalSize,
                                 max_cpu=max_cpu, parallel_depth=parallel_depth)
            if len(roots) != 0:
                boundingBoxes.append(boxes)
                yroots.append(roots)
        if len(yroots) > 0:
            yroots = np.vstack(yroots)
            boundingBoxes = np.vstack(boundingBoxes)
        else:
            #Always hand back arrays of the documented shape, even when nothing was found
            yroots = np.empty((0,dim))
            boundingBoxes = np.empty((0,dim,2))
        if verbose:
            _printRootCount(len(yroots))
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
            roots, boxes = solve(funcs, a=newA, b=newB, verbose=verbose, returnBoundingBoxes=True, exact=exact, minBoundingIntervalSize=minBoundingIntervalSize,
                                 max_cpu=max_cpu, parallel_depth=parallel_depth)
            if len(roots) > 0:
                finalRoots.append(roots)
                finalBoxes.append(boxes)
        else:
            #Transform back
            transformedBox = ChebyshevApproximator.transform(box.finalInterval.T,a,b).T
            #Get the roots from this box, and repeat the box once per root it reports, so
            #finalRoots and finalBoxes stay index-aligned. A box that could not separate the
            #roots inside it reports more than one, and each of them gets that same box.
            boxRoots = ChebyshevSubdivisionSolver.getRootsInInterval(box)
            finalRoots.append(ChebyshevApproximator.transform(np.array(boxRoots),a,b))
            finalBoxes.append(np.repeat(transformedBox[np.newaxis], len(boxRoots), axis=0))
    if len(finalBoxes) != 0:
        finalBoxes = np.vstack(finalBoxes)
    else:
        finalBoxes = np.empty((0,dim,2))
    if len(finalRoots) != 0:
        finalRoots = np.vstack(finalRoots)
    else:
        finalRoots = np.empty((0,dim))
    
    # Find and return the roots (and, optionally, the bounding boxes)
    if verbose:
        _printRootCount(len(finalRoots))
    if returnBoundingBoxes:
        return finalRoots, finalBoxes
    else:
        return finalRoots
