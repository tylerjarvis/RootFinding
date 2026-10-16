import numpy as np
from numba import njit
import itertools
import functools
import warnings
import yroots.ChebyshevSubdivisionSolver as ChebyshevSubdivisionSolver
import yroots.ChebyshevApproximator as ChebyshevApproximator
from yroots.polynomial import MultiCheb,MultiPower

def _printRootCount(numRoots):
    """Prints how many roots are being returned, closing out the solver's progress marks."""
    finish_string = '\n' + f"Found {numRoots} roots"
    print((finish_string if numRoots != 1 else finish_string[:-1]),end='\n\n')

def relativeRoundingError(coeff, macheps=2**-52):
    """Bound the error of a polynomial given exactly as coefficients: rounding in those coefficients.

    Each coefficient is off by at most macheps times its own size, so the polynomial is off by at most
    macheps times the sum of their sizes anywhere on [-1,1]^n. A fixed macheps instead is only right
    when the coefficients are of order 1: scale them down and it becomes a large fraction of them --
    about 2% at 1e-14, more than all of them from 1e-16 down -- and the solver no longer finishes.

    The identically zero polynomial keeps the fixed macheps. Its relative error is 0, which makes the
    solver report no roots for a polynomial that vanishes everywhere.
    """
    absSum = np.sum(np.abs(coeff))
    return macheps*absSum if absSum > 0 else macheps

#A root whose final box is wider than this, relative to the size of the search interval's bounds in that
#dimension (as minBoundingIntervalSize is), converged the slow way an ill-conditioned root does; a simple
#root's box ends up around 1e-13. Such a box is solved again on a padded neighborhood of itself.
REFINE_BOX_SIZE = 1e-10
#The neighborhood re-solved around a wide box reaches this many of the box's widths past it on each side.
REFINE_PADDING = 2
#An approximation on that neighborhood may not need a degree above this. A smooth function needs only a
#handful of coefficients on so small an interval. One whose rounding error is large next to its values
#there (1 - cos(y) near 0, where the cancellation happens inside the function) never converges and
#keeps doubling its degree, so this is where refinement gives up and the box keeps what it had.
REFINE_MAX_DEGREE = 100

def _warnDuplicateRoots(dupSets):
    """Warns once about every set of roots that could not be told apart from each other.

    Parameters
    ----------
    dupSets : list of numpy array
        One (k, dim) array per bounding box that reports k > 1 roots, in the original coordinates.
    """
    if len(dupSets) == 0:
        return
    def formatRoot(root):
        return "(" + ", ".join(f"{x:.16g}" for x in root) + ")"
    lines = [f"  Set {i+1}: " + ", ".join(formatRoot(root) for root in roots)
             for i, roots in enumerate(dupSets)]
    warnings.warn("The roots in each of the following sets might be duplicates of each other:\n"
                  + "\n".join(lines)
                  + "\nWe suggest you check the rank of the Jacobian at these points.")

def _refineWideBoxes(funcs, a, b, entries, dupSets, **solveKwargs):
    """Solve again around each root whose final box is wide, and replace that box's roots with what is found.

    Two roots closer than about sqrt(macheps) times the width of the interval an approximation was built
    on are one dip below that approximation's error, so they come back as one point, or a pair straddling
    the dip. Neither the error nor the dip is set by the roots: the dip is (d/2)^2 times the function's
    curvature, and the error shrinks with the function's size on the interval. Approximating again on a
    small neighborhood of the box therefore separates roots that are far closer together, whenever the
    function can be evaluated that accurately near them.

    Parameters
    ----------
    funcs : list
        The functions being solved, as passed to solve.
    a, b : numpy array
        The bounds of the search interval the boxes were found on.
    entries : list
        One (roots, boxes, isWide, isDupSet) tuple per final box, in the original coordinates, where roots
        has shape (k, dim), boxes shape (k, dim, 2), isWide says whether the box is to be solved again, and
        isDupSet whether its k > 1 roots might be duplicates of each other.
    dupSets : list
        The duplicate sets that the solves made here find among the roots they hand back are added to it.
    solveKwargs
        Passed through to solve.

    Returns
    -------
    entries : list
        The same list with the roots of each wide box replaced. Each new root's box is the box the neighborhood
        found for it, widened to contain the wide box, so it still holds a root the neighborhood's box missed
        (at a multiple root the neighborhood's approximation is less accurate than its error bound says). A box
        keeps what it had when its neighborhood cannot be solved or turns up no root; a neighborhood this small
        is at the edge of what the solver handles: y**2 = 0 on one, for one, reports no root at all, and a
        function that cannot be evaluated accurately there needs a degree above REFINE_MAX_DEGREE.
    """
    scale = functools.reduce(np.maximum, [np.abs(a), np.abs(b), 1])
    #Every box in units of scale, and the entry it came from, to decide which entry a root found on a
    #neighborhood belongs to.
    allBoxes = np.vstack([entry[1] for entry in entries]) / scale[:, np.newaxis]
    boxOwner = np.concatenate([[k] * len(entry[1]) for k, entry in enumerate(entries)])
    for k, (roots, boxes, isWide, isDupSet) in enumerate(entries):
        if not isWide:
            continue
        box = boxes[0]
        #Pad in units of the box's widest side in every dimension, so a dimension that already converged
        #to a subnormal width is still a neighborhood worth approximating on.
        halfWidth = REFINE_PADDING * np.max((box[:, 1] - box[:, 0]) / scale) * scale
        newA = np.maximum(box[:, 0] - halfWidth, a)
        newB = np.minimum(box[:, 1] + halfWidth, b)
        found = []
        try:
            newRoots, newBoxes = solve(funcs, newA, newB, returnBoundingBoxes=True, _refine=False,
                                       _maxDegree=REFINE_MAX_DEGREE, _dupSets=found, **solveKwargs)
        except (RecursionError, ChebyshevApproximator.DegreeCapExceeded):
            continue
        if len(newRoots) == 0:
            continue
        #The neighborhood can reach into another box; a root that belongs to that box is left to it.
        #A root belongs to the box it is nearest, measured in units of scale.
        scaledRoots = (newRoots / scale)[:, np.newaxis, :]
        distances = np.max(np.maximum(allBoxes[np.newaxis, :, :, 0] - scaledRoots,
                                      scaledRoots - allBoxes[np.newaxis, :, :, 1]), axis=2)
        own = boxOwner[np.argmin(distances, axis=1)] == k
        if np.any(own):
            #Keep the original box in every refined box. The neighborhood's boxes can be tighter than its
            #approximation is accurate (its error bound leaves out rounding), which at a multiple root moves
            #them off the root; the original box still contains it.
            unionBoxes = newBoxes[own].copy()
            unionBoxes[:, :, 0] = np.minimum(unionBoxes[:, :, 0], box[:, 0])
            unionBoxes[:, :, 1] = np.maximum(unionBoxes[:, :, 1], box[:, 1])
            entries[k] = (newRoots[own], unionBoxes, False, False)
            #Keep only the duplicate sets among the roots this box keeps.
            ownRoots = newRoots[own]
            dupSets.extend(roots_ for roots_ in found
                           if all(np.any(np.all(root == ownRoots, axis=1)) for root in roots_))
    return entries

def solve(funcs,a=-1,b=1, verbose = False, returnBoundingBoxes = False, exact=False, minBoundingIntervalSize=1e-5, max_cpu=1,
          parallel_depth=1, _refine=True, _maxDegree=None, _dupSets=None):
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
    _refine : bool
        Internal. Whether to solve again around a root whose box stayed wide (see _refineWideBoxes).
        False on the solves that refinement itself makes, so it happens once.
    _maxDegree : int or None
        Internal. The highest degree a callable's approximation may take (see REFINE_MAX_DEGREE).
    _dupSets : list or None
        Internal. Where the solves this one makes collect the sets of roots that might be duplicates, so
        the call the user made warns about all of them once. None on that call.

    Returns
    -------
    roots : numpy array
        The roots of the system of functions on the interval.
    boundingBoxes : numpy array, optional
        Only returned when ``returnBoundingBoxes`` is True. The exact intervals (boxes) in
        which each root is bound to lie.
    """
    #Only the call the user made warns; the solves it makes again add their duplicate sets to its list.
    isTopLevel = _dupSets is None
    if isTopLevel:
        _dupSets = []

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
            errs[i] = relativeRoundingError(polys[i])
            if not unit_box:
                polys[i], errs[i] = ChebyshevSubdivisionSolver.transformCheb(polys[i], alphas, betas, errs[i], exact)
        elif isinstance(funcs[i], MultiCheb):
            polys[i] = funcs[i].coeff
            errs[i] = relativeRoundingError(polys[i])
            if not unit_box:
                polys[i], errs[i] = ChebyshevSubdivisionSolver.transformCheb(polys[i], alphas, betas, errs[i], exact)
        else:
            polys[i], errs[i] = ChebyshevApproximator.chebApproximate(funcs[i],a,b,maxDegree=_maxDegree)
        if verbose:
            print(f"{i}: {polys[i].shape}", end = " " if i != dim-1 else '\n')
    if verbose:
        print(f"Searching on interval {[[a[i],b[i]] for i in range(dim)]}")

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
                                 max_cpu=max_cpu, parallel_depth=parallel_depth, _refine=_refine, _maxDegree=_maxDegree,
                                 _dupSets=_dupSets)
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
        if isTopLevel:
            _warnDuplicateRoots(_dupSets)
        if returnBoundingBoxes:
            return yroots, boundingBoxes
        else:
            return yroots
    
    #TODO: Handle if we have extra roots at the top level.

    #If any of the bounding boxes is too large, re-solve that box.
    #Each entry is (roots, boxes, isWide, isDupSet); see _refineWideBoxes.
    entries = []
    #Get the relative max size in each dimension. If a or b > 1 in magnitude, minBoundingIntervalSize is a relative number.
    #If they are < 1 in magnitude, it is an absolute number.
    scale = functools.reduce(np.maximum, [np.abs(a),np.abs(b), 1])
    relMaxSize = minBoundingIntervalSize * scale
    for box in boundingBoxes:
        newA, newB = ChebyshevApproximator.transform(box.finalInterval.T,a,b)
        if np.all(newB - newA > relMaxSize):
            #Re-solve this box
            if verbose:
                print("Re-solving on:", newA, newB)
            roots, boxes = solve(funcs, a=newA, b=newB, verbose=verbose, returnBoundingBoxes=True, exact=exact, minBoundingIntervalSize=minBoundingIntervalSize,
                                 max_cpu=max_cpu, parallel_depth=parallel_depth, _refine=_refine, _maxDegree=_maxDegree,
                                 _dupSets=_dupSets)
            if len(roots) > 0:
                entries.append((roots, boxes, False, False))
        else:
            #Transform back
            transformedBox = ChebyshevApproximator.transform(box.finalInterval.T,a,b).T
            #Get the roots from this box, and repeat the box once per root it reports, so
            #finalRoots and finalBoxes stay index-aligned. A box that could not separate the
            #roots inside it reports more than one, and each of them gets that same box.
            boxRoots = ChebyshevSubdivisionSolver.getRootsInInterval(box)
            isWide = np.max((transformedBox[:,1] - transformedBox[:,0]) / scale) > REFINE_BOX_SIZE
            entries.append((ChebyshevApproximator.transform(np.array(boxRoots),a,b),
                            np.repeat(transformedBox[np.newaxis], len(boxRoots), axis=0), isWide, len(boxRoots) > 1))

    #Only a callable is approximated again on a neighborhood. A polynomial given as coefficients is carried
    #there by transformCheb, whose error stays that of the coefficients however small the neighborhood.
    if _refine and any(entry[2] for entry in entries) and \
            not all(isinstance(f, (MultiPower, MultiCheb)) for f in funcs):
        entries = _refineWideBoxes(funcs, a, b, entries, _dupSets, verbose=verbose, exact=exact,
                                   minBoundingIntervalSize=minBoundingIntervalSize, max_cpu=max_cpu,
                                   parallel_depth=parallel_depth)
    #The boxes refinement did not separate still report more than one root each.
    _dupSets.extend(entry[0] for entry in entries if entry[3])
    finalRoots = [entry[0] for entry in entries]
    finalBoxes = [entry[1] for entry in entries]
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
    if isTopLevel:
        _warnDuplicateRoots(_dupSets)
    if returnBoundingBoxes:
        return finalRoots, finalBoxes
    else:
        return finalRoots
