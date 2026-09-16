"""Chebyshev approximation utilities used by :mod:`yroots.Combined_Solver`.

Provides :func:`chebApproximate` (the public entry point) along with the
helpers it relies on to choose a degree per dimension, evaluate on the
Chebyshev grid via :func:`scipy.fftpack.dctn`, and estimate the approximation
error.
"""
import numpy as np
from numba import njit
from yroots.polynomial import MultiCheb, MultiPower
import itertools
from scipy.fftpack import dctn
import warnings

@njit(cache=True)
def transform(x, a, b):
    """Transforms points from the interval [-1, 1] to the interval [a, b].

    Parameters
    ----------
    x : numpy array
        The points to be transformed.
    a : numpy array
        The lower bounds on the interval.
    b : numpy array
        The upper bounds on the interval.

    Returns
    -------
    transformed_pts : numpy array
        The transformed points.
    """
    return ((b-a)*x+(b+a))/2

def evaluateGrid(f, cheb_grid, shape):
    """Evaluates f at every point of the Chebyshev grid.

    Passes the whole grid through f in a single call, one coordinate array per dimension, which
    is what the documented contract for an input function asks for ("must be smooth on the domain
    and vectorized"). The grid holds one point per coefficient of the approximation, so past a low
    degree in two dimensions it is thousands of points, and calling f once per point spends nearly
    all of its time on Python call overhead rather than in f.

    A function that turns out not to be vectorized is still evaluated one point at a time, the way
    it always was. A scalar-only function either raises when handed arrays (a ``math`` function, a
    Python ``if`` on a value) or hands back something that is not one value per grid point, and
    both are caught here, as is a result that is neither one value per grid point nor a single
    value for the whole grid -- numpy would right align such a result against the grid, which when
    the grid is square silently orients the values along the wrong axis.

    Parameters
    ----------
    f : function from R^n -> R
        The function to evaluate.
    cheb_grid : list of numpy arrays
        The coordinate arrays of the grid, each of shape ``shape``, as returned by
        :func:`numpy.meshgrid` with ``indexing='ij'``.
    shape : tuple of ints
        The shape of the grid.

    Returns
    -------
    values : numpy array
        The value of f at each grid point, of shape ``shape``.
    """
    def pointByPoint():
        """Evaluate f once per grid point, the way this was always done."""
        cheb_pts = np.column_stack(tuple(map(lambda x: x.flatten(), cheb_grid)))
        return np.array([f(*cheb_pt) for cheb_pt in cheb_pts]).reshape(shape)

    try:
        values = np.asarray(f(*cheb_grid), dtype=float)
    except Exception:
        # f did not take the whole grid at once; evaluate it one point at a time instead.
        return pointByPoint()

    if values.shape == shape:
        return values
    if values.size == 1:
        # A function that ignores its inputs hands back a single value for the whole grid. That is
        # the only shape other than the grid's own that lands on it unambiguously.
        return np.full(shape, values.reshape(-1)[0])
    # Anything else cannot be placed on the grid without guessing. numpy would right align it,
    # which on a square grid spreads the values along the wrong axis and returns them as though
    # they were correct, so take the slow path instead: it cannot get the orientation wrong.
    return pointByPoint()

def interval_approximate_nd(f, degs, a, b, retSupNorm = False):
    """Generates an approximation of f on [a,b] using Chebyshev polynomials of degs degrees.

    Calculates the values of the function at the Chebyshev grid points and performs the FFT
    on these points to achieve the desired approximation.

    Parameters
    ----------
    f : function from R^n -> R
        The function to interpolate.
    degs : list of ints
        A list of the degree of interpolation in each dimension.
    a : numpy array
        The lower bound on the interval.
    b : numpy array
        The upper bound on the interval.
    retSupNorm : bool
        Whether to return the sup norm of the function.

    Returns
    -------
    coeffs : numpy array
        The coefficients of the Chebyshev interpolating polynomial.
    supNorm : float (optional)
        The sup norm of the function, approximated as the maximum function evaluation.
    """
    dim = len(degs)
    # If any dimension has degree 0, turn it to degree 1 (will be sliced out at the end).
    # Work on a copy so the degrees passed in by the caller are left untouched.
    originalDegs = degs
    degs = np.array(degs)
    degs[degs == 0] = 1

    # Get the Chebyshev Grid Points
    cheb_grid = np.meshgrid(*([transform(np.cos(np.arange(deg+1)*np.pi/deg), a_,b_) 
                               for deg, a_, b_ in zip(degs, a, b)]),indexing='ij')

    if isinstance(f, MultiCheb) or isinstance(f, MultiPower): # for faster function evaluations
        #TODO: Evaluate Grid???
        cheb_pts = np.column_stack(tuple(map(lambda x: x.flatten(), cheb_grid)))
        values = f(cheb_pts).reshape(*(degs+1))
    else:
        values = evaluateGrid(f, cheb_grid, tuple(degs+1))
    #Get the supNorm if we want it
    if retSupNorm:
        supNorm = np.max(np.abs(values))

    #TODO: Save the duplicated function values when we double the approximation.
    #Less efficient in higher dimensions, we save 1/2**(dim-1) of the functions evals

    #Do real DCT
    coeffs = dctn(values/np.prod(degs), type=1, overwrite_x=True)
    #Divide edges by 2    
    for d in range(dim):
        coeffs[tuple([slice(None) if i != d else 0 for i in range(dim)])] /= 2
        coeffs[tuple([slice(None) if i != d else degs[i] for i in range(dim)])] /= 2
        
    #Return the coefficient tensor and the sup norm
    slices = tuple([slice(0, d+1) for d in originalDegs]) # get values corresponding to originalDegs only
    if retSupNorm:
        return coeffs[slices], supNorm
    else:
        return coeffs[slices]

def startedConverging(coeffList, tol):
    """Determine whether the high-degree coefficients of a given Chebyshev approximation are near 0.

    Parameters
    ----------
    coeffList : numpy array
        Absolute values of chebyshev coefficients.
    tol : float
        Tolerance (distance from zero) used to determine whether coeffList has started converging.
    
    Returns
    -------
    startedConverging : bool
        True if the last 5 coefficients of coeffList are less than tol; False otherwise
    """
    return np.all(coeffList[-5:] <= tol)
    
def hasConverged(coeff, coeff2, tol):
    """Determine whether the high-degree coefficients of a Chebyshev approximation have converged
    to machine epsilon.

    Parameters
    ----------
    coeff : numpy array
        Absolute values of chebyshev coefficients of degree n approximation.
    coeff2 : numpy array
        Absolute values of chebyshev coefficients of degree 2n+1 approximation.
    tol : float
        Tolerance (distance from zero) used to determine whether the coefficients have converged.
    
    Returns
    -------
    hasConverged : bool
        True if all the values of coeff and coeff2 are within tol of each other; False otherwise
    """
    coeff3 = coeff2.copy()
    # Subtract off coeff from coeff2 elementwise and ensure all elements are then less than tol
    coeff3[tuple([slice(0, d) for d in coeff.shape])] -= coeff 
    return np.max(np.abs(coeff3)) <= tol
    
def getFinalDegree(coeff,tol,macheps = 2**-52):
    """Finalize the degree of Chebyshev approximation to use along one particular dimension.

    This function is called after the coefficients have started converging at degree n. A degree
    2n+1 approximation is passed in. Assuming that the coefficients have fully converged by degree 
    3n/2, the cutoff epsVal is calculated as twice the max coefficient of degree at least 3n/2.
    The final degree is then set as the largest coefficient with magnitude greater than epsVal.
    
    The rate of convergence is calculated assuming that the coefficients converge geometrically
    starting from the largest coefficient until machine epsilon is reached. This is a lower bound, as
    in practice, the coefficients usually slowly decrease at first but drop off fast at the end.

    Parameters
    ----------
    coeff : numpy array
        Absolute values of chebyshev coefficients.
    tol : float
        Tolerance below which a coefficient is treated as zero when deciding whether ``f`` is
        effectively constant (in which case the returned degree is 0).
    macheps : float
        Machine epsilon used as the floor when computing ``epsVal``. Defaults to ``2**-52``.
    
    Returns
    -------
    degree : int
        The numerical degree of the approximation
    epsVal : float
        The epsilon value to which the coefficients have converged
    rho : float
        The geometric rate of convergence of the coefficients
    """
    # Set the final degree to the position of the last coefficient greater than convergence value
    convergedDeg = int(3 * (len(coeff) - 1) / 4) # Assume convergence at degree 3n/2.
    epsVal = 2*max(macheps,np.max(coeff[convergedDeg:])) # Set epsVal to 2x the largest coefficient past degree 3n/2
    nonZeroCoeffs = np.where(coeff > epsVal)[0]
    degree = 1 if len(nonZeroCoeffs) == 0 else max(1, nonZeroCoeffs[-1])

    # Set degree to 0 for constant functions (all coefficients but first are less than tol)
    if np.all(coeff[1:] <= tol):
        degree = 0
    
    # Calculate the rate of convergence
    maxSpot = np.argmax(coeff)
    if coeff[maxSpot] == 0:
        #Every coefficient is 0, so the approximation is exact. Report perfect convergence
        #instead of dividing 0 by epsVal, which would give a rate of 0 (and a negative error bound).
        return degree, 0, np.inf
    if epsVal == 0:
        epsVal = coeff[maxSpot] * 1e-24
    rho = (coeff[maxSpot]/epsVal)**(1/((degree - maxSpot) + 1))
    return degree, epsVal, rho

def checkConstantInDimension(f,a,b,currDim, relApproxTol, absApproxTol = 0):
    """Check to see if the output of f is not dependent on the input coordinate of a dimension.
    
    Uses predetermined random numbers to find a point x in the interval where f(x) != 0 and checks
    whether the value of f changes as the dimension currDim coordinate of x changes. Repeats twice.

    Parameters
    ----------
    f : function
        The function being evaluated.
    a : numpy array
        The lower bound on the interval.
    b : numpy array
        The upper bound on the interval.
    currDim : int
        The dimension being examined.
    relApproxTol : float
        Relative tolerance passed to :func:`numpy.isclose`/:func:`numpy.allclose` when comparing
        function evaluations to decide whether ``f`` varies along ``currDim``.
    absApproxTol : float
        Absolute tolerance passed to :func:`numpy.isclose`/:func:`numpy.allclose`. Defaults to 0.

    Returns
    -------
    is_constant : bool
        Whether ``f`` is constant in dimension ``currDim``. Returns False if the test is
        indeterminate or f is seen to vary with different values of x[dim]. Returns True otherwise.
    """
    if isinstance(f,MultiPower) or isinstance(f,MultiCheb): # Points evaluated differently for these
        is_m = True
    else:
        is_m = False
    dim = len(a)

    # First test point x_1
    x_1 = transform(np.array([0.8984743990614998**(val+1) for val in range(dim)]),a,b)
    eval1 = (f(*x_1) if not is_m else f(x_1))
    if np.isclose(eval1,0,rtol=relApproxTol, atol=absApproxTol): # Make sure f(x_1) != 0 (unlikely)
        return False
    # Test how changing x_1[dim] changes the value of f for several values         
    for val in transform(np.array([-0.7996847717584993,0.18546110255464776,-0.13975937255055182,0.,1.,-1.]),a[currDim],b[currDim]):
        x_1[currDim] = val
        eval2 = (f(*x_1) if not is_m else f(x_1))
        if not np.allclose(eval1,eval2,rtol=relApproxTol, atol=absApproxTol): # Corresponding points gave different values for f(x)
            return False

    # Second test point x_2
    x_2 = transform(np.array([(-0.2598647169391334*(val+1)/(dim))**2 for val in range(dim)]),a,b)
    eval1 = (f(*x_2) if not is_m else f(x_2))
    if np.isclose(eval1,0,rtol=relApproxTol, atol=absApproxTol): # Make sure f(x_2) != 0 (unlikely)
            return False
    for val in transform(np.array([-0.17223860129797386,0.10828286380141305,-0.5333148248321931,0.46471703497219596]),a[currDim],b[currDim]):
        x_2[currDim] = val
        eval2 = (f(*x_2) if not is_m else f(x_2))
        if not np.allclose(eval1,eval2,rtol=relApproxTol, atol=absApproxTol):
            return False # Corresponding points gave different values for f(x)
    # Both test points had not zeros of f and had no variance along dimension currDim.
    return True
        
def getChebyshevDegrees(f, a, b, relApproxTol, absApproxTol = 0):
    """Compute the minimum degrees in each dimension that give a reliable Chebyshev approximation for f.

    For each dimension, starts with degree 8, generates an approximation, and checks to see if the
    sequence of coefficients is converging. Repeats, doubling the degree guess until the coefficients
    are seen to converge to 0. Then calls getFinalDegree to get the exact degree of convergence.
    
    Parameters
    ----------
    f : function
        The function being approximated.
    a : numpy array
        The lower bound on the interval.
    b : numpy array
        The upper bound on the interval.
    relApproxTol : float
        The relative tolerance (distance from zero) used to determine convergence.
    absApproxTol : float
        The absolute tolerance (distance from zero) used to determine convergence.
    
    Returns
    -------
    chebDegrees : numpy array
        The numerical degree in each dimension.
    epsilons : numpy array
        The value the coefficients converged to in each dimension.
    rhos : numpy array
        The rate of convergence in each dimension.
    """    
    dim = len(a)
    chebDegrees = [np.inf]*dim # the approximation degree in each dimension
    epsilons = [] # the value the approximation has converged to in each dimension
    rhos = [] # the calculated rate of convergence in each dimension
    # Check to see if f varies each input; set degree to 0 if not
    for currDim in range(dim):
        if checkConstantInDimension(f,a,b,currDim,relApproxTol,absApproxTol):
            chebDegrees[currDim] = 0

    # Find the degree in each dimension seperately
    for currDim in range(dim):
        if chebDegrees[currDim] == 0: # skip the guessing algorithm if f is constant in dim currDim
            epsilons.append(0)
            rhos.append(np.inf)
            continue
        # Isolate the current dimension by fixing all other dimensions at constant degree approximation
        degs = np.array([5]*dim if dim <= 5 else [2]*dim)
        for i in range(len(chebDegrees)): # save computation by using already computed degrees if lower
            if chebDegrees[i] < degs[i]:
                degs[i] = chebDegrees[i]
        currGuess = 8 # Take initial guess degree 8 in the current dimension
        tupleForChunk = tuple([i for i in range(currDim)] + [i for i in range(currDim+1,dim)])
        while True: # Runs until the coefficients are shown to converge to 0 in this dimension
            if currGuess > 1e5:
                warnings.warn(f"Approximation bound exceeded!\n\nApproximation degree in dimension {currDim} "
                              + "has exceeded 1e5, so the process may not finish.\n\nConsider interrupting "
                              "and restarting the process after ensuring that the function(s) inputted are " +
                              "continuous and smooth on the approximation interval.\n\n")
            degs[currDim] = currGuess
            coeff, supNorm = interval_approximate_nd(f, degs, a, b, retSupNorm=True) # get approximation
            # Get "average" coefficients along the current dimension
            coeffChunk = np.average(np.abs(coeff), axis=tupleForChunk)
            tol = absApproxTol + supNorm * relApproxTol # Set tolerance for convergence from the supNorm
            currGuess *= 2 # Ensure the degree guess is doubled in case of another iteration

            # Check if the coefficients have started converging; iterate if they have not.
            if not startedConverging(coeffChunk, tol):
                continue

            # Since the coefficients have started to converge, check if they have fully converged.
            # Degree n and 2n+1 are unlikely to have higher degree terms alias into the same spot.
            degs[currDim] = currGuess + 1 # 2n+1
            coeff2, supNorm2 = interval_approximate_nd(f, degs, a, b, retSupNorm=True)
            tol = absApproxTol + max(supNorm, supNorm2) * relApproxTol
            if not hasConverged(coeff, coeff2, tol):
                continue # Keep doubling if the coefficients have not fully converged.
            
            # The coefficients have been shown to converge to 0. Get the exact degree where this occurs.
            coeffChunk = np.average(np.abs(coeff2), axis=tupleForChunk)
            deg, eps, rho = getFinalDegree(coeffChunk,tol)
            chebDegrees[currDim] = deg
            epsilons.append(eps)
            rhos.append(rho)
            break # Finished with the current dimension
    return np.array(chebDegrees), np.array(epsilons), np.array(rhos)

def getApproxError(degs, epsilons, rhos):
    """Computes an upper bound for the error of the Chebyshev approximation.

    Using the epsilon values and rates of geometric convergence calculated in getChebyshevDegrees,
    calculates the infinite sum of the coefficients past those used in the approximation along each
    dimension and multiplies these sums appropriately by the number of coefficients used to approximate
    in the other dimensions.

    Parameters
    ----------
    degs: numpy array
        The degrees in each dimension of the approximation.
    epsilons : numpy array
        The values to which the approximation converged in each dimension.
    rhos : numpy array
        The calculated rate of convergence in each dimension.
    
    Returns
    -------
    approxError : float
        An upper bound on the approximation error
    """    
    approxError = 0
    # Create a partition of coefficients where idx[i]=1 represents coefficients being greater than
    # degs[i] in dimension i and idx[i]=0 represens coefficients being less than [i] in dimension i.
    for idxs in itertools.product(range(2), repeat=len(degs)):
        # Skip the set of all 0's, corresponding to the terms actually included in the approximation.
        if np.sum(idxs) == 0:
            continue
        s = 1
        thisEps = 0
        for i, used in enumerate(idxs):
            if used:
                # multiply by infinite sum of coeffs past the degree at which the approx stops in dim i
                #1/rhos[i] is the rate, so this is (1/rhos[i]) / (1 - 1/rhos[i]) = 1/(rhos[i]-1)
                s /= (rhos[i]-1)
                #The points in this section are going to be < max(epsilons[i] that contribute to it)
                thisEps = max(thisEps, epsilons[i])
            else:
                # multiply by the number of coefficients in the approximation along dim i
                s *= (degs[i] + 1)
        # Append to the error
        approxError += s * thisEps
    return approxError

def chebApproximate(f, a, b, relApproxTol=1e-10):
    """Generate and return an approximation for the function f on the interval [a,b].

    Uses properties of Chebyshev polynomials and the FFT to quickly generate a reliable
    approximation. Examines approximation one dimension at a time to determine the degree at which
    the coefficients geometrically converge to 0 in each dimension, then calculates and returns a
    final approximation of these degree values along with the associated approximation error.

    NOTE: The approximate function is only guaranteed to work well on functions that are continuous
    and smooth on the approximation interval. If the input function is not continuous and smooth on
    the interval, the approximation may get stuck in recursion.

    Examples
    --------

    >>> f = lambda x,y,z: x**2 - y**2 + 3*x*y
    >>> approx, error = yroots.chebApproximate(f,[-1,-1,-1],[1,1,1])
    >>> print(approx)
    [[[ 0.00000000e+00]
      [ 1.11022302e-16]
      [-5.00000000e-01]]
     [[ 1.11022302e-16]
      [ 3.00000000e+00]
      [-1.11022302e-16]]
     [[ 5.00000000e-01]
      [-1.11022302e-16]
      [ 0.00000000e+00]]]
    >>> print(error)
    2.8014584982224306e-24

    >>> g = np.sqrt
    >>> approx = yroots.chebApproximate(g,[0],[5])[0]
    >>> print(approx)
    [ 1.42352509e+00  9.49016725e-01 -1.89803345e-01 ... -1.24418041e-10
      1.24418045e-10 -6.22090244e-11]


    Parameters
    ----------
    f : function
        The function to be approximated. NOTE: Valid input is restricted to callable Python functions
        (including user-created functions) and yroots Polynomial (MultiCheb and MultiPower) objects.
        String representations of functions are not valid input.
    a : list or numpy array
        An array containing the lower bound of the approximation interval in each dimension, listed in
        dimension order.
    b : list or numpy array
        An array containing the upper bound of the approximation interval in each dimension, listed in
        dimension order.
    relApproxTol : float
        The relative tolerance used to determine at what degree the Chebyshev coefficients have
        converged to zero. If all coefficients after degree n are within relApproxTol * supNorm
        (the maximum function evaluation on the interval) of zero, the coefficients will be
        considered to have converged at degree n. Defaults to 1e-10.
    
    Returns
    -------
    coefficient_matrix : numpy array
        The coefficient matrix of the Chebyshev approximation.
    error : float
        The error associated with the approximation.
    """

    # Ensure passed in arguments are valid inputs
    if not hasattr(f, '__call__'):
            raise ValueError(f"Invalid input: input function is not callable")
    if type(a) == list:
        a = np.array(a)
    if type(b) == list:
        b = np.array(b)
    if type(a) != np.ndarray:
        a = np.array([a])
    if type(b) != np.ndarray:
        b = np.array([b])
    if len(a) != len(b):
        raise ValueError(f"Invalid input: {len(a)} lower bounds were given but {len(b)} upper bounds were given")
    if (b<a).any():
        raise ValueError(f"Invalid input: at least one lower bound is greater than the corresponding upper bound.")
    try:
        if isinstance(f,MultiPower) or isinstance(f,MultiCheb):
            f([(u+l)/2 for l,u in zip(a,b)])
        else:
            f(*[(u+l)/2 for l,u in zip(a,b)])
    except TypeError as e:
        raise ValueError("Invalid input: length of the upper/lower bound lists must match the dimension of the function")
    
    # Generate and return the approximation
    degs, epsilons, rhos = getChebyshevDegrees(f, a, b, relApproxTol)
    return interval_approximate_nd(f, degs, a, b), getApproxError(degs, epsilons, rhos)
