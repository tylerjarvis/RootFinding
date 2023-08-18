import numpy as np
from numba import njit
from yroots.polynomial import MultiCheb, MultiPower
import itertools
import warnings

@njit
def transform(x, a, b):
    """Transforms points from the interval [-1, 1] to the interval [a, b].

    Parameters
    ----------
    x : numpy array
        The points to be tranformed.
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

def chebyshevBlockCopy(values):
    """Expands function evaluation values into the full matrix needed for the Chebyshev FFT.

    Parameters
    ----------
    values : numpy array
        Function evaluations

    Returns
    -------
    result : numpy array
        The block matrix of Chebyshev values ready for the FFT
    """    
    dim = values.ndim
    degs = [i-1 for i in values.shape]

    #Initialize result as a larger copy of values
    result = np.zeros([2*i for i in degs])
    slice1 = [slice(0, d) for d in values.shape]
    slice2 = slice1.copy()
    result[tuple(slice2)] = values

    #Unfold the result one dimension at a time
    for i in range(dim):
        slice1[i] = slice(degs[i]-1, 0, -1)
        slice2[i] = slice(degs[i]+1, 2*degs[i])
        result[tuple(slice2)] = result[tuple(slice1)]
        slice1[i] = slice(None, None)
        slice2[i] = slice(None, None)
    return result

def interval_approximate_nd(f, degs, a, b, retSupNorm = False):
    """Generates an approximation of f on [a,b] using Chebyshev polynomials of degs degrees.

    Calculates the values of the function at the Chebyshev grid points and performs the FFT
    on these points to achieve the desired approximation.

    Parameters
    ----------
    f : function from R^n -> R
        The function to interpolate.
    a : numpy array
        The lower bound on the interval.
    b : numpy array
        The upper bound on the interval.
    degs : list of ints
        A list of the degree of interpolation in each dimension.
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
    # If any dimension has degree 0, turn it to degree 1 (will be sliced out at the end)
    originalDegs = degs.copy()
    degs[degs == 0] = 1 

    # Get the Chebyshev Grid Points
    cheb_grid = np.meshgrid(*([transform(np.cos(np.arange(deg+1)*np.pi/deg), a_,b_) 
                               for deg, a_, b_ in zip(degs, a, b)]),indexing='ij')
    cheb_pts = np.column_stack(tuple(map(lambda x: x.flatten(), cheb_grid)))
    if isinstance(f, MultiCheb) or isinstance(f, MultiPower): # for faster function evaluations
        values_block = f(cheb_pts).reshape(*(degs+1))
    else:
        values_block = f(*cheb_pts.T).reshape(*(degs+1))

    # Get the Chebyshev coefficients
    fft_block = chebyshevBlockCopy(values_block)
    coeffs = np.fft.rfftn(fft_block/np.product(degs)).real
    for d in range(dim):
        coeffs[tuple([slice(None) if i != d else 0 for i in range(dim)])] /= 2
        coeffs[tuple([slice(None) if i != d else degs[i] for i in range(dim)])] /= 2
        
    #Return the coefficient tensor and the sup norm
    slices = tuple([slice(0, d+1) for d in originalDegs]) # get values corresponding to originalDegs only
    if retSupNorm:
        supNorm = np.max(np.abs(values_block))
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
    return np.all(coeffList[-5:] < tol)
    
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
        Tolerance (distance from zero) used to determine wheher the coefficients have converged.
    
    Returns
    -------
    hasConverged : bool
        True if all the values of coeff and coeff2 are within tol of each other; False otherwise
    """
    coeff3 = coeff2.copy()
    # Subtract off coeff from coeff2 elementwise and ensure all elements are then less than tol
    coeff3[tuple([slice(0, d) for d in coeff.shape])] -= coeff 
    return np.max(np.abs(coeff3)) < tol
    
def getFinalDegree(coeff,tol):
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
    epsVal = 2*np.max(coeff[convergedDeg:]) # Set epsVal to 2x the largest coefficient past degree 3n/2
    nonZeroCoeffs = np.where(coeff > epsVal)[0]
    degree = 1 if len(nonZeroCoeffs) == 0 else max(1, nonZeroCoeffs[-1])

    # Set degree to 0 for constant functions (all coefficients but first are less than tol)
    if np.all(coeff[1:] < tol):
        degree = 0
    
    # Calculate the rate of convergence
    maxSpot = np.argmax(coeff)
    rho = (coeff[maxSpot]/epsVal)**(1/((degree - maxSpot) + 1)) 
    return degree, epsVal, rho
        
def getChebyshevDegrees(f, a, b, absApproxTol, relApproxTol):
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
    absApproxTol : float
        The absolute tolerance (distance from zero) used to determine convergence
    relApproxTol : float
        The relative tolerance (distance from zero) used to determine convergence
    
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

    # Find the degree in each dimension seperately
    for i in range(dim):
        eval_pts = np.array([(0.209+val)*0.463**val for val in range(dim)])
        eval1 = f(*transform(eval_pts,a,b))
        eval_pts[i] = -np.sqrt(np.exp(1))/2
        eval2 = f(*transform(eval_pts,a,b))
        if np.allclose(eval1,eval2):
            eval_pts = np.array([-np.sin(1.389*val**2) for val in range(dim)])
            eval1 = f(*transform(eval_pts,a,b))
            eval_pts[i] = np.pi**(1/3)/4
            eval2 = f(*transform(eval_pts,a,b))
            if np.allclose(eval1,eval2):
                chebDegrees[i] = 0
    for currDim in range(dim):
        if chebDegrees[currDim] == 0:
            epsilons.append(0)
            rhos.append(np.inf)
            continue
        # Isolate the current dimension by fixing all other dimensions at degree 5 approximation
        degs = np.array([5]*dim if dim <= 5 else [2]*dim)
        for i in range(len(chebDegrees)):
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
                continue # Iterate if the coefficients have not fully converged.
            
            # The coefficients have been shown to converge to 0. Get the exact degree where this occurs.
            coeffChunk = np.average(np.abs(coeff2), axis=tupleForChunk)
            deg, eps, rho = getFinalDegree(coeffChunk,tol)
            chebDegrees[currDim] = deg
            epsilons.append(eps)
            rhos.append(rho)
            break # Repeat for next dimension
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
        for i, used in enumerate(idxs):
            if used:
                # multiply by infinite sum of coeffs past the degree at which the approx stops in dim i
                s *= epsilons[i] / (1-1/rhos[i])
            else:
                # multiply by the number of coefficients in the approximation along dim i
                s *= degs[i]
        # Divide by rho[i] if only index i
        usedSpots = np.where(np.array(idxs) == 1)[0]
        if len(usedSpots) == 1:
            s /= rhos[usedSpots[0]]
        # Append to the error
        approxError += s
    return approxError

def chebApproximate(f, a, b, absApproxTol=1e-10, relApproxTol=1e-10):
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
    >>> approx, error = yroots.approximate(f,[-1,-1,-1],[1,1,1])
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
    >>> approx = yroots.approximate(g,[0],[5])[0]
    >>> print(approx)
    [ 1.42352509e+00  9.49016725e-01 -1.89803345e-01 ... -1.24418041e-10
      1.24418045e-10 -6.22090244e-11]


    Parameters
    ----------
    f : function
        The function to be approximated. NOTE: Valid input is restricted to callable Python functions
        (including user-created functions) and yroots Polynomial (MultiCheb and MultiPower) objects.
        String representations of functions are not valid input.
    a: list or numpy array
        An array containing the lower bound of the approximation interval in each dimension, listed in
        dimension order
    b: list or numpy array
        An array containing the upper bound of the approximation interval in each dimension, listed in
        dimension order.
    absApproxTol : float
        The absolute tolerance used to determine at what degree the Chebyshev coefficients have
        converged to zero. If all coefficients after degree n are within absApproxTol from zero,
        the coefficients will be considered to have converged at degree n. Defaults to 1e-10.
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
    
    # If the function is a MultiCheb object on [-1,1]^n, then return its matrix as the approximation
    if isinstance(f,MultiCheb) and np.allclose(a,-np.ones_like(a)) and np.allclose(b,np.ones_like(b)):
        return f.coeff.astype(float), 0
    
    # Generate and return the approximation
    degs, epsilons, rhos = getChebyshevDegrees(f, a, b, absApproxTol, relApproxTol)
    return interval_approximate_nd(f, degs, a, b), getApproxError(degs, epsilons, rhos)