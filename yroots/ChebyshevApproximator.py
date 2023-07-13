import numpy as np
from yroots.polynomial import MultiCheb, MultiPower
from yroots.utils import transform
import itertools

def chebyshevBlockCopy(values):
    """Expands the function evaluations values into the full matrix needed for the Chebyshev FFT.

    Parameters
    ----------
    values : numpy array
        Function evaluations

    Returns
    -------
    result : numpy array
        Chebyshev Interpolation values for FFT
    """    
    dim = values.ndim
    degs = [i-1 for i in values.shape]
    #Initialize result as a larger copy of values
    result = np.zeros([2*i for i in degs])
    slice1 = [slice(0, d) for d in values.shape] #Slice From
    slice2 = slice1.copy() #Slice To
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
    """Finds the chebyshev approximation of an n-dimensional function on an interval.

    Parameters
    ----------
    f : function from R^n -> R
        The function to interpolate.
    a : numpy array
        The lower bound on the interval.
    b : numpy array
        The upper bound on the interval.
    degs : list
        the degree of the interpolation in each dimension.
    retSupNorm : bool
        whether to return the sup norm of the function.

    Returns
    -------
    coeffs : numpy array
        The coefficient of the chebyshev interpolating polynomial.
    supNorm : float
        The supNorm of the function, approximated as the maximum function evaluation.
    """
    dim = len(degs)
    #Get the Chebyshev Grid Points
    cheb_grid = np.meshgrid(*([transform(np.cos(np.arange(deg+1)*np.pi/deg), a_,b_) 
                               for deg, a_, b_ in zip(degs, a, b)]),indexing='ij')
    cheb_pts = np.column_stack(tuple(map(lambda x: x.flatten(), cheb_grid)))
    if isinstance(f, MultiCheb) or isinstance(f, MultiPower):
        values_block = f(cheb_pts).reshape(*(degs+1))
    else:
        values_block = f(*cheb_pts.T).reshape(*(degs+1))

    #TODO: Save the duplicated function values when we double the approximation.
    #Less efficient in higher dimensions, we save 1/2**(dim-1) of the functions evals
    
    #Do the function evaluations
    values = chebyshevBlockCopy(values_block)

    #Do real FFT
    coeffs = np.fft.rfftn(values/np.product(degs)).real
    #Divide edges by 2
    for d in range(dim):
        coeffs[tuple([slice(None) if i != d else 0 for i in range(dim)])] /= 2
        coeffs[tuple([slice(None) if i != d else degs[i] for i in range(dim)])] /= 2
        
    #Return Coeff Tensor and SupNorm if desired
    slices = tuple([slice(0, d+1) for d in degs])
    if retSupNorm:
        supNorm = np.max(np.abs(values_block))
        return coeffs[slices], supNorm
    else:
        return coeffs[slices]

def startedConverging(coeffList, tol):
    """Checks if a list of chebyshev coefficients has started to converge.

    Parameters
    ----------
    coeffList : numpy array
        Absolute values of chebyshev coefficients.
    tol : float
        Tolerance to decide if we've started converging.
    
    Returns
    -------
    startedConverging : bool
        If we've started converging. Determined by the last 5 coefficients all being less than tol.
    """
    return np.all(coeffList[-5:] < tol)
    
def hasConverged(coeff, coeff2, tol):
    """Checks if a chebyshev approximation has converged.

    Parameters
    ----------
    coeff : numpy array
        Absolute values of chebyshev coefficients of degree n approximation.
    coeff2 : numpy array
        Absolute values of chebyshev coefficients of degree 2n-1 approximation.
    tol : float
        Tolerance to decide if we've converged.
    
    Returns
    -------
    hasConverged : bool
        If we've converged. All the coefficients of coeff and coeff2 being withing tol of each other.
    """
    coeff3 = coeff2.copy()
    coeff3[tuple([slice(0, d) for d in coeff.shape])] -= coeff
    return np.max(np.abs(coeff3)) < tol
    
def getFinalDegree(coeff):
    """Computes the degree of a chebyshev approximation.

    We assume that we started converging at some degree n, and that coeff is degree 2n-1.
    
    We then assume that by degree 3n/2, we have fully converged. We calculate epsVal as twice
    the max coefficient of degree at least 3n/2.
    
    We then set the degree to be the largest coefficient that has magnitude greater than epsVal.
    
    For the rate of convergence, we assume the coefficients converge geometrically from the largest
    coefficient until they hit machine epsilon. This is a lower bound, as in reality they probably
    slowly decrease, and then start converging faster once they get near the end.

    Parameters
    ----------
    coeff : numpy array
        Absolute values of chebyshev coefficients.
    
    Returns
    -------
    degree : int
        The numerical degree of the approximation
    epsVal : float
        The epsilon value the coefficients have converged to
    rho : float
        The geometric rate of convergence of the coefficients
    """
    #Assumes we converged at the smaller degree.
    convergedDeg = int(3 * (len(coeff) + 1) / 4)
    #Calculate epsVal
    epsVal = 2*np.max(coeff[convergedDeg:])
    #Get the numerical degree. Make at least 1.
    nonZeroCoeffs = np.where(coeff > epsVal)[0]
    degree = 1 if len(nonZeroCoeffs) == 0 else max(1, nonZeroCoeffs[-1])
    #Calculate the min rate of convergence
    maxSpot = np.argmax(coeff)
    rho = (coeff[maxSpot]/epsVal)**(1/((degree - maxSpot) + 1))
    return degree, epsVal, rho
        
def getChebyshevDegree(f, a, b, absApproxTol, relApproxTol):
    """Compute the numerical Chebyshev degree of a function.

    Looks for the Chebyshev coefficients to converge to a value less than absApproxTol + supNorm*relApproxTol

    Parameters
    ----------
    f : function
        The function we wish to approximate.
    a : numpy array
        The lower bound on the interval.
    b : numpy array
        The upper bound on the interval.
    absApproxTol : float
        The absolute tolerance for convergence
    relApproxTol : float
        The absolute tolerance for convergence
    
    Returns
    -------
    chebDegrees : numpy array
        The numerical degree in each dimension.
    epsilons : numpy array
        The value the coefficients converged to in each dimension.
    rhos : numpy array
        The rate of convergence in each dimension.
    """    
    #Get the chebyshev degree of f
    dim = len(a)
    chebDegrees = [] #The approximation degree in each dimension
    epsilons = [] #What the approximation has converged to in each dimension
    rhos = [] #The rate of convergence in each dimsension
    #Find the degree in each dimension seperately
    for currDim in range(dim):
        #Use degree 5 in every other dimension. High enough that we are unlikely to have constant or
        #0 values everywhere
        degs = np.array([5]*dim)
        currGuess = 8 #Start at degree 8 in currDim
        tupleForChunk = tuple([i for i in range(currDim)] + [i for i in range(currDim+1,dim)])
        while True: #Double the degree in currDim until we converge
            #Approximate with degree currGuess in this dimension
            degs[currDim] = currGuess
            coeff, supNorm = interval_approximate_nd(f, degs, a, b, retSupNorm=True)
            #Average the contributions from the other dimensions
            coeffChunk = np.average(np.abs(coeff), axis=tupleForChunk)
            tol = absApproxTol + supNorm * relApproxTol
            
            currGuess *= 2
            #Check if we've started converging
            if not startedConverging(coeffChunk, tol):
                continue

            #Check if we've finally converged at degree n by comparing against the degree 2n-1.
            #Degree n and 2n+1 are unlikely to have higher degree terms alias into the same spot.
            degs[currDim] = currGuess - 1
            coeff2, supNorm2 = interval_approximate_nd(f, degs, a, b, retSupNorm=True)
            tol = absApproxTol + max(supNorm, supNorm2) * relApproxTol
            if not hasConverged(coeff, coeff2, tol):
                continue
            
            #Get the final degree from coeff2
            coeffChunk = np.average(np.abs(coeff2), axis=tupleForChunk)
            deg, eps, rho = getFinalDegree(coeffChunk)
            chebDegrees.append(deg)
            epsilons.append(eps)
            rhos.append(rho)
            break
    return np.array(chebDegrees), np.array(epsilons), np.array(rhos)

def getApproxError(degs, epsilons, rhos):
    """Compute the error of a Chebysev Approximation

    Takes the infinite sum of the terms not in the approximation.

    Parameters
    ----------
    degs: numpy array
        The degrees of the approximation.
    epsilons : numpy array
        What the approximation converged to in each dimension.
    rhos : numpy array
        The rate of convergence in each dimension.
    
    Returns
    -------
    approxError : float
        A bound on the approximation error from the unused terms
    """    
    approxError = 0
    #Power set of dimensions.
    for idxs in itertools.product(range(2), repeat=len(degs)):
        #Skip first which is all 0s.
        if np.sum(idxs) == 0:
            continue
        #Sum this sets contributions
        s = 1
        for i, used in enumerate(idxs):
            if used:
                s *= epsilons[i] / (1-1/rhos[i])
            else:
                s *= degs[i]
        #Divide by p[i] if only index i
        usedSpots = np.where(np.array(idxs) == 1)[0]
        if len(usedSpots) == 1:
            s /= rhos[usedSpots[0]]
        #Append to the error
        approxError += s
    return approxError

def chebApproximate(f, a, b, absApproxTol=1e-10, relApproxTol=1e-10):
    """Approximation a function on the interval [a,b]

    Parameters
    ----------
    f : function
        The function we wish to approximate.
    a : numpy array
        The lower bound on the interval.
    b : numpy array
        The upper bound on the interval.
    absApproxTol : float
        The absolute tolerance for convergence
    relApproxTol : float
        The absolute tolerance for convergence
    
    Returns
    -------
    coeff : numpy array
        A coefficient matrix of the chebyshev approximation
    error : float
        The error in the approximation from all higher degree terms we threw out.
    """
    degs, epsilons, rhos = getChebyshevDegree(f, a, b, absApproxTol, relApproxTol)
    return interval_approximate_nd(f, degs, a, b), getApproxError(degs, epsilons, rhos)