import numpy as np
from numba import njit
from itertools import product
#from IntervalChecks import constant_term_check

def getLinearTerms(M):
    """Helper Function, returns the linear terms of a matrix

    Uses the fact that the linear terms are indexed at 
    (0,0, ... ,0,1)
    (0,0, ... ,1,0)
    ...
    (0,1, ... ,0,0)
    (1,0, ... ,0,0)
    which are indexes
    1, n, n^2, ... when looking at M.ravel().

    Parameters
    ----------
    M : numpy array
        The coefficient array ot get the linear terms from

    Returns
    -------
    A 1D numpy array with the linear terms of M
    """
    spot = 1
    MArray = M.ravel()
    A = [MArray[spot]]
    for i in M.shape[1:][::-1]:
        spot *= i
        A.append(MArray[spot])
    return A[::-1]

def BoundingIntervalLinearSystem(Ms, errors, constant_term_check=False):
    """Finds a smaller region in which any root must be.

    Parameters
    ----------
    Ms : list of numpy arrays
        Each numpy array is the coefficient tensor of a chebyshev polynomials
    errors : iterable of floats
        The maximum error of chebyshev approximations

    Returns
    -------
    newInterval : numpy array
        The smaller interval where any root must be
    changed : bool
        Whether the interval has shrunk at all
    """
    #Get the matrix of the linear terms
    A = np.array([getLinearTerms(M) for M in Ms])
    #Get the Vector of the constant terms
    consts = np.array([M.ravel()[0] for M in Ms])
    #Get the Error of everything else combined.
    #TODO: We could have catastrophic cancelation on this subtraction. Add sum(abs(M))/2**52 to err for safety?
    linear_sums = np.sum(np.abs(A),axis=1)
    err = np.array([np.sum(np.abs(M))-abs(c)-l+e for M,e,c,l in zip(Ms,errors,consts,linear_sums)])
    #Constant term check inside (3)
    if constant_term_check in ["inside", 3] and np.any(np.abs(consts) > linear_sums + err):
        print("Root eliminated by method 3")
        return None, None
    #Solve for the extrema
    #We use the matrix inverse to find the width, so might as well use it both spots. Should be fine as dim is small.
    try:
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError as e: #If it's not invertible we can't zoom in
        if len(A) == 1:
            return np.array([[-1.0,1.0]]), False
        else:
            return np.vstack([[-1.0]*len(A),[1.0]*len(A)]), False
    center = -Ainv@consts
        
    #Ainv transforms the hyperrectangle of side lengths err into a parallelogram with these as the principal direction
    #So summing over them gets the farthest the parallelogram can reach in each dimension.
    width = np.sum(np.abs(Ainv*err),axis=1)
    a = center - width
    b = center + width
    #Bound at [-1,1]. TODO: Kate has a good way to bound this even more.
    a[a < -1] = -1.0
    b[b > 1] = 1.0
    changed = np.any(a > -1) or np.any(b < 1)
    return np.vstack([a,b]).T, changed

@njit
def isValidSpot(i,j):
    """Helper for makeMatrix.

    Parameters
    ----------
    i : number
        The row of the matrix
    j : number
        The col of the matrix

    Returns
    -------
    isValid : bool
        True if this is a spot in the matrix that I should be updating to make the Chebyshev Transformation Matrix.
        This means the index is on the upper diagonal of a matrix.
    """
    return i >= 0 and j >= i

@njit
def makeMatrix(n,a,b,subMatrix=None):
    """Creates the Chebyshev transformation matrix.

    Parameters
    ----------
    n : integer
        The size of the matrix to create. Will be the degree + 1. Must be at least 2.
    a : number
        The lower bound of the interval we are transforming onto
    b : number
        The upper bound of the interval we are transforming onto
    subMatrix : numpy array (optional)
        The mxm Chebyshev Transformation matrix for the same interval where m < n. Used to speed up construction if known.
    
    Returns
    -------
    M : numpy array
        The Chebyshev Transformation matrix to transform a polynomial of degree n-1 from [-1,1] to [a,b].
    """
    #Matrix creation with njit
    M = np.zeros(n*n)
    M = M.reshape(n,n)
    #Use the submatrix if exists
    startValue = 2
    if subMatrix is not None:
        M[:subMatrix.shape[0],:subMatrix.shape[1]] = subMatrix[:n,:n]
        startValue = min(2, min(subMatrix.shape[0], subMatrix.shape[1]))
    #Initial Values of the Matrix
    M[0,0] = 1
    M[0,1] = b
    M[1,1] = a
    #Use the reccurence relation
    #M[i,j] = 2*b*M[i,j-1] - M[i,j-2] + a*M[i-1,j-1] + a*M[i+1,j-1]*(2 if i==1 else 1)
    for j in range(startValue, n): #Loop over the columns starting at 2
        for i in range(j+1): #Loop over the rows on the upper diagonal
            val = 0
            if isValidSpot(i,j-2):
                val -= M[i,j-2]
            if isValidSpot(i-1,j-1):
                val += a * M[i-1,j-1] * (2 if i == 1 else 1)
            if isValidSpot(i,j-1):
                val += 2*b*M[i,j-1]
            if isValidSpot(i+1,j-1):
                val += a * M[i+1,j-1]
            M[i,j] = val
    return M

def transformPoint(x, a, b):
    """Transforms x from [-1,1] onto [a,b]
    
    This function guarantees the following two conditions.
    
    1. If x is exactly -1 or 1, the result will be exactly a or b, respectively. It does this right now with 
        explicit checks. We want this so that all the edges of our boundaries match up exactly.
    
    2. If a <= b and x1 <= x2, then transformPoint(x1, a, b) <= transformPoint(x2, a, b). 
        We want this to avoid throwing out an interval incorrectly. Because of this we can't use
        "return (b*(1.0+x)+a*(1.0-x))/2.0", which would solve condition 1.
    
    Parameters
    ----------
    x : float
        The point to transform
    a : float
        The min of the interval we are transforming to
    b : float
        The max of the interval we are transforming to
    
    Returns
    -------
    x : float
        The transformed point
    """
    if x == -1.0:
        return a
    elif x == 1.0:
        return b
    return ((b-a)*x+(b+a))/2

def transformArray(x, a, b):
    """Transforms x from [-1,1] onto [a,b]
    
    Does so using transformPoint, which gives the same guarantees as that function.
    
    Parameters
    ----------
    x : list or numpy array
        A list of points to transform
    a : float
        The min of the interval we are transforming to
    b : float
        The max of the interval we are transforming to
    
    Returns
    -------
    x : numpy array
        The transformed points
    """
    return np.array([transformPoint(x_, a, b) for x_ in x])

def getTransformPoints(a,b):
    """Given the new interval [a,b], gives c,d for reduction xHat = cx+d"""
    return (b-a)/2, (b+a)/2

def transformCheb(M, As, Bs):
    """Transforms the chebyshev coefficient matrix M to the new interval [As, Bs].

    Parameters
    ----------
    M : numpy array
        The chebyshev coefficient matrix
    As : iterable
        The min values of the interval we are transforming to
    Bs : iterable
        The max values of the interval we are transforming to
    
    Returns
    -------
    M : numpy array
        The coefficient matrix on the new interval
    """    
    #This just does the matrix multiplication on each dimension. Except it's by a tensor.
    for n,a,b in zip(M.shape,As,Bs):
        if a == 1 and b == 0: #No Transformation
            M = M.transpose(*([i for i in range(1,len(As))] + [0])) #Identity Martrix
        else:
            C = makeMatrix(n,a,b)
            M = np.tensordot(M, C, [0,1])
    return M

def transformChebToInterval(Ms, interval):
    """Transforms chebyshev coefficient matrices to a new interval.

    Parameters
    ----------
    Ms : list of numpy arrays
        The chebyshev coefficient matrices
    interval : numpy array
        The new interval to transform to
    
    Returns
    -------
    Ms : list of numpy arrays
        The chebyshev coefficient matrices on the new interval
    """    
    #Get the transform points
    As,Bs = np.array([getTransformPoints(*i) for i in interval]).T
    #Transform the chebyshev polynomials
    return [transformCheb(M, As,Bs) for M in Ms]
    
def zoomInOnIntervalIter(Ms, errors, result, constant_term_check=False):
    """One iteration of the linear check and transforming to a new interval.

    TODO: This should update the error as well, as the matrix multiplication could introduce error.

    Parameters
    ----------
    Ms : list of numpy arrays
        The chebyshev coefficient matrices
    errors : numpy array
        A bound on the error of each chebyshev approximation
    result : numpy array
        The current interval that the chebyshev approximations are valid for
    Returns
    -------
    Ms : list of numpy arrays
        The chebyshev coefficient matrices on the new interval
    result : numpy array
        The new interval that the chebyshev approximations are valid for
    changed : bool
        Whether the interval changed as a result of this step.
    """    
    dim = len(Ms)
    #Zoom in on the current interval
    interval, changed = BoundingIntervalLinearSystem(Ms, errors, constant_term_check=constant_term_check)
    #Check if we can throw out the whole thing
    if interval is None or np.any(interval[:,0] > interval[:,1]):
        return Ms, None, None
    #Check if we are done interating
    if not changed:
        return Ms, result, changed
    #Transform the chebyshev polynomials
    Ms = transformChebToInterval(Ms, interval)
    #result is the overall interval
    result = np.array([transformArray(i, *r) for i,r in zip(interval, result)])
    return Ms, result, changed
    
def getTransposeDims(dim,transformDim):
    """Helper function for chebTransform1D"""
    return [i for i in range(transformDim,dim)] + [i for i in range(transformDim)]

def chebTransform1D(M, C, transformDim):
    """Transform a chebyshev polynomial in a single dimension"""
    dim = M.ndim
    T1 = getTransposeDims(dim,transformDim)
    T2 = getTransposeDims(dim,dim-transformDim-1)
    return np.tensordot(M.transpose(*T1), C, [0,1]).transpose(*T2)

def getInverseOrder(order, dim):
    """Helper function to make the subdivide order match the subdivideInterval order"""
    order = 2**(len(order)-1 - order)
    newOrder = np.array([i@order for i in product([0,1],repeat=dim)])
    invOrder = np.zeros_like(newOrder)
    invOrder[newOrder] = np.arange(len(newOrder))
    return tuple(invOrder)
    
class Subdivider():
    #This class handles subdividing and stores the precomputed matrices to save time.
    def __init__(self):
        self.RAND = 0.5139303900908738 #Don't subdivide exactly in half.
        self.precomputedArrayDeg = 1 #We don't compute the first 2
        self.subdivisionPoint =  self.RAND * 2 - 1
        self.transformPoints1 = getTransformPoints(-1, self.subdivisionPoint)
        self.transformPoints2 = getTransformPoints(self.subdivisionPoint, 1)
        self.C1 = np.zeros([0,0]) #The transformation onto [-1, self.subdivisionPoint]
        self.C2 = np.zeros([0,0]) #The transformation onto [self.subdivisionPoint, 1]
        self.precomputeTransformMatrices(3) #Precompute the first 3 cause we'll probably need them
        #Note that a transformation of a lower degree is just the submatrix of the higher degree transformation
        #So we can just store the highest degree transformation we have to save on space.
        #And we can compute the higher degree transformation starting at the degree we already have.

    def precomputeTransformMatrices(self, degree):
        #Precomputes and store the Chebyshev Transformation Matrices up to a certain degree
        if degree < self.C1.shape[0]:
            return
        self.C1 = makeMatrix(degree,*self.transformPoints1, self.C1)
        self.C2 = makeMatrix(degree,*self.transformPoints2, self.C2)
        
    def subdivideInterval(self, interval):
        #Get the new interval that will correspond with the new polynomials
        results = [interval]
        for thisDim in range(len(interval)):
            newResults = []
            for oldInterval in results:
                #Transform the interval
                transformedMidpoint = transformPoint(self.subdivisionPoint, *oldInterval[thisDim])
                newInterval = oldInterval.copy()
                newInterval[thisDim] = [oldInterval[thisDim][0], transformedMidpoint]
                newResults.append(newInterval.copy())
                newInterval[thisDim] = [transformedMidpoint, oldInterval[thisDim][1]]
                newResults.append(newInterval.copy())
            results = newResults
        return results
        
    def subdivide(self, M):
        #Get the new Chebyshev approximations on the 2^n subintervals
        dim = M.ndim
        degs = M.shape
        order = np.argsort(degs)[::-1] #We want to subdivide from highest degree to lowest.
        #Precompute transform matrices if we don't have this degree yet
        self.precomputeTransformMatrices(degs[order[0]])
        #Iterate through the dimensions, highest degree first.
        resultMs = [M]
        for thisDim in order:
            thisDeg = M.shape[thisDim]
            newResults = []
            for T in resultMs:
                #Transform the polys
                #TODO: This should update the error as well, as the matrix multiplication could introduce error.
                newResults.append(chebTransform1D(T, self.C1[:thisDeg, :thisDeg], thisDim))
                newResults.append(chebTransform1D(T, self.C2[:thisDeg, :thisDeg], thisDim))
            resultMs = newResults
        if dim == 1:
            return resultMs #Already ordered because there's only 1.
        else:
            #Order the polynomials so they match the intervals in subdivideInterval
            return [resultMs[i] for i in getInverseOrder(order, dim)]
        
#The subdivider class. Stores the precomputed matrices.
mySubdivider = Subdivider()
        
def trimMs(Ms, errors, absErrorIncrease, relErrorIncrease):
    """Reduces the degree of chebyshev approximations and adds the resulting error to errors

    If the incoming error is E, will increase the error by at most max(relErrorIncrease * E, absErrorIncrease)

    Parameters
    ----------
    Ms : list of numpy arrays
        The chebyshev approximations of the functions
    errors : numpy array
        The max error of the chebyshev approximation from the function on the interval
    absErrorIncrease : float
        The largest increase in error allowed
    relErrorIncrease : float
        The largest relative increase in error allowed
    
    Returns
    -------
    No return value, the Ms and errors and changed in place.
    """
    dim = Ms[0].ndim
    for polyNum in range(len(Ms)): #Loop through the polynomials
        allowedErrorIncrease = max(relErrorIncrease * errors[polyNum], absErrorIncrease)
        totalSum = np.sum(np.abs(Ms[polyNum]))
        #Use these to look at a slice of the highest degree in the dimension we want to trim
        slices = [slice(None) for i in range(dim)]
        for currDim in range(dim): #Loop over the dimensions
            slices[currDim] = -1
            lastSum = np.sum(np.abs(Ms[polyNum][tuple(slices)]))
            #Check if the sum of the highest degree is of low error
            #Keeps the degree at least 2
            while lastSum < allowedErrorIncrease and Ms[polyNum].shape[currDim] > 3:
                allowedErrorIncrease -= lastSum #Update the error we are allowed
                errors[polyNum] += lastSum #Update the error
                slices[currDim] = slice(None,-1)
                Ms[polyNum] = Ms[polyNum][tuple(slices)] #Trim the polynomial
                slices[currDim] = -1
                lastSum = np.sum(np.abs(Ms[polyNum][tuple(slices)]))
            slices[currDim] = slice(None)

def combineTouchingIntervals(intervals):
    """Combines intervals that are touching into one interval the contains them

    It might be useful to pass in the boundaries so we can check which itervals lie on a boundary and which ones don't.
    That will be a floating point equality check but each interval on the boundary, but as long as transformPoint()
    always handles the -1,1 case exactly, this should be fine, as the boundary points will be exactly the same everywhere.

    Parameters
    ----------
    intervals : list of lists of numpy arrays
        Each list corresponds to one of the 2^n subintervals from the last step of subdivision, and contains a list of
        all the intervals in the subinterval in which there could be a root. Intervals in the same list will not be touching.
    
    Returns
    -------
    intervals : list of numpy arrays
        A new list of intervals such that each interval in the any of the old lists is contained in some interval in
        the new list, but none of the intervals in the new list are touching.
    """
    #TODO: WRITE THIS!!!
    newIntervals = []
    for interval in intervals:
        newIntervals += interval
    return newIntervals

def shouldStopSubdivision(interval):
    #TODO: WRITE THIS!!!
    #In 1D a good check could be if the linear term is less than the error term (or close to it).
    #Not sure how to extend that to ND yet.
    #For now just checks if the interval is small. This won't work if there is a large error term.
    return np.all(interval[:,1]-interval[:,0] < 1e-10)

def solvePolyRecursive(Ms, interval, errors, constant_term_check=False):
    """Recursively finds regions in which any common roots of functions must be using subdivision

    Parameters
    ----------
    Ms : list of numpy arrays
        The chebyshev approximations of the functions
    interval : numpy array
        The interval on which the chebyshev approximations are valid.
    errors : numpy array
        The max error of the chebyshev approximation from the function on the interval
    constant_term_check : False, "outside", "inside"
        Decides whether or not and where to do constant term check
    Returns
    -------
    boundingBoxes : list of numpy arrays (optional)
        Each element of the list is an interval in which there may be a root.
    """
    # Constant term check outside (2)
    if constant_term_check in ["outside", 2]:
        #Same check as in bounding interval linear system
        consts = np.array([M.ravel()[0] for M in Ms])
        err = np.array([np.sum(np.abs(M))-abs(c)+e for M,e,c in zip(Ms,errors,consts)])
        if np.any(np.abs(consts) > err):
            print("Root eliminated by method 2")
            return []
        #for i in range(len(Ms)):
        #    if not constant_term_check(Ms[i], errors[i]):
                #print("An interval was eliminated by method 2")
                #return []

    #The random numbers used below. TODO: Choose these better
    #Error For Trim trimMs
    trimErrorAbsBound = 1e-16
    trimErrorRelBound = 1e-16
    #How long we are allowed to zoom before giving up
    maxZoomCount1 = 10
    maxZoomCount2 = 50
    #When to stop, again, maybe this should just be 0???
    minIntervalSize = 1e-16
    #Assume that once we have shrunk this interval this much, we will be able to shrink it all the way.
    #The is something to look into.
    zoomRatioToZip = 0.01
    
    #Trim
    trimMs(Ms, errors, trimErrorAbsBound, trimErrorRelBound)
    
    #Solve
    dim = Ms[0].ndim
    changed = True
    zoomCount = 0
    originalIntervalSize = np.product(interval[:,1]-interval[:,0])
    #The choosing when to stop zooming logic is really ugly. Needs more analysis.
    #Keep zooming while it's larger than minIntervalSize.
    while changed and np.max(interval[:,1] - interval[:,0]) > minIntervalSize:
        #If we've zoomed more than maxZoomCount1 and haven't shrunk the size by zoomRatioToZip, assume
        #we aren't making progress and subdivide. Once we get to zoomRatioToZip, assume we will just converge
        #quickly and zoom all the way by maxZoomCount2.
        if zoomCount > maxZoomCount1:
            newIntervalSize = np.product(interval[:,1]-interval[:,0])
            zoomRatio = (newIntervalSize / originalIntervalSize) ** (1/len(Ms))
            if zoomRatio >= zoomRatioToZip:
                break
            elif zoomCount > maxZoomCount2:
                break
        #Zoom in until we stop changing or we hit machine epsilon
        Ms, interval, changed = zoomInOnIntervalIter(Ms, errors, interval, constant_term_check=constant_term_check)
        if interval is None: #Throw out the interval
            return []
        zoomCount += 1
    if shouldStopSubdivision(interval):
        #Return the interval. Maybe we should return the linear approximation of the root here as well as the interval?
        #Might be better than just taking the midpoint later.
        return [interval]
    else:
        #Otherwise, Subdivide
        result = []
        #Get the new intervals and polynomials
        newInts = mySubdivider.subdivideInterval(interval)
        allMs = [mySubdivider.subdivide(M) for M in Ms]
        #Run each interval
        for i in range(len(newInts)):
            result.append(solvePolyRecursive([allM[i] for allM in allMs], newInts[i], errors))
        return combineTouchingIntervals(result)

def solveChebyshevSubdivision(Ms, interval, errors, returnBoundingBoxes = False, constant_term_check=False):
    """Finds regions in which any common roots of functions must be

    Parameters
    ----------
    Ms : list of numpy arrays
        The chebyshev approximations of the functions
    interval : numpy array
        The interval on which the chebyshev approximations are valid.
    errors : numpy array
        The max error of the chebyshev approximation from the function on the interval
    returnBoundingBoxes : bool
        If True, returns the bounding boxes around each root as well as the roots.
    constant_term_check : False, "inside", or "outside"
        Decides whether or not and where to carry out the constant term check
        if False, does not perform constant term check
        if "inside", performs constant term check every time the interval shrinks
        if "outside", performs constant term check at the beginning and after each subdivision
    
    Returns
    -------
    roots : list
        The roots
    boundingBoxes : list of numpy arrays (optional)
        Each element of the list is an interval in which there may be a root.
    """
    #Transform interval to [-1,1]^n
    transformedMs = transformChebToInterval(Ms, interval)
    boundingBoxes = solvePolyRecursive(transformedMs, interval, errors, constant_term_check=constant_term_check)
    #TODO: Have an options to reRun all the bounding boxes with tight precision after a first run at lower precision.
    #For Example: 
    #boundingBoxes = [solvePolyRecursive(transformedMs, box, errors, <tigher tolerance params>) for box in boundingBoxes]
    roots = [(interval[:,1] + interval[:,0]) / 2 for interval in boundingBoxes]
    if returnBoundingBoxes:
        return roots, boundingBoxes
    else:
        return roots