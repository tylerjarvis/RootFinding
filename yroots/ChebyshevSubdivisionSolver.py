"""Chebyshev subdivision root solver.

Implements the recursive solver invoked by :func:`yroots.Combined_Solver.solve`. The
core entry point is :func:`solveChebyshevSubdivision`; the rest of the module
provides the supporting primitives (linear-system bounding, transformation of
Chebyshev coefficients, subdivision bookkeeping via :class:`TrackedInterval`,
and an optional multilevel parallel driver).
"""
import numpy as np
from numba import njit, float64
from numba.types import UniTuple
from itertools import product
from yroots.QuadraticCheck import quadratic_check
import copy
import threading
import warnings

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

@dataclass
class SolveTask:
    """One unit of work for the multilevel parallel driver.

    Holds the Chebyshev coefficient tensors, the :class:`TrackedInterval` they are being
    solved on, the per-poly approximation error bounds, and the bookkeeping (parent id,
    subdivision depth) needed to reassemble results once child tasks finish.
    """
    Ms: object
    trackedInterval: object
    errors: object
    parent_id: int | None = None
    level: int = 0

@dataclass
class SubdivisionState:
    """Snapshot of a parent interval waiting on its children to finish.

    Stores everything the driver needs to reconstruct and finalize the parent
    once all child tasks have completed.
    """
    originalMs: object
    originalInterval: object
    trackedInterval: object
    errors: object
    solverOptions: object
    isFinalStep: bool

@dataclass
class TaskResult:
    """Result of a single solve step.

    If ``childTasks`` is empty, this task is finished. Otherwise the task subdivided
    and the driver must solve the children before finishing the parent.
    """
    interior: list
    exterior: list
    childTasks: list
    subdivisionState: SubdivisionState | None = None

class StallCounter():
    """Counts the intervals a solve returned because subdividing them no longer shrinks them.

    An interval stalls when every dimension it would be split in is too narrow to split: the split
    point rounds onto an endpoint, so one half is the whole interval. (getSubdivisionDims stops
    splitting dimensions narrower than about 1e-8 while any other dimension is wider, so an interval
    can stall with one dimension at that width.) An isolated root that is multiple or badly
    conditioned stalls at most a few intervals. A curve of roots, an identically zero function or
    two equations that agree to within rounding error stall an interval all along the solution set,
    far too many to finish, so stalledIntervalResult raises once the count passes
    SolverOptions.maxStalledIntervals.

    SolverOptions.copy is shallow, so every copy made during one solve shares one counter. The lock
    keeps the count right when the parallel driver's threads stall intervals at the same time.
    """
    def __init__(self):
        self.count = 0
        self.lock = threading.Lock()

    def increment(self):
        """Adds one stalled interval and returns the new count."""
        with self.lock:
            self.count += 1
            return self.count

class SolverOptions():
    """Settings for running interval checks, transformations, and subdivision in solvePolyRecursive.

    Parameters
    ----------
    verbose : bool
        Defaults to False. Whether or not to output progress of solving to the terminal.
    exact : bool
        Defaults to False. Whether the transformation in TransformChebInPlaceND should minimize error.
    constant_check : bool
        Defaults to True. Whether or not to run constant term check after each subdivision.
    low_dim_quadratic_check : bool
        Defaults to True. Whether or not to run quadratic check in dim 2, 3.
    all_dim_quadratic_check : bool
        Defaults to False. Whether or not to run quadratic check in dim >= 4.
    maxZoomCount : int
        Maximum number of zooms allowed before subdividing (prevents infinite infinitesimal shrinking).
    level : int
        Depth of subdivision for the given interval.
    max_cpu : int
        Defaults to 1. Maximum number of worker processes the multilevel parallel driver may use.
    allowParallel : bool
        Defaults to True. Whether parallel dispatch is allowed for this solve. Workers flip this to
        False on their copy of the options to prevent nested parallelism.
    parallel_depth : int
        Subdivision depth below which child tasks are pushed to the process pool. Tasks at or beyond
        this depth solve their children serially in the worker, avoiding scheduling overhead on
        tiny tasks. Defaults to 0 (i.e. fully serial).
    maxStalledIntervals : int
        Defaults to 50. The most intervals a solve may return because subdividing them no longer
        shrinks them (see StallCounter). Past this many the roots are not finitely many isolated
        points the solver can separate, and the solve raises a ValueError.
    stallCounter : StallCounter
        Counts stalled intervals. Shared by every copy of the options made during one solve.
    """
    
    def __init__(self):
        #Init all the Options to default value
        self.verbose = False
        self.exact = False
        self.constant_check = True
        self.low_dim_quadratic_check = True
        self.all_dim_quadratic_check = False
        self.maxZoomCount = 25
        self.level = 0
        self.useFinalStep = True

        self.max_cpu = 1
        self.allowParallel = True
        self.parallel_depth = 0
        self.maxStalledIntervals = 50
        self.stallCounter = StallCounter()

    def copy(self):
        return copy.copy(self) #Shallow copy: basic types, plus the stallCounter every copy shares

@njit(cache=True)
def TransformChebInPlace1D(coeffs, alpha, beta):
    """Applies the transformation alpha*x + beta to one dimension of a Chebyshev approximation.

    Recursively finds each column of the transformation matrix C from the previous two columns
    and then performs entrywise matrix multiplication for each entry of the column, thus enabling
    the transformation to occur while only retaining three columns of C in memory at a time.

    Parameters
    ----------
    coeffs : numpy array
        The coefficient array
    alpha : double
        The scaler of the transformation
    beta : double
        The shifting of the transformation

    Returns
    -------
    transformedCoeffs : numpy array
        The new coefficient array following the transformation
    """
    transformedCoeffs = np.zeros_like(coeffs)

    #Initialize three arrays to represent subsequent columns of the transformation matrix.
    arr1 = np.zeros(len(coeffs))
    arr2 = np.zeros(len(coeffs))
    arr3 = np.zeros(len(coeffs))

    #The first column of the transformation matrix C. Since T_0(alpha*x + beta) = T_0(x) = 1 has 1 in the top entry and 0's elsewhere.
    arr1[0] = 1.
    transformedCoeffs[0] = coeffs[0] # arr1[0] * coeffs[0] (matrix multiplication step)
    #The second column of C. Note that T_1(alpha*x + beta) = alpha*T_1(x) + beta*T_0(x).
    arr2[0] = beta
    arr2[1] = alpha
    transformedCoeffs[0] += beta * coeffs[1] # arr2[0] * coeffs[1] (matrix muliplication)
    transformedCoeffs[1] += alpha * coeffs[1] # arr2[1] * coeffs[1] (matrix multiplication)

    maxRow = 2
    for col in range(2, len(coeffs)): # For each column, calculate each entry and do matrix mult
        thisCoeff = coeffs[col] # the row of coeffs corresponding to the column col of C (for matrix mult)
        # The first entry
        arr3[0] = -arr1[0] + alpha*arr2[1] + 2*beta*arr2[0]
        transformedCoeffs[0] += thisCoeff * arr3[0]

        # The second entry
        if maxRow > 2:
            arr3[1] = -arr1[1] + alpha*(2*arr2[0] + arr2[2]) + 2*beta*arr2[1]
            transformedCoeffs[1] += thisCoeff * arr3[1]

        # All middle entries
        for i in range(2, maxRow - 1):
            arr3[i] = -arr1[i] + alpha*(arr2[i-1] + arr2[i+1]) + 2*beta*arr2[i]
            transformedCoeffs[i] += thisCoeff * arr3[i]

        # The second to last entry
        i = maxRow - 1
        arr3[i] = -arr1[i] + (2 if i == 1 else 1)*alpha*(arr2[i-1]) + 2*beta*arr2[i]
        transformedCoeffs[i] += thisCoeff * arr3[i]

        #The last entry
        finalVal = alpha*arr2[i]
        # This final entry is typically very small. If it is essentially machine epsilon,
        # zero it out to save calculations.
        if abs(finalVal) > 1e-16: #TODO: Justify this val!
            arr3[maxRow] = finalVal
            transformedCoeffs[maxRow] += thisCoeff * finalVal
            maxRow += 1 # Next column will have one more entry than the current column.

        # Save the values of arr2 and arr3 to arr1 and arr2 to get ready for calculating the next column.
        arr = arr1
        arr1 = arr2
        arr2 = arr3
        arr3 = arr
    #
    return transformedCoeffs[:maxRow]

@njit(cache=True)
def TransformChebInPlace1DErrorFree(coeffs, alpha, beta):
    """Applies the transformation alpha*x + beta to the Chebyshev polynomial coeffs with minimal error.

    This function is identical to TransformChebInPlace1D except that this function is more careful to
    minimize error by calling on functions to more precisely perform the multiplication and addition.

    Parameters
    ----------
    coeffs : numpy array
        The coefficient array
    alpha : double
        The scaler of the transformation
    beta : double
        The shifting of the transformation

    Returns
    -------
    coeffs : numpy array
        The new coefficient array following the transformation
    """
    if alpha == 0.5 and abs(beta) == 0.5:
        return TransformChebInPlace1DErrorFreeSplit(coeffs, np.sign(beta))
    transformedCoeffs = np.zeros_like(coeffs)
    arr1 = np.zeros(len(coeffs))
    arr2 = np.zeros(len(coeffs))
    arr3 = np.zeros(len(coeffs))
    arr1E = np.zeros(len(coeffs))
    arr2E = np.zeros(len(coeffs))
    arr3E = np.zeros(len(coeffs))

    alpha1,alpha2 = Split(alpha)
    beta1,beta2 = Split(beta)

    #The first array
    arr1[0] = 1.
    transformedCoeffs[0] = coeffs[0]
    #The second array
    arr2[0] = beta
    arr2[1] = alpha
    transformedCoeffs[0] += beta * coeffs[1]
    transformedCoeffs[1] += alpha * coeffs[1]
    #Loop
    maxRow = 2
    for col in range(2, len(coeffs)):
        thisCoeff = coeffs[col]

        #Get the next arr from arr1 and arr2

        #The 0 spot
        # Calculate and store arr3[0] = -arr1[0] + alpha*arr2[1] + 2*beta*arr2[0]
        V1, E1 = TwoProdWithSplit(beta, 2*arr2[0], beta1, beta2)
        V2, E2 = TwoProdWithSplit(alpha, arr2[1], alpha1, alpha2)
        V3, E3 = TwoSum(V1, V2)
        V4, E4 = TwoSum(V3, -arr1[0])
        arr3[0] = V4
        # Now sum the error associated with this calculation and add it to the calculated value,
        # then perform the matrix multiplication associated with this entry.
        arr3E[0] = -arr1E[0] + alpha*arr2E[1] + 2*beta*arr2E[0] + E1 + E2 + E3 + E4
        transformedCoeffs[0] += thisCoeff * (arr3[0] + arr3E[0])

        # The procedure associated with minimizing error is the same for subsequent spots.
        #The 1 spot
        if maxRow > 2:
            #arr3[1] = -arr1[1] + alpha*(2*arr2[0] + arr2[2]) + 2*beta*arr2[1]
            V1, E1 = TwoSum(2*arr2[0], arr2[2])
            V2, E2 = TwoProdWithSplit(beta, 2*arr2[1], beta1, beta2)
            V3, E3 = TwoProdWithSplit(alpha, V1, alpha1, alpha2)
            V4, E4 = TwoSum(V2, V3)
            V5, E5 = TwoSum(V4, -arr1[1])
            arr3[1] = V5
            arr3E[1] = -arr1E[1] + alpha*(2*arr2E[0] + arr2E[2] + E1) + 2*beta*arr2E[1] + E2 + E3 + E4 + E5
            transformedCoeffs[1] += thisCoeff * (arr3[1] + arr3E[1])

        #The middle spots
        for i in range(2, maxRow - 1):
            #arr3[i] = -arr1[i] + alpha*(arr2[i-1] + arr2[i+1]) + 2*beta*arr2[i]
            V1, E1 = TwoSum(arr2[i-1], arr2[i+1])
            V2, E2 = TwoProdWithSplit(beta, 2*arr2[i], beta1, beta2)
            V3, E3 = TwoProdWithSplit(alpha, V1, alpha1, alpha2)
            V4, E4 = TwoSum(V2, V3)
            V5, E5 = TwoSum(V4, -arr1[i])
            arr3[i] = V5
            arr3E[i] = -arr1E[i] + alpha*(arr2E[i-1] + arr2E[i+1] + E1) + 2*beta*arr2E[i] + E2 + E3 + E4 + E5
            transformedCoeffs[i] += thisCoeff * (arr3[i] + arr3E[i])

        #The second to last spot
        i = maxRow - 1
        C1 = (2 if i == 1 else 1)
        #arr3[i] = -arr1[i] + C1*alpha*(arr2[i-1]) + 2*beta*arr2[i]
        V1, E1 = TwoProdWithSplit(beta, 2*arr2[i], beta1, beta2)
        V2, E2 = TwoProdWithSplit(alpha, C1*arr2[i-1], alpha1, alpha2)
        V3, E3 = TwoSum(V1, V2)
        V4, E4 = TwoSum(V3, -arr1[i])
        arr3[i] = V4
        arr3E[i] = -arr1E[i] + C1*alpha*arr2E[i-1] + 2*beta*arr2E[i] + E1 + E2 + E3 + E4
        transformedCoeffs[i] += thisCoeff * (arr3[i] + arr3E[i])

        #The last spot
        finalVal, finalValE = TwoProdWithSplit(alpha, arr2[i], alpha1, alpha2)
        arr3E[maxRow] = finalValE + alpha * arr2E[i]
        arr3[maxRow] = finalVal
        transformedCoeffs[maxRow] += thisCoeff * (arr3[maxRow] + arr3E[maxRow])
        if abs(arr3[maxRow] + arr3E[maxRow]) > 1e-32: #TODO: Justify this val!
            maxRow += 1

        #Rotate the vectors
        arr = arr1
        arr1 = arr2
        arr2 = arr3
        arr3 = arr
        arr = arr1E
        arr1E = arr2E
        arr2E = arr3E
        arr3E = arr
    return transformedCoeffs[:maxRow]

@njit(cache=True)
def TransformChebInPlace1DErrorFreeSplit(coeffs, betaSign):
    """Applies the transformation 0.5*x +- 0.5 to the Chebyshev polynomial coeffs with minimal error.

    This function is a special case of TransformChebInPlace1DErrorFree used to minimize computation
    when alpha = 0.5 and beta = +- 0.5

    Parameters
    ----------
    coeffs : numpy array
        The coefficient array
    betaSign : int
        1 if beta = 0.5; -1 if beta is -0.5

    Returns
    -------
    coeffs : numpy array
        The new coefficient array following the transformation

    """
    transformedCoeffs = np.zeros_like(coeffs)
    arr1 = np.zeros(len(coeffs))
    arr2 = np.zeros(len(coeffs))
    arr3 = np.zeros(len(coeffs))
    arr1E = np.zeros(len(coeffs))
    arr2E = np.zeros(len(coeffs))
    arr3E = np.zeros(len(coeffs))

    #The first array
    arr1[0] = 1.
    transformedCoeffs[0] = coeffs[0]
    #The second array
    arr2[0] = betaSign*0.5
    arr2[1] = 0.5
    transformedCoeffs[0] += betaSign*coeffs[1]/2
    transformedCoeffs[1] += coeffs[1]/2
    #Loop
    maxRow = 2
    for col in range(2, len(coeffs)):
        thisCoeff = coeffs[col]
        #Get the next arr from arr1 and arr2

        #The 0 spot
        #arr3[0] = -arr1[0] + alpha*arr2[1] + 2*beta*arr2[0]
        V1, E1 = TwoSum(arr2[1]/2, betaSign*arr2[0])
        V2, E2 = TwoSum(V1, -arr1[0])
        arr3[0] = V2
        arr3E[0] = -arr1E[0] + arr2E[1]/2 + betaSign*arr2E[0] + E1 + E2
        transformedCoeffs[0] += thisCoeff * (arr3[0] + arr3E[0])

        #The 1 spot
        if maxRow > 2:
            #arr3[1] = -arr1[1] + alpha*(2*arr2[0] + arr2[2]) + 2*beta*arr2[1]
            V1, E1 = TwoSum(arr2[0], arr2[2]/2)
            V2, E2 = TwoSum(V1, betaSign*arr2[1])
            V3, E3 = TwoSum(V2, -arr1[1])
            arr3[1] = V3
            arr3E[1] = -arr1E[1] + arr2E[0] + arr2E[2]/2 + betaSign*arr2E[1] + E1 + E2 + E3
            transformedCoeffs[1] += thisCoeff * (arr3[1] + arr3E[1])

        #The middle spots
        for i in range(2, maxRow - 1):
            #arr3[i] = -arr1[i] + alpha*(arr2[i-1] + arr2[i+1]) + 2*beta*arr2[i]
            V1, E1 = TwoSum(arr2[i-1], arr2[i+1])
            V2, E2 = TwoSum(V1/2, betaSign*arr2[i])
            V3, E3 = TwoSum(V2, -arr1[i])
            arr3[i] = V3
            arr3E[i] = -arr1E[i] + (arr2E[i-1] + arr2E[i+1] + E1)/2 + betaSign*arr2E[i] + E2 + E3
            transformedCoeffs[i] += thisCoeff * (arr3[i] + arr3E[i])

        #The second to last spot
        i = maxRow - 1
        C1 = (1 if i == 1 else 0.5)
        #arr3[i] = -arr1[i] + C1*alpha*(arr2[i-1]) + 2*beta*arr2[i]
        V1, E1 = TwoSum(C1*arr2[i-1], betaSign*arr2[i])
        V2, E2 = TwoSum(V1, -arr1[i])
        arr3[i] = V2
        arr3E[i] = -arr1E[i] + C1*arr2E[i-1] + betaSign*arr2E[i] + E1 + E2
        transformedCoeffs[i] += thisCoeff * (arr3[i] + arr3E[i])

        #The last spot
        arr3[maxRow] = arr2[i]/2
        arr3E[maxRow] = arr2E[i] / 2
        transformedCoeffs[maxRow] += thisCoeff * (arr3[maxRow] + arr3E[maxRow])
        if abs(arr3[maxRow] + arr3E[maxRow]) > 1e-32: #TODO: Justify this val!
            maxRow += 1

        #Rotate the vectors
        arr = arr1
        arr1 = arr2
        arr2 = arr3
        arr3 = arr
        arr = arr1E
        arr1E = arr2E
        arr2E = arr3E
        arr3E = arr
    return transformedCoeffs[:maxRow]

#Transpose orders used by TransformChebInPlaceND, keyed by (ndim, dim). See getTransposeOrders.
_transposeOrders = {}

def getTransposeOrders(ndim, dim):
    """Gets the axis orders that move dimension dim to the front and then put it back.

    The orders depend only on ``ndim`` and ``dim``, not on the coefficients, so they are built
    once per pair and reused. Rebuilding them per transformation costs several array
    allocations, which at these tensor sizes is comparable to the transformation itself.

    Parameters
    ----------
    ndim : int
        The number of dimensions of the tensor being transformed.
    dim : int
        The dimension to move to the front.

    Returns
    -------
    order : tuple of ints
        The axis order that moves dimension dim to the front.
    backOrder : tuple of ints
        The axis order that undoes it.
    """
    orders = _transposeOrders.get((ndim, dim))
    if orders is None:
        # Move the current dimension to the dim 0 spot in the np array.
        order = [dim] + [i for i in range(dim)] + [i for i in range(dim+1, ndim)]
        # Then transpose with the inverted order after the transformation occurs.
        backOrder = [0]*ndim
        for i, d in enumerate(order):
            backOrder[d] = i
        orders = (tuple(order), tuple(backOrder))
        _transposeOrders[(ndim, dim)] = orders
    return orders

def TransformChebInPlaceND(coeffs, dim, alpha, beta, exact):
    """Transforms a single dimension of a Chebyshev approximation for a polynomial.

    Parameters
    ----------
    coeffs : numpy array
        The coefficient tensor to transform
    dim : int
        The index of the dimension to transform
    alpha: double
        The scaler of the transformation
    beta: double
        The shifting of the transformation
    exact: bool
        Whether to perform the transformation with higher precision to minimize error

    Returns
    -------
    transformedCoeffs : numpy array
        The new coefficient array following the transformation
    """

    #TODO: Could we calculate the allowed error beforehand and pass it in here?
    #TODO: Make this work for the power basis polynomials
    if (alpha == 1.0 and beta == 0.0) or coeffs.shape[dim] == 1:
        return coeffs # No need to transform if the degree of dim is 0 or transformation is the identity.
    TransformFunc = TransformChebInPlace1DErrorFree if exact else TransformChebInPlace1D
    if dim == 0:
        return TransformFunc(coeffs, alpha, beta)
    else: # Need to transpose the matrix to line up the multiplication for the current dim
        order, backOrder = getTransposeOrders(coeffs.ndim, dim)
        return TransformFunc(coeffs.transpose(order), alpha, beta).transpose(backOrder)

@njit(float64[:,:](float64[:,:], float64[:,:]), cache=True)
def applySubInterval(interval, subInterval):
    """Shrinks interval down to subInterval in place and returns the transformation that did it.

    subInterval is given in the coordinates of interval, where -1 and 1 are its endpoints. Each
    endpoint of the result is ``alpha*x + beta`` for the alpha and beta of that dimension, except
    that an x of exactly +-1 is snapped to the matching endpoint of interval rather than being
    put through the arithmetic, so that a subinterval that reaches the edge stays exactly on it.

    Compiled because it runs once per zoom and once per subdivision on arrays of only a few
    entries, where numpy's per-operation overhead is what dominates.

    Parameters
    ----------
    interval : numpy array
        The interval to shrink, shape ``(ndim, 2)``. Modified in place.
    subInterval : numpy array
        The subinterval to shrink it to, shape ``(ndim, 2)``, in ``[-1,1]`` coordinates.

    Returns
    -------
    transform : numpy array
        Shape ``(2, ndim)``: the alpha of each dimension followed by the beta of each dimension.
    """
    ndim = interval.shape[0]
    transform = np.empty((2, ndim))
    for d in range(ndim):
        a1 = subInterval[d,0]
        b1 = subInterval[d,1]
        a2 = interval[d,0]
        b2 = interval[d,1]
        transform[0,d] = (b1-a1)/2
        transform[1,d] = (b1+a1)/2
        alpha2 = (b2-a2)/2
        beta2 = (b2+a2)/2
        #Be exact if the endpoint is +-1
        if a1 == -1.0:
            newLower = a2
        elif a1 == 1.0:
            newLower = b2
        else:
            newLower = alpha2*a1 + beta2
        #An upper bound of -1 collapses onto the lower bound we just computed, not the old one.
        if b1 == -1.0:
            newUpper = newLower
        elif b1 == 1.0:
            newUpper = b2
        else:
            newUpper = alpha2*b1 + beta2
        interval[d,0] = newLower
        interval[d,1] = newUpper
    return transform

@njit(cache=True)
def applyTransforms(topIntervalT, transforms):
    """Applies a chain of transformations back onto the original interval, tracking the error.

    Each transformation is applied with a two-product and a two-sum so that the rounding error
    it introduces is carried along separately instead of being lost, exactly as the original
    numpy version did -- but as one compiled pass over the chain rather than a dozen array
    temporaries per link.

    Parameters
    ----------
    topIntervalT : numpy array
        The original interval transposed, shape ``(2, ndim)``.
    transforms : numpy array
        Shape ``(n, 2, ndim)``: the alphas and betas of each transformation, oldest first.
        They are applied newest first.

    Returns
    -------
    finalInterval : numpy array
        The transformed interval, shape ``(2, ndim)``, before the error is added back.
    finalIntervalError : numpy array
        The accumulated rounding error of each entry, same shape.
    """
    finalInterval = topIntervalT.copy()
    finalIntervalError = np.zeros_like(finalInterval)
    ndim = finalInterval.shape[1]
    for k in range(transforms.shape[0]-1, -1, -1): # Iteratively apply each saved transform
        for j in range(ndim):
            alpha = transforms[k,0,j]
            beta = transforms[k,1,j]
            for i in range(2):
                val, temp = TwoProd(finalInterval[i,j], alpha)
                finalIntervalError[i,j] = alpha * finalIntervalError[i,j] + temp
                val, temp2 = TwoSum(val, beta)
                finalIntervalError[i,j] += temp2
                finalInterval[i,j] = val
    return finalInterval, finalIntervalError

class TrackedInterval:
    """Tracks the properties of and changes to each interval as it passes through the solver.

    Parameters
    ----------
    interval : numpy array
        The starting interval, shape ``(ndim, 2)`` with each row holding the lower and upper bound
        for one dimension. Stored as both ``topInterval`` (the original) and ``interval`` (the
        current, mutable bounds).

    Attributes
    ----------
    topInterval : numpy array
        The original interval before any changes.
    interval : numpy array
        The current interval (lower bound and upper bound for each dimension in order).
    transforms : list
        List of the alpha and beta values for all the transformations the interval has undergone.
    ndim : int
        The number of dimensions of which the interval consists.
    empty : bool
        Whether the interval is known to contain no roots.
    finalStep : bool
        Whether the interval is in the final step (zooming in on the bounding box to a point at the end).
    canThrowOutFinalStep : bool
        Defaults to False. Whether or not the interval should be thrown out if empty in the final step
        of solving. Changed to True if subdivision occurs in the final step.
    possibleDuplicateRoots : list
        Any multiple roots found through subdivision in the final step that would have been
        returned as just one root before the final step.
    possibleExtraRoot : bool
        Defaults to False. Whether or not the interval would have been thrown out during the final step.
    nextTransformPoints : numpy array
        Where the midpoint of the next subdivision should be for each dimension.
    """
    def __init__(self, interval):
        #Every computation on an interval produces reals, so the stored arrays are float64 whatever
        #the caller handed in. An integer array would silently truncate each new bound to a whole
        #number -- shrinking onto 0.5 would store 0 -- and a float32 one would drop half the digits
        #the solver relies on. asarray keeps the caller's array when it is already float64, so the
        #normal path is unchanged.
        self.topInterval = np.asarray(interval, dtype=np.float64)
        self.interval = np.array(interval, dtype=np.float64)
        self.transforms = []
        self.ndim = len(self.interval)
        self.empty = False
        self.finalStep = False
        self.canThrowOutFinalStep = False
        self.possibleDuplicateRoots = []
        self.possibleExtraRoot = False
        self.nextTransformPoints = np.array([0.0394555475981047]*self.ndim) #Random Point near 0

    def canThrowOut(self):
        """Ensures that an interval that has not subdivided cannot be thrown out on the final step."""
        return not self.finalStep or self.canThrowOutFinalStep

    def addTransform(self, subInterval):
        """Adds the next alpha and beta values to the list transforms and updates the current interval.

        Parameters
        -----------
        subInterval : numpy array
            The subinterval to which the current interval is being reduced
        """
        #Match the stored interval's dtype. Only a caller passing something other than float64 takes
        #a copy here, so the clamping below still writes through to the caller's array on the path
        #the solver itself uses, where the same subInterval is reused across calls.
        if subInterval.dtype != np.float64:
            subInterval = subInterval.astype(np.float64)
        #Ensure the interval has non zero size; mark it empty if it doesn't
        isEmpty = (subInterval[:,0] > subInterval[:,1]).any()
        if isEmpty and self.canThrowOut():
            self.empty = True
            return
        elif isEmpty:
            #If we can't throw the interval out, it should be bounded by [-1,1].
            subInterval[:,0] = np.minimum(subInterval[:,0], np.ones_like(subInterval[:,0]))
            subInterval[:,0] = np.maximum(subInterval[:,0], -np.ones_like(subInterval[:,0]))
            subInterval[:,1] = np.minimum(subInterval[:,1], np.ones_like(subInterval[:,0]))
            subInterval[:,1] = np.maximum(subInterval[:,1], subInterval[:,0])
        # Get the alpha and beta of the transformation and apply it to the current interval.
        self.transforms.append(applySubInterval(self.interval, subInterval))

    def getLastTransform(self):
        """Gets the alpha and beta values of the last transformation the interval underwent."""
        return self.transforms[-1]

    def stackTransforms(self, transforms):
        """Packs a list of (2, ndim) transformations into one (n, 2, ndim) array for applyTransforms."""
        #reshape rather than vstack so that an empty list still comes back with the right shape.
        return np.array(transforms, dtype=float).reshape(-1, 2, self.ndim)

    def getFinalInterval(self):
        """Finds the interval that should be reported as containing a root.

        The final interval is calculated by applying all of the recorded transformations that
        occurred before the final step to topInterval, the original interval.

        Returns
        -------
        finalInterval: numpy array
            The final interval to be reported as containing a root
        """
        transformsToUse = self.transforms if not self.finalStep else self.preFinalTransforms
        finalInterval, finalIntervalError = applyTransforms(self.topInterval.T, self.stackTransforms(transformsToUse))
        finalInterval = finalInterval.T
        finalIntervalError = finalIntervalError.T
        self.finalInterval = finalInterval + finalIntervalError # Add the error and save the result.
        self.finalAlpha, alphaError = TwoSum_NoNumba(-finalInterval[:,0]/2,finalInterval[:,1]/2)
        self.finalAlpha += alphaError + (finalIntervalError[:,1] - finalIntervalError[:,0])/2
        self.finalBeta, betaError = TwoSum_NoNumba(finalInterval[:,0]/2,finalInterval[:,1]/2)
        self.finalBeta += betaError + (finalIntervalError[:,1] + finalIntervalError[:,0])/2
        return self.finalInterval

    def getFinalPoint(self):
        """Finds the point that should be reported as the root (midpoint of the final step interval).

        Returns
        -------
        root: numpy array
            The final point to be reported as the root of the interval
        """
        if not self.finalStep: #If no final step, use the midpoint of the calculated final interval.
            self.root = (self.finalInterval[:,0] + self.finalInterval[:,1]) / 2
        else: #If using the final step, recalculate the final interval using post-final transforms.
            finalInterval, finalIntervalError = applyTransforms(self.topInterval.T, self.stackTransforms(self.transforms))
            finalInterval = finalInterval.T + finalIntervalError.T
            self.root = (finalInterval[:,0] + finalInterval[:,1]) / 2 # Return the midpoint
        return self.root

    def size(self):
        """Gets the volume of the current interval."""
        return np.prod(self.interval[:,1] - self.interval[:,0])

    def dimSize(self):
        """Gets the lengths along each dimension of the current interval."""
        return self.interval[:,1] - self.interval[:,0]

    def finalDimSize(self):
        """Gets the lengths along each dimension of the final interval."""
        return self.finalInterval[:,1] - self.finalInterval[:,0]

    def copy(self):
        """Returns a deep copy of the current interval with all changes and properties preserved."""
        newone = TrackedInterval(self.topInterval)
        newone.interval = self.interval.copy()
        newone.transforms = self.transforms.copy()
        newone.empty = self.empty
        newone.nextTransformPoints = self.nextTransformPoints.copy()
        if self.finalStep:
            newone.finalStep = True
            newone.canThrowOutFinalStep = self.canThrowOutFinalStep
            newone.possibleDuplicateRoots = self.possibleDuplicateRoots.copy()
            newone.possibleExtraRoot = self.possibleExtraRoot
            newone.preFinalInterval = self.preFinalInterval.copy()
            newone.preFinalTransforms = self.preFinalTransforms.copy()
        return newone

    def __contains__(self, point):
        """Determines if point is contained in the current interval."""
        return (point >= self.interval[:,0]).all() and (point <= self.interval[:,1]).all()

    def overlapsWith(self, otherInterval):
        """Determines if the otherInterval overlaps with the current interval.

        Returns True if the lower bound of one interval is less than the upper bound of the other
            in EVERY dimension; returns False otherwise."""
        for (a1,b1),(a2,b2) in zip(self.getIntervalForCombining(), otherInterval.getIntervalForCombining()):
            if a1 > b2 or a2 > b1:
                return False
        return True

    def isPoint(self):
        """Determines if the current interval has essentially length 0 in each dimension."""
        return (np.abs(self.interval[:,0] - self.interval[:,1]) < 1e-32).all()

    def startFinalStep(self):
        """Prepares for the final step by saving the current interval and its transform list."""
        self.finalStep = True
        self.preFinalInterval = self.interval.copy()
        self.preFinalTransforms = self.transforms.copy()

    def getIntervalForCombining(self):
        """Returns the interval to be used in combining intervals to report at the end."""
        return self.preFinalInterval if self.finalStep else self.interval

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.interval)

def absSum(M):
    """Returns the sum of the absolute values of every entry of M.

    Same value as ``np.sum(np.abs(M))`` bit for bit -- both reduce the same temporary with the
    same pairwise summation -- but it skips numpy's reduction dispatch wrapper, which at the
    size of these coefficient tensors costs more than the arithmetic it is dispatching.

    Parameters
    ----------
    M : numpy array
        The array to sum. Any number of dimensions, contiguous or not.

    Returns
    -------
    total : float
        The sum of ``abs(M)`` over every entry.
    """
    return np.abs(M).sum()

#Index tuples of the linear terms of a tensor, keyed by dimension. See getLinearTerms.
_linearTermIndices = {}

def getLinearTerms(M):
    """Gets the linear terms of the Chebyshev coefficient tensor M.

    Uses the fact that the linear terms are located at
    M[(0,0, ... ,0,1)]
    M[(0,0, ... ,1,0)]
    ...
    M[(0,1, ... ,0,0)]
    M[(1,0, ... ,0,0)]

    The index tuples depend only on the number of dimensions, so they are built once per
    dimension and reused. Indexing M directly also avoids M.ravel(), which copies the whole
    tensor whenever M is not contiguous -- which it is not after a transformation of any
    dimension other than the first.

    Parameters
    ----------
    M : numpy array
        The coefficient array to get the linear terms from

    Returns
    -------
    A: numpy array
        An array with the linear terms of M
    """
    #A degree 0 dimension has no linear term.
    return [0 if M.shape[i] == 1 else M[idx] for i, idx in enumerate(getLinearTermIndices(M.ndim))]

def getLinearTermIndices(ndim):
    """Gets the index tuple of the linear term of each dimension of an ndim tensor."""
    idxs = _linearTermIndices.get(ndim)
    if idxs is None:
        idxs = tuple(tuple(1 if j == i else 0 for j in range(ndim)) for i in range(ndim))
        _linearTermIndices[ndim] = idxs
    return idxs


@njit(cache=True)
def linearCheck1(totalErrs, A, consts):
    """Takes A, the linear terms of each function approximation, and makes any possible reduction
        in the interval based on the totalErrs."""
    dim = len(A)
    a = -np.ones(dim) * np.inf
    b = np.ones(dim) * np.inf
    for row in range(dim):
        for col in range(dim):
            if A[row,col] != 0: #Don't bother running the check if the linear term is too small.
                v1 = totalErrs[row] / abs(A[row,col]) - 1
                v2 = 2 * consts[row] / A[row,col]
                if v2 >= 0:
                    a_, b_ = -v1, v1-v2
                else:
                    a_, b_ = -v2-v1, v1
                a[col] = max(a[col], a_)
                b[col] = min(b[col], b_)
    return a, b

@njit(cache=True)
def stackInterval(a, b):
    """Pairs up lower bounds a and upper bounds b into one (dim, 2) interval array."""
    dim = len(a)
    interval = np.empty((dim, 2))
    for i in range(dim):
        interval[i,0] = a[i]
        interval[i,1] = b[i]
    return interval

@njit(cache=True)
def boundingIntervalCore(A, consts, totalErrs, err, errors, finalStep, macheps):
    """The numeric core of BoundingIntervalLinearSystem, compiled.

    Takes the linear system already extracted from the coefficient tensors and does the
    preconditioning, the SVD, and the two shrinking passes. Everything here works on ``dim`` by
    ``dim`` (or length ``dim``) arrays, so in pure numpy the per-call dispatch overhead of the
    couple hundred tiny operations dwarfs the arithmetic; compiling the whole block removes it.
    It is written as explicit loops rather than whole-array expressions, which at these sizes are
    no faster and cost several seconds of compile time.

    A, consts, totalErrs, err and errors are all modified in place; they are scratch copies
    owned by the caller.

    Parameters
    ----------
    A : numpy array
        The (dim, dim) matrix of linear terms, one row per polynomial. Must be C contiguous.
    consts : numpy array
        The constant term of each polynomial.
    totalErrs : numpy array
        Sum of the absolute values of all the coefficients of each polynomial, plus its error.
    err : numpy array
        The part of totalErrs coming from neither the constant nor the linear terms.
    errors : numpy array
        The approximation error of each polynomial.
    finalStep : bool
        Whether we are in the final step of the algorithm.
    macheps : float
        Machine epsilon.

    Returns
    -------
    newInterval : numpy array
        The smaller interval where any root must be, shape ``(dim, 2)``.
    changed : bool
        Whether the interval has shrunk at all.
    should_stop : bool
        Whether we should stop subdividing.
    throwout : bool
        Whether the interval can be discarded entirely.
    """
    dim = A.shape[0]
    #Some constants we use here
    minZoomForChange = 0.99 #If the volume doesn't shrink by this amount say that it hasn't changed
    minZoomForBaseCaseEnd = 0.4**dim #If the volume doesn't change by at least this amount when running with no error, stop

    #Scale all the polynomials relative to one another
    for i in range(dim):
        scaleVal = 0.
        for j in range(dim):
            if abs(A[i,j]) > scaleVal:
                scaleVal = abs(A[i,j])
        if scaleVal > 0:
            s = 2.**int(np.floor(np.log2(scaleVal)))
            for j in range(dim):
                A[i,j] /= s
            consts[i] /= s
            totalErrs[i] /= s
            err[i] /= s
            errors[i] /= s
    #Precondition the columns. (AP)X = B -> A(PX) = B. So scale columns, solve, then scale the solution.
    colScaler = np.ones(dim)
    for i in range(dim):
        scaleVal = 0.
        for j in range(dim):
            if abs(A[j,i]) > scaleVal:
                scaleVal = abs(A[j,i])
        if scaleVal > 0:
            s = 2.**(-np.floor(np.log2(scaleVal)))
            colScaler[i] = s
            for j in range(dim):
                totalErrs[j] += abs(A[j,i]) * (s - 1)
            for j in range(dim):
                A[j,i] *= s

    #Run linear algorithm for shrinking or deciding whether to subdivide.
    #Calculate the SVD outside of the loop below because it doesn't change
    U, S, Vh = np.linalg.svd(A)
    #Test the reciprocal condition number rather than the condition number itself, so that
    #a singular A gives 0 instead of a divide by zero. S[0] == 0 means A is all zeros.
    invCondNum = S[dim-1]/S[0] if S[0] > 0 else 0.
    wellConditioned = S[0] > 0 and invCondNum > 1e-10
    condNum = 1/invCondNum if wellConditioned else 1.
    widthToAdd = max(condNum,2.)*macheps
    Ainv = np.zeros((dim,dim))
    center = np.zeros(dim)
    if wellConditioned:
        #Only invert A when it is safe to do so. Otherwise S has (nearly) zero entries and the
        #inverse is meaningless anyway, so computing it just raises divide by zero warnings.
        Ainv = (Vh.T * (1/S)) @ U.T
        center = -Ainv@consts
    #Use the first interval shrinking method
    a_init, b_init = linearCheck1(totalErrs, A, consts)
    a_orig = a_init
    b_orig = b_init
    #This loop only runs a second time if the interval did not change on the first pass and so
    #needs to run again with tighter errors.
    for i in range(2):
        if wellConditioned: #Make sure conditioning is ok.
            a = np.empty(dim)
            b = np.empty(dim)
            for j in range(dim):
                #Ainv transforms the hyperrectangle of side lengths err into a parallelogram with
                #these as the principal direction, so summing over them gets the farthest the
                #parallelogram can reach in each dimension.
                width = 0.
                for k in range(dim):
                    width += abs(Ainv[j,k]*err[k])
                #Bound with previous result
                low = center[j] - width
                high = center[j] + width
                a[j] = low if low > a_init[j] else a_init[j]
                b[j] = high if high < b_init[j] else b_init[j]
        else:
            #Deliberately not copies. The numpy version bound a to a_init here and then scaled it
            #in place, so a second pass starts from the values the first pass left behind, and the
            #interval it hands back carries them too. Copying here would change what is returned.
            a = a_init
            b = b_init
        #Undo the column preconditioning, then add the error and bound
        for j in range(dim):
            a[j] = a[j]*colScaler[j] - widthToAdd
            b[j] = b[j]*colScaler[j] + widthToAdd
        throwOut = False
        for j in range(dim):
            if a[j] > b[j] or a[j] > 1 or b[j] < -1:
                throwOut = True
        for j in range(dim):
            if a[j] < -1:
                a[j] = -1
            if b[j] < -1:
                b[j] = -1
            if a[j] > 1:
                a[j] = 1
            if b[j] > 1:
                b[j] = 1

        forceShouldStop = finalStep and not wellConditioned
        # Calculate the "changed" variable
        newRatio = 1.
        for j in range(dim):
            newRatio *= b[j] - a[j]
        newRatio /= 2**dim
        if throwOut:
            changed = True
        elif i == 0:
            changed = newRatio < minZoomForChange
        else:
            changed = newRatio < minZoomForBaseCaseEnd

        if i == 0 and changed:
            #First time through and there was a change: return what it shrunk down to, not done.
            return stackInterval(a, b), changed, forceShouldStop, throwOut
        elif i == 0 and not changed:
            #First time through and there was no change: save a and b as the values to return, then
            #run the loop again with a tighter error to see if it shrinks then.
            a_orig = a
            b_orig = b
            err = errors
        elif changed:
            #Second time through and it did change: it didn't change the first time, but the
            #interval did shrink with tighter errors. Return the original interval, not done.
            return stackInterval(a_orig, b_orig), False, forceShouldStop, False
        else:
            #Second time through and it did NOT change: the interval will not shrink even if we
            #subdivide, so return the original interval and stop if the system is well conditioned.
            return stackInterval(a_orig, b_orig), False, wellConditioned or forceShouldStop, False
    #Unreachable: the loop always returns. Here so numba sees a single return type.
    return stackInterval(a_orig, b_orig), False, False, False

def BoundingIntervalLinearSystem(Ms, errors, finalStep, macheps = 2**-52):
    """Finds a smaller region in which any root must be.

    Pulls the linear system out of the coefficient tensors and hands it to
    :func:`boundingIntervalCore`, which does the numeric work.

    Parameters
    ----------
    Ms : list of numpy arrays
        Each numpy array is the coefficient tensor of a chebyshev polynomials
    errors : iterable of floats
        The maximum error of chebyshev approximations
    finalStep : bool
        Whether we are in the final step of the algorithm
    macheps : float
        Machine epsilon used when bounding the linear system. Defaults to ``2**-52``.

    Returns
    -------
    newInterval : numpy array
        The smaller interval where any root must be, shape ``(dim, 2)``.
    changed : bool
        Whether the interval has shrunk at all.
    should_stop : bool
        Whether we should stop subdividing.
    throwout : bool
        Whether the interval can be discarded entirely (no root is possible inside it).
    """
    dim = Ms[0].ndim
    #Get the matrix of the linear terms
    A = np.empty((dim,dim))
    #Get the Vector of the constant terms
    consts = np.empty(dim)
    #Get the Error of everything else combined.
    totalErrs = np.empty(dim)
    zeroIdx = (0,)*dim
    for i, M in enumerate(Ms):
        A[i] = getLinearTerms(M)
        consts[i] = M[zeroIdx]
        totalErrs[i] = absSum(M) + (0. if finalStep else errors[i])
    linear_sums = np.sum(np.abs(A),axis=1)
    err = totalErrs - np.abs(consts) - linear_sums
    #On the final step the approximations are treated as exact.
    errors = np.zeros(dim) if finalStep else np.array(errors, dtype=float)
    return boundingIntervalCore(A, consts, totalErrs, err, errors, finalStep, macheps)

@njit(UniTuple(float64,2)(float64, float64), cache=True)
def TwoSum(a,b):
    """Returns x,y such that a+b=x+y exactly, and a+b=x in floating point using numba."""
    x = a+b
    z = x-a
    y = (a-(x-z)) + (b-z)
    return x,y
def TwoSum_NoNumba(a,b):
    """Returns x,y such that a+b=x+y exactly, and a+b=x in floating point without using numba."""
    x = a+b
    z = x-a
    y = (a-(x-z)) + (b-z)
    return x,y

@njit(UniTuple(float64,2)(float64), cache=True)
def Split(a):
    """Returns x,y such that a = x+y exactly and a = x in floating point using numba."""
    c = (2**27 + 1) * a
    x = c-(c-a)
    y = a-x
    return x,y
def Split_NoNumba(a):
    """Returns x,y such that a = x+y exactly and a = x in floating point without using numba."""
    c = (2**27 + 1) * a
    x = c-(c-a)
    y = a-x
    return x,y

@njit(UniTuple(float64,2)(float64, float64), cache=True)
def TwoProd(a,b):
    """Returns x,y such that a*b=x+y exactly and a*b=x in floating point using numba."""
    x = a*b
    a1,a2 = Split(a)
    b1,b2 = Split(b)
    y=a2*b2-(((x-a1*b1)-a2*b1)-a1*b2)
    return x,y
def TwoProd_NoNumba(a,b):
    """Returns x,y such that a*b=x+y exactly and a*b=x in floating point without using numba."""
    x = a*b
    a1,a2 = Split_NoNumba(a)
    b1,b2 = Split_NoNumba(b)
    y=a2*b2-(((x-a1*b1)-a2*b1)-a1*b2)
    return x,y

@njit(UniTuple(float64,2)(float64, float64, float64, float64), cache=True)
def TwoProdWithSplit(a,b,a1,a2):
    """Returns x,y such that a*b = x+y exactly and a*b = x in floating point but with a already split."""
    x = a*b
    b1,b2 = Split(b)
    y=a2*b2-(((x-a1*b1)-a2*b1)-a1*b2)
    return x,y

def getTransformPoints(newInterval):
    """Gets the alpha and beta points needed to transform the current interval to newInterval."""
    a,b = newInterval
    return (b-a)/2, (b+a)/2

def getTransformationError(M, dim):
    """Returns an upper bound on the error of transforming the Chebyshev approximation M

    In the transformation of dimension dim in M, the matrix multiplication of M by the transformation
    matrix C has each element of M involved in n element multiplications, where n is the number of rows
    in C, which is equal to the degree of approximation of M in dimension dim, or M.shape[dim].

    Parameters
    ----------
    M : numpy array
        The Chebyshev approximation coefficient tensor being transformed
    dim : int
        The dimension of M being transformed

    Returns
    -------
    error : float
        The upper bound for the error associated with the transformation of dimension dim in M
    """
    machEps = 2**-52
    error = M.shape[dim] * machEps * absSum(M)
    return error #TODO: Figure out a more rigurous bound!

def transformCheb(M, alphas, betas, error, exact):
    """Transforms an entire Chebyshev coefficient matrix using the transformation xHat = alpha*x + beta.

    Parameters
    ----------
    M : numpy array
        The chebyshev coefficient matrix
    alphas : iterable
        The scalers in each dimension of the transformation.
    betas : iterable
        The offset in each dimension of the transformation.
    error : float
        A bound on the error of the chebyshev approximation
    exact : bool
        Whether to perform the transformation with higher precision to minimize error

    Returns
    -------
    M : numpy array
        The coefficient matrix transformed to the new interval
    error : float
        An upper bound on the error of the transformation
    """
    #This just does the matrix multiplication on each dimension. Except it's by a tensor.
    for dim,n,alpha,beta in zip(range(M.ndim),M.shape,alphas,betas):
        error += getTransformationError(M, dim)
        M = TransformChebInPlaceND(M,dim,alpha,beta,exact)
    return M, error

def transformChebToInterval(Ms, alphas, betas, errors, exact):
    """Transforms an entire list of Chebyshev approximations to a new interval xHat = alpha*x + beta.

    Parameters
    ----------
    Ms : list of numpy arrays
        The chebyshev coefficient matrices
    alphas : iterable
        The scalers of the transformation we are doing.
    betas : iterable
        The offsets of the transformation we are doing.
    errors : numpy array
        A bound on the error of each Chebyshev approximation
    exact : bool
        Whether to perform the transformation with higher precision to minimize error

    Returns
    -------
    newMs : list of numpy arrays
        The coefficient matrices transformed to the new interval
    newErrors : list of numpy arrays
        The new errors associated with the transformed coefficient matrices
    """
    #Transform the chebyshev polynomials
    newMs = []
    newErrors = []
    for M,e in zip(Ms, errors):
        newM, newE = transformCheb(M, alphas, betas, e, exact)
        newMs.append(newM)
        newErrors.append(newE)
    return newMs, np.array(newErrors)

def zoomInOnIntervalIter(Ms, errors, trackedInterval, exact):
    """One iteration of shrinking an interval that may contain roots.

    Calls BoundingIntervaLinearSystem which determines a smaller interval in which any roots are
    bound to lie. Then calls transformChebToInterval to transform the current coefficient
    approximations to the new interval.

    Parameters
    ----------
    Ms : list of numpy arrays
        The Chebyshev coefficient tensors of each approximation
    errors : numpy array
        An upper bound on the error of each Chebyshev approximation
    trackedInterval : TrackedInterval
        The current interval for which the Chebyshev approximations are valid
    exact : bool
        Whether the transformation should be done with higher precision to minimize error

    Returns
    -------
    Ms : list of numpy arrays
        The chebyshev coefficient matrices transformed to the new interval
    errors : numpy array
        The new errors associated with the transformed coefficient matrices
    trackedInterval : TrackedInterval
        The new interval that the transformed coefficient matrices are valid for
    changed : bool
        Whether or not the interval shrunk significantly during the iteration
    should_stop : bool
        Whether or not to continue subdiviing after the iteration of shrinking is completed
    """

    dim = len(Ms)
    #Zoom in on the current interval
    interval, changed, should_stop, throwOut = BoundingIntervalLinearSystem(Ms, errors, trackedInterval.finalStep)
    #Don't zoom in if we're already at a point
    for dim in range(len(Ms)):
        if trackedInterval.interval[dim,0] == trackedInterval.interval[dim,1]:
            interval[dim, 0] = -1.
            interval[dim, 1] = 1.
    #We can't throw out on the final step
    if throwOut and not trackedInterval.canThrowOut():
        throwOut = False
        should_stop = True
        changed = True
    #Check if we can throw out the whole thing
    if throwOut:
        trackedInterval.empty = True
        return Ms, errors, trackedInterval, True, True
    #Check if we are done iterating
    if not changed:
        return Ms, errors, trackedInterval, changed, should_stop
    #Transform the chebyshev polynomials
    trackedInterval.addTransform(interval)
    Ms, errors = transformChebToInterval(Ms, *trackedInterval.getLastTransform(), errors, exact)
    #We should stop in the final step once the interval has become a point
    if trackedInterval.finalStep and trackedInterval.isPoint():
        should_stop = True
        changed = False

    return Ms, errors, trackedInterval, changed, should_stop

def chebTransform1D(M, alpha, beta, transformDim, exact):
    """Transforms a single dimension of a Chebyshev coefficient matrix.

    Parameters
    ----------
    M : numpy array
        The Chebyshev coefficient matrix
    alpha:
        The scaler of the transformation
    beta:
        The shifting of the transformation
    transformDim:
        The particular dimension of the approximation to be transformed
    exact:
        Whether the transformation should be performed with higher precision to minimize error

    Returns
    -------
    transformed_M : numpy array
        The Chebyshev coefficient matrix transformed to the new interval in dimension transformDim
    """
    return TransformChebInPlaceND(M, transformDim, alpha, beta, exact)

def getInverseOrder(order):
    """Gets a particular order of matrices needed in getSubdivisionIntervals (helper function).

    Takes the order of dimensions in which a Chebyshev coefficient tensor M was subdivided and gets
    the order of the indexes that will arrange the list of resulting transformed matrices as if the
    dimensions had bee subdivided in standard index order. For example, if dimensions 0, 3, 1 were
    subdivided in that order, this function returns the order [0,2,1,3,4,6,5,7] corresponding to the
    indices of currMs such that when arranged in this order, it appears as if the dimensions were
    subdivided in order 0, 1, 3.

    Parameters
    ----------
    order : numpy array
        The order of dimensions along which a coefficient tensor was subdivided

    Returns
    -------
    invOrder : numpy array
        The order of indices of currMs (in the function getSubdivisionIntervals) that arranges the
        matrices resulting from the subdivision as if the original matrix had been subdivided in
        numerical order
    """

    t = np.zeros_like(order)
    t[np.argsort(order)] = np.arange(len(t))
    order = t
    order = 2**(len(order)-1 - order)
    newOrder = np.array([i@order for i in product([0,1],repeat=len(order))])
    invOrder = np.zeros_like(newOrder)
    invOrder[newOrder] = np.arange(len(newOrder))
    return tuple(invOrder)

def hasStalled(trackedInterval, subIntervals):
    """Whether subdividing trackedInterval failed to shrink it.

    Splitting a dimension that is too narrow leaves one half equal to the whole, so trackedInterval
    has stalled exactly when one of the intervals it was split into is identical to it.
    """
    return any(np.array_equal(sub.interval, trackedInterval.interval) for sub in subIntervals)

def stalledIntervalResult(originalInterval, trackedInterval, solverOptions):
    """Returns a stalled interval as a bounding box for a root, or raises if too many have stalled.

    Parameters
    ----------
    originalInterval : TrackedInterval
        The interval solvePolyRecursive was called on, to tell interior from exterior intervals.
    trackedInterval : TrackedInterval
        The interval that no longer shrinks when subdivided.
    solverOptions : SolverOptions
        Supplies the stall counter and maxStalledIntervals.

    Returns
    -------
    interior, exterior : lists of TrackedInterval
        trackedInterval in whichever list it belongs to, as solvePolyRecursive reports a root.

    Raises
    ------
    ValueError
        If more than solverOptions.maxStalledIntervals intervals have stalled during this solve.
    """
    count = solverOptions.stallCounter.increment()
    if count > solverOptions.maxStalledIntervals:
        raise ValueError(f"More than {solverOptions.maxStalledIntervals} intervals could not be "
                         "shrunk any further by subdivision. The solutions do not appear to be finitely "
                         "many isolated roots: the system may have a curve or region of solutions, a "
                         "function that is identically zero, or equations that agree to within "
                         "rounding error.")
    warnings.warn("Subdivision could not shrink an interval any further, so it is returned as a "
                  "bounding box for a root. The root may be multiple or badly conditioned; the "
                  "bounding box, not the reported point, is what the solver can guarantee.")
    if isExteriorInterval(originalInterval, trackedInterval):
        return [], [trackedInterval]
    return [trackedInterval], []

def getSubdivisionDims(Ms,trackedInterval,level):
    """Decides which dimensions to subdivide in and in what order.

    Parameters
    ----------
    Ms : list of numpy arrays
        The chebyshev coefficient matrices
    trackedInterval : trackedInterval
        The interval to be subdivided
    level : int
        The current depth of subdivision from the original interval

    Returns
    -------
    allDims : numpy array
        The ith row gives the dimensions in which Ms[i] should be subdivided, in order.
    """
    dim = len(Ms)
    dims_to_consider = np.arange(dim)
    for i in range(dim):
        if np.isclose(trackedInterval.interval[i,0], trackedInterval.interval[i,1]):
            if len(dims_to_consider) != 1:
                dims_to_consider = np.delete(dims_to_consider, np.argwhere(dims_to_consider==i))
    if level > 5:
        return np.vstack([dims_to_consider[np.argsort(np.array(M.shape)[dims_to_consider])[::-1]] for M in Ms])
    else:
        dim_lengths = trackedInterval.dimSize()
        max_length = np.max([dim_lengths[i] for i in dims_to_consider])
        dims_to_consider = np.extract(dim_lengths[dims_to_consider]>max_length/5,dims_to_consider)
        if len(dims_to_consider) > 1:
            shapes = np.array([np.array(M.shape) for M in Ms])
            degree_sums = np.sum(shapes,axis=0)
            total_sum = np.sum(degree_sums)
            for i in dims_to_consider.copy():
                if len(dims_to_consider) > 1 and degree_sums[i] < np.floor(total_sum/(dim+1)):
                    dims_to_consider = np.delete(dims_to_consider, np.argwhere(dims_to_consider==i))
        return np.vstack([dims_to_consider[np.argsort(np.array(M.shape)[dims_to_consider])[::-1]] for M in Ms])

def getSubdivisionIntervals(Ms, errors, trackedInterval, exact, level):
    """Gets the matrices, error bounds, and intervals for the next iteration of subdivision.

    Parameters
    ----------
    Ms : list of numpy arrays
        The chebyshev coefficient matrices
    errors : numpy array
        An upper bound on the error of each Chebyshev approximation
    trackedInterval : trackedInterval
        The interval to be subdivided
    exact : bool
        Whether transformations should be completed with higher precision to minimize error
    level : int
        The current depth of subdivision from the original interval

    Returns
    -------
    allMs : list of numpy arrays
        The transformed coefficient matrices associated with each new interval
    allErrors : numpy array
        A list of upper bounds for the errors associated with each transformed coefficient matrix
    allIntervals : list of TrackedIntervals
        The intervals from the subdivision (corresponding one to one with the matrices in allMs)
    """
    subdivisionDims = getSubdivisionDims(Ms,trackedInterval,level)
    dimSet = set(subdivisionDims.flatten())
    if len(dimSet) != subdivisionDims.shape[1]:
        raise ValueError("Subdivision Dimensions are invalid! Each Polynomial must subdivide in the same dimensions!")
    allMs = []
    allErrors = []
    idx = 0
    for M,error,order in zip(Ms, errors, subdivisionDims):
        idx += 1
        #Iterate through the dimensions, highest degree first.
        currMs, currErrs = [M],[error]
        for thisDim in order:
            newMidpoint = trackedInterval.nextTransformPoints[thisDim]
            alpha, beta = (newMidpoint+1)/2, (newMidpoint-1)/2
            tempMs = []
            tempErrs = []
            for T,E in zip(currMs, currErrs):
                #Transform the polys
                P1, P2 = chebTransform1D(T, alpha, beta, thisDim, exact), chebTransform1D(T, -beta, alpha, thisDim, exact)
                E1 = getTransformationError(T, thisDim)
                tempMs += [P1, P2]
                tempErrs += [E1 + E, E1 + E]
            currMs = tempMs
            currErrs = tempErrs
        if M.ndim == 1:
            allMs.append(currMs) #Already ordered because there's only 1.
            allErrors.append(currErrs) #Already ordered because there's only 1.
        else:
            #Order the polynomials so they match the intervals in subdivideInterval
            invOrder = getInverseOrder(order)
            allMs.append([currMs[i] for i in invOrder])
            allErrors.append([currErrs[i] for i in invOrder])
    allMs = [[allMs[i][j] for i in range(len(allMs))] for j in range(len(allMs[0]))]
    allErrors = [[allErrors[i][j] for i in range(len(allErrors))] for j in range(len(allErrors[0]))]
    #Get the intervals
    allIntervals = [trackedInterval]
    for thisDim in dimSet:
        newMidpoint = trackedInterval.nextTransformPoints[thisDim]
        newSubinterval = np.ones_like(trackedInterval.interval) #TODO: Make this outside for loop
        newSubinterval[:,0] = -1.
        newIntervals = []
        for oldInterval in allIntervals:
            newInterval1 = oldInterval.copy()
            newInterval2 = oldInterval.copy()
            newSubinterval[thisDim] = [-1., newMidpoint]
            newInterval1.addTransform(newSubinterval)
            newSubinterval[thisDim] = [newMidpoint, 1.]
            newInterval2.addTransform(newSubinterval)
            newInterval1.nextTransformPoints[thisDim] = 0
            newInterval2.nextTransformPoints[thisDim] = 0
            newIntervals.append(newInterval1)
            newIntervals.append(newInterval2)
        allIntervals = newIntervals
    return allMs, allErrors, allIntervals

#Slices used by trimMs, keyed by (ndim, dim). See getTrimSlices.
_trimSlices = {}

def getTrimSlices(ndim, dim):
    """Gets the index tuples that select, and that drop, the highest degree row of a dimension.

    Parameters
    ----------
    ndim : int
        The number of dimensions of the tensor being trimmed.
    dim : int
        The dimension whose last row is in question.

    Returns
    -------
    lastRow : tuple
        Indexes the highest degree row of dimension dim.
    dropLastRow : tuple
        Indexes everything but that row.
    """
    slices = _trimSlices.get((ndim, dim))
    if slices is None:
        lastRow = tuple(-1 if i == dim else slice(None) for i in range(ndim))
        dropLastRow = tuple(slice(None,-1) if i == dim else slice(None) for i in range(ndim))
        slices = (lastRow, dropLastRow)
        _trimSlices[(ndim, dim)] = slices
    return slices

def trimMs(Ms, errors, relApproxTol=1e-3, absApproxTol=0):
    """Reduces the degree of each chebyshev approximation M when doing so has negligible error.

    The coefficient matrices are trimmed in place. This function iteratively looks at the highest
    degree coefficient row of each M along each dimension and trims it as long as the error introduced
    is less than the allowed error increase for that dimension.

    Parameters
    ----------
    Ms : list of numpy arrays
        The chebyshev approximations of the functions
    errors : numpy array
        The max error of the chebyshev approximation from the function on the interval
    relApproxTol : double
        The relative error increase allowed
    absApproxTol : double
        The absolute error increase allowed
    """
    dim = Ms[0].ndim
    for polyNum in range(len(Ms)): #Loop through the polynomials
        allowedErrorIncrease = absApproxTol + errors[polyNum] * relApproxTol
        M = Ms[polyNum]
        for currDim in range(dim):
            #Slicing to look at a slice of the highest degree in the dimension we want to trim,
            #and to drop it. Both depend only on the dimension, so they are built once and reused.
            lastRow, dropLastRow = getTrimSlices(dim, currDim)
            lastSum = absSum(M[lastRow])

            # Iteratively eliminate the highest degree row of the current dimension if
            # the sum of its approximation coefficients is of low error, but keep deg at least 2
            while lastSum < allowedErrorIncrease and M.shape[currDim] > 3:
                # Trim the polynomial
                M = M[dropLastRow]
                # Update the remaining error increase allowed an the error of the approximation.
                allowedErrorIncrease -= lastSum
                errors[polyNum] += lastSum
                # Reset for the next iteration with the next highest degree of the current dimension.
                lastSum = absSum(M[lastRow])
        Ms[polyNum] = M

def isExteriorInterval(originalInterval, trackedInterval):
    """Determines if the current interval is exterior to its original interval."""
    return (trackedInterval.getIntervalForCombining() == originalInterval.getIntervalForCombining()).any()

def getRootsInInterval(interval):
        """Gets the roots that a final bounding interval reports."""
        if len(interval.possibleDuplicateRoots) > 0:
            return list(interval.possibleDuplicateRoots)
        return [interval.getFinalPoint()]

def make_child_tasks(allMs, allErrors, allIntervals, parent_id=None, level=0):
    """Bundle subdivided children into :class:`SolveTask` records for the parallel driver.

    Parameters
    ----------
    allMs : iterable
        One coefficient-tensor list per child interval.
    allErrors : iterable
        Per-poly error bounds, aligned with ``allMs``.
    allIntervals : iterable of TrackedInterval
        The child intervals produced by subdivision.
    parent_id : int or None
        Identifier of the parent interval whose results these children feed into. ``None`` for
        top-level tasks.
    level : int
        Subdivision depth assigned to each child task.

    Returns
    -------
    list of SolveTask
        One task per child interval, ready to be queued.
    """
    return [
        SolveTask(newMs, newInt, newErrs, parent_id=parent_id, level=level)
        for newMs, newErrs, newInt in zip(allMs, allErrors, allIntervals)
    ]

def solvePolySequential(Ms, trackedInterval, errors, solverOptions):
    """
    Fully sequential solve.

    Use this inside workers when you do not want nested parallelism.
    """
    localOptions = solverOptions.copy()
    localOptions.allowParallel = False
    return solvePolyRecursive(
        Ms,
        trackedInterval,
        errors,
        localOptions,
        returnChildren=False
    )

def _solve_one_level_worker(task, solverOptions):
    """
    Worker for one unit of multilevel work.

    It solves one interval until either:
      1. it finishes, or
      2. it reaches subdivision and returns child tasks.
    """
    localOptions = solverOptions.copy()
    localOptions.allowParallel = False
    localOptions.level = task.level

    return solvePolyRecursive(
        task.Ms,
        task.trackedInterval,
        task.errors,
        localOptions,
        returnChildren=True
    )

def finish_subdivision_state(state, childInterior, childExterior):
    """
    Finish a parent interval after its children have completed.

    This contains the logic that used to happen immediately after the
    recursive child calls returned.
    """
    originalMs = state.originalMs
    originalInterval = state.originalInterval
    trackedInterval = state.trackedInterval
    errors = state.errors
    solverOptions = state.solverOptions

    resultInterior = list(childInterior)
    resultExterior = list(childExterior)

    if state.isFinalStep:
        resultsAll = resultInterior + resultExterior

        if len(resultsAll) == 0:
            trackedInterval.possibleExtraRoot = True

            if isExteriorInterval(originalInterval, trackedInterval):
                return [], [trackedInterval]
            else:
                return [trackedInterval], []

        # Combine all roots that converged to the same point.
        allFoundRoots = set()
        tempResults = []
        for result in resultsAll:
            point = tuple(result.interval[:,0])
            if point in allFoundRoots:
                continue
            allFoundRoots.add(point)
            tempResults.append(result)

        for result in tempResults:
            if len(result.possibleDuplicateRoots) > 0:
                trackedInterval.possibleDuplicateRoots += result.possibleDuplicateRoots
            else:
                trackedInterval.possibleDuplicateRoots.append(result.getFinalPoint())

        if isExteriorInterval(originalInterval, trackedInterval):
            return [], [trackedInterval]
        else:
            return [trackedInterval], []
    
    idx1 = 0
    idx2 = 1

    for tempInterval in resultExterior:
        tempInterval.reRun = False

    while idx1 < len(resultExterior):
        while idx2 < len(resultExterior):
            if resultExterior[idx1].overlapsWith(resultExterior[idx2]):
                combinedInterval = originalInterval.copy()

                if combinedInterval.finalStep:
                    combinedInterval.interval = combinedInterval.preFinalInterval.copy()
                    combinedInterval.transforms = combinedInterval.preFinalTransforms.copy()

                newAs = np.min(
                    [
                        resultExterior[idx1].getIntervalForCombining()[:, 0],
                        resultExterior[idx2].getIntervalForCombining()[:, 0]
                    ],
                    axis=0
                )

                newBs = np.max(
                    [
                        resultExterior[idx1].getIntervalForCombining()[:, 1],
                        resultExterior[idx2].getIntervalForCombining()[:, 1]
                    ],
                    axis=0
                )

                final1 = resultExterior[idx1].getFinalInterval()
                final2 = resultExterior[idx2].getFinalInterval()

                newAsFinal = np.min([final1[:, 0], final2[:, 0]], axis=0)
                newBsFinal = np.max([final1[:, 1], final2[:, 1]], axis=0)

                oldAs = originalInterval.interval[:, 0]
                oldBs = originalInterval.interval[:, 1]
                oldAsFinal, oldBsFinal = originalInterval.getFinalInterval().T

                equalMask = oldBsFinal == oldAsFinal
                oldBsFinal[equalMask] = oldBsFinal[equalMask] + 1

                currSubinterval = (
                    (
                        2 * np.array([newAsFinal, newBsFinal])
                        - oldAsFinal
                        - oldBsFinal
                    )
                    / (oldBsFinal - oldAsFinal)
                ).T

                currSubinterval[equalMask, 0] = -1
                currSubinterval[equalMask, 1] = 1

                currSubinterval[:, 0][oldAs == newAs] = -1
                currSubinterval[:, 1][oldBs == newBs] = 1

                combinedInterval.addTransform(currSubinterval)
                combinedInterval.interval = np.array([newAs, newBs]).T
                combinedInterval.reRun = True

                del resultExterior[idx2]
                del resultExterior[idx1]

                resultExterior.append(combinedInterval)
                idx2 = idx1 + 1
            else:
                idx2 += 1

        idx1 += 1
        idx2 = idx1 + 1

    # Rerun touching intervals.
    newResultExterior = []

    for tempInterval in resultExterior:
        if tempInterval.reRun:
            if (tempInterval.interval == originalInterval.interval).all():
                newResultExterior.append(tempInterval)
            else:
                tempMs, tempErrors = transformChebToInterval(
                    originalMs,
                    *tempInterval.getLastTransform(),
                    errors,
                    solverOptions.exact
                )

                tempResultsInterior, tempResultsExterior = solvePolySequential(
                    tempMs,
                    tempInterval,
                    tempErrors,
                    solverOptions
                )

                resultInterior += tempResultsInterior
                newResultExterior += tempResultsExterior

        elif isExteriorInterval(originalInterval, tempInterval):
            newResultExterior.append(tempInterval)

        else:
            resultInterior.append(tempInterval)

    return resultInterior, newResultExterior


def solvePolyParallelMultilevel(Ms, trackedInterval, errors, solverOptions):
    """Multilevel parallel driver for the subdivision solver.
    Submits :class:`SolveTask` units to a :class:`ThreadPoolExecutor` sized by
    ``solverOptions.max_cpu``. Each worker solves one task until it either finishes or
    subdivides; subdivided tasks come back with child tasks that are queued and joined to their
    parent via :func:`finish_subdivision_state`. This is the only place where a process pool is
    created — workers themselves run with ``allowParallel`` disabled to prevent nested pools.

    Parameters
    ----------
    Ms : list of numpy arrays
        Chebyshev coefficient tensors for the top-level interval.
    trackedInterval : TrackedInterval
        The interval to solve over.
    errors : numpy array
        Per-poly approximation error bounds.
    solverOptions : SolverOptions
        Options for the solve. ``max_cpu`` and ``parallel_depth`` are read here.

    Returns
    -------
    finalInterior : list of TrackedInterval
        Intervals strictly inside the original domain that contain a root.
    finalExterior : list of TrackedInterval
        Intervals on the boundary of the original domain that contain a root.
    """
    max_workers = max(1, solverOptions.max_cpu)

    workerOptions = solverOptions.copy()
    workerOptions.allowParallel = False

    next_parent_id = 0

    pendingTasks = [SolveTask(Ms, trackedInterval, errors, parent_id=None)]

    futures = set()

    # parent_id -> bookkeeping
    waitingParents = {}

    finalInterior = []
    finalExterior = []

    def submit_task(executor, task):
        return executor.submit(_solve_one_level_worker, task, workerOptions), task.parent_id

    def complete_result(result, parent_id):
        """
        Handle a completed TaskResult.

        If parent_id is None, add directly to final result.
        Otherwise, accumulate into the waiting parent.
        """
        nonlocal next_parent_id

        # Case 1: the task finished normally.
        if len(result.childTasks) == 0:
            if parent_id is None:
                finalInterior.extend(result.interior)
                finalExterior.extend(result.exterior)
            else:
                parent = waitingParents[parent_id]
                parent["interior"].extend(result.interior)
                parent["exterior"].extend(result.exterior)
                parent["remaining"] -= 1

            return

        # Case 2: the task subdivided.
        this_parent_id = next_parent_id
        next_parent_id += 1

        waitingParents[this_parent_id] = {
            "state": result.subdivisionState,
            "parent_id": parent_id,
            "remaining": len(result.childTasks),
            "interior": list(result.interior),
            "exterior": list(result.exterior),
        }

        for child in result.childTasks:
            child.parent_id = this_parent_id
            pendingTasks.append(child)

    def finish_ready_parents():
        """
        Some parent may become ready after its final child finishes.

        Finishing a parent produces normal interior/exterior results,
        which then need to be passed upward to that parent's parent.
        """
        changed = True

        while changed:
            changed = False

            ready_ids = [
                parent_id
                for parent_id, parent in waitingParents.items()
                if parent["remaining"] == 0
            ]

            for parent_id in ready_ids:
                parent = waitingParents.pop(parent_id)

                interior, exterior = finish_subdivision_state(
                    parent["state"],
                    parent["interior"],
                    parent["exterior"]
                )

                parent_result = TaskResult(
                    interior=interior,
                    exterior=exterior,
                    childTasks=[],
                    subdivisionState=None
                )

                complete_result(parent_result, parent["parent_id"])
                changed = True

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_parent = {}

        # Fill pool initially.
        while pendingTasks and len(futures) < max_workers:
            task = pendingTasks.pop()
            fut, parent_id = submit_task(executor, task)
            futures.add(fut)
            future_to_parent[fut] = parent_id

        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)

            for fut in done:
                parent_id = future_to_parent.pop(fut)
                result = fut.result()

                complete_result(result, parent_id)
                finish_ready_parents()

            # Refill available worker slots.
            while pendingTasks and len(futures) < max_workers:
                task = pendingTasks.pop()
                fut, parent_id = submit_task(executor, task)
                futures.add(fut)
                future_to_parent[fut] = parent_id

    # After all futures finish, make sure all parent continuations are finished.
    finish_ready_parents()

    return finalInterior, finalExterior


def solvePolyRecursive(Ms, trackedInterval, errors, solverOptions, returnChildren=False):
    """
    Recursively shrinks and subdivides the given interval to find the locations of all roots.

    When returnChildren=False:
        behaves like the original sequential recursive function.

    When returnChildren=True:
        solves until it reaches a subdivision point, then returns a TaskResult
        containing child tasks instead of recursively solving those children.
    """

    if trackedInterval.isPoint():
        if returnChildren:
            return TaskResult([], [trackedInterval], [])
        return [], [trackedInterval]

    solverOptions = solverOptions.copy()
    solverOptions.level += 1

    # Constant term check.
    if solverOptions.constant_check:
        zeroIdx = (0,)*Ms[0].ndim
        #Bail out on the first polynomial whose constant term dominates, so the rest need not be summed.
        for M, e in zip(Ms, errors):
            c = M[zeroIdx]
            if abs(c) > absSum(M) - abs(c) + e:
                if returnChildren:
                    return TaskResult([], [], [])
                return [], []

    # Quadratic check.
    if (solverOptions.low_dim_quadratic_check and Ms[0].ndim <= 3) or solverOptions.all_dim_quadratic_check:
        for i in range(len(Ms)):
            if quadratic_check(Ms[i], errors[i]):
                if returnChildren:
                    return TaskResult([], [], [])
                return [], []

    # Trim.
    Ms = Ms.copy()
    originalMs = Ms.copy()
    trackedInterval = trackedInterval.copy()
    errors = errors.copy()

    trimMs(Ms, errors)

    changed = True
    zoomCount = 0

    originalInterval = trackedInterval.copy()

    lastSizes = trackedInterval.dimSize()

    while changed and zoomCount <= solverOptions.maxZoomCount:
        Ms, errors, trackedInterval, changed, should_stop = zoomInOnIntervalIter(Ms,errors,trackedInterval,solverOptions.exact)

        if trackedInterval.empty:
            if returnChildren:
                return TaskResult([], [], [])
            return [], []

        newSizes = trackedInterval.dimSize()

        if (newSizes >= lastSizes / 2).all(): #Check all dims and use >= to account for a dimension being 0.
            zoomCount += 1

        lastSizes = newSizes

    if should_stop:
        if trackedInterval.finalStep or not solverOptions.useFinalStep:
            if solverOptions.verbose:
                print("*", end="")

            if isExteriorInterval(originalInterval, trackedInterval):
                if returnChildren:
                    return TaskResult([], [trackedInterval], [])
                return [], [trackedInterval]
            else:
                if returnChildren:
                    return TaskResult([trackedInterval], [], [])
                return [trackedInterval], []

        else:
            trackedInterval.startFinalStep()

            if returnChildren and solverOptions.level <= solverOptions.parallel_depth:
                # Continue solving this same interval in the global scheduler.
                child = SolveTask(Ms, trackedInterval, errors, level=solverOptions.level)
                state = SubdivisionState(
                    originalMs=originalMs,
                    originalInterval=originalInterval,
                    trackedInterval=trackedInterval,
                    errors=errors,
                    solverOptions=solverOptions,
                    isFinalStep=False
                )

                return TaskResult(interior=[], exterior=[], childTasks=[child], subdivisionState=state)

            serialInterior, serialExterior = solvePolyRecursive(Ms, trackedInterval, errors, solverOptions, returnChildren=False)

            if returnChildren:
                return TaskResult(interior=serialInterior, exterior=serialExterior, childTasks=[], subdivisionState=None)
            
            return serialInterior, serialExterior

    elif trackedInterval.finalStep:
        trackedInterval.canThrowOutFinalStep = True

        resultInterior, resultExterior = [], []

        allMs, allErrors, allIntervals = getSubdivisionIntervals(
            Ms,
            errors,
            trackedInterval,
            solverOptions.exact,
            solverOptions.level
        )

        if hasStalled(trackedInterval, allIntervals):
            stalledInterior, stalledExterior = stalledIntervalResult(originalInterval, trackedInterval, solverOptions)
            if returnChildren:
                return TaskResult(stalledInterior, stalledExterior, [])
            return stalledInterior, stalledExterior

        state = SubdivisionState(
            originalMs=originalMs,
            originalInterval=originalInterval,
            trackedInterval=trackedInterval,
            errors=errors,
            solverOptions=solverOptions,
            isFinalStep=True
        )

        childTasks = make_child_tasks(
            allMs, allErrors, allIntervals, level=solverOptions.level
        )

        if returnChildren and solverOptions.level <= solverOptions.parallel_depth:
            return TaskResult(
                interior=resultInterior,
                exterior=resultExterior,
                childTasks=childTasks,
                subdivisionState=state
            )

        # Solve children serially in this worker/process.
        for child in childTasks:
            newInterior, newExterior = solvePolyRecursive(
                child.Ms,
                child.trackedInterval,
                child.errors,
                solverOptions,
                returnChildren=False
            )

            resultInterior += newInterior
            resultExterior += newExterior

        resultInterior, resultExterior = finish_subdivision_state(
            state, resultInterior, resultExterior
        )

        if returnChildren:
            return TaskResult(
                interior=resultInterior,
                exterior=resultExterior,
                childTasks=[],
                subdivisionState=None
            )
        
        return resultInterior, resultExterior

    else:
        # Normal subdivision.
        if solverOptions.level == 15:
            warnings.warn(
                "High subdivision depth!\n"
                "Subdivision on the search interval has now reached "
                "at least depth 15. Runtime may be prolonged."
            )

        elif solverOptions.level == 25:
            warnings.warn(
                "Extreme subdivision depth!\n"
                "Subdivision on the search interval has now reached "
                "at least depth 25, which is unusual. The solver may not finish running. "
                "Ensure the input functions meet the requirements of being continuous, "
                "smooth, and having only finitely many simple roots on the search interval."
            )

        resultInterior, resultExterior = [], []

        allMs, allErrors, allIntervals = getSubdivisionIntervals(
            Ms,
            errors,
            trackedInterval,
            solverOptions.exact,
            solverOptions.level
        )

        if hasStalled(trackedInterval, allIntervals):
            stalledInterior, stalledExterior = stalledIntervalResult(originalInterval, trackedInterval, solverOptions)
            if returnChildren:
                return TaskResult(stalledInterior, stalledExterior, [])
            return stalledInterior, stalledExterior

        state = SubdivisionState(
            originalMs=originalMs,
            originalInterval=originalInterval,
            trackedInterval=trackedInterval,
            errors=errors,
            solverOptions=solverOptions,
            isFinalStep=False
        )

        childTasks = make_child_tasks(
            allMs, allErrors, allIntervals, level=solverOptions.level
        )

        if returnChildren and solverOptions.level <= solverOptions.parallel_depth:
            return TaskResult(
                interior=resultInterior,
                exterior=resultExterior,
                childTasks=childTasks,
                subdivisionState=state
            )

        # Solve children serially in this worker/process.
        for child in childTasks:
            newInterior, newExterior = solvePolyRecursive(
                child.Ms,
                child.trackedInterval,
                child.errors,
                solverOptions,
                returnChildren=False
            )

            resultInterior += newInterior
            resultExterior += newExterior

        resultInterior, resultExterior = finish_subdivision_state(
            state, resultInterior, resultExterior
        )

        if returnChildren:
            return TaskResult(
                interior=resultInterior,
                exterior=resultExterior,
                childTasks=[],
                subdivisionState=None
            )
        return resultInterior, resultExterior


def solvePoly(Ms, trackedInterval, errors, solverOptions):
    """Dispatches to the parallel or recursive solver based on ``solverOptions``.

    Public entry point — call this instead of :func:`solvePolyRecursive` directly.
    Routes to :func:`solvePolyParallelMultilevel` when both ``parallel_depth > 0`` and
    ``max_cpu > 1``; otherwise falls back to serial :func:`solvePolyRecursive`.
    """
    if solverOptions.parallel_depth > 0 and solverOptions.max_cpu > 1:
        return solvePolyParallelMultilevel(
            Ms,
            trackedInterval,
            errors,
            solverOptions
        )

    return solvePolyRecursive(
        Ms,
        trackedInterval,
        errors,
        solverOptions,
        returnChildren=False
    )

def solveChebyshevSubdivision(Ms, errors, verbose = False, exact = False, constant_check = True, low_dim_quadratic_check = True,
                              all_dim_quadratic_check = False, max_cpu=1, parallel_depth=0):
    """Initiates shrinking and subdivision recursion and returns the roots and bounding boxes.

    Parameters
    ----------
    Ms : list of numpy arrays
        The chebyshev approximations of the functions on the interval given to CombinedSolver
    errors : numpy array
        The max error of the chebyshev approximation from the function on the interval
    verbose : bool
        Defaults to False. Whether or not to output progress of solving to the terminal.
    exact : bool
        Defaults to False. Whether transformations should be done with higher precision to minimize error.
    constant_check : bool
        Defaults to True. Whether or not to run constant term check after each subdivision.
    low_dim_quadratic_check : bool
        Defaults to True. Whether or not to run quadratic check in dim 2, 3.
    all_dim_quadratic_check : bool
        Defaults to False. Whether or not to run quadratic check in dim >= 4.
    max_cpu : int
        Defaults to 1. Maximum number of CPUs to use when dispatching subdivided regions to the
        multilevel parallel driver. CPU is not reserved for the main thread, so the worker pool
        is sized at ``max_cpu``, not ``max_cpu - 1``.
    parallel_depth : int
        Defaults to 0. Subdivision depth at which child tasks start being pushed to the worker
        pool. Higher values keep work serial for longer before parallelizing.

    Returns
    -------
    boundingIntervals : list of TrackedInterval
        A finalized bounding interval for each root found on the interval given to Combined Solver.
        The roots themselves are not returned; call :func:`getRootsInInterval` on an interval to get
        the root or roots it reports.
    """
    #Assert that we have n nD polys
    if any(M.ndim != len(Ms) for M in Ms):
        raise ValueError("Solver Takes in N polynomials of dimension N!")
    if len(Ms) != len(errors):
        raise ValueError("Ms and errors must be same length!")

    #Solve
    originalInterval = TrackedInterval(np.array([[-1.,1.]]*Ms[0].ndim))
    solverOptions = SolverOptions()
    solverOptions.verbose = verbose
    solverOptions.exact = exact
    solverOptions.constant_check = constant_check
    solverOptions.low_dim_quadratic_check = low_dim_quadratic_check
    solverOptions.all_dim_quadratic_check = all_dim_quadratic_check
    solverOptions.useFinalStep = True
    solverOptions.max_cpu=max_cpu
    solverOptions.parallel_depth=parallel_depth

    if verbose:
        print("Finding roots...", end=' ')
    b1, b2 = solvePoly(Ms, originalInterval, errors, solverOptions)

    boundingIntervals = b1 + b2
    hasDupRoots = False
    hasExtraRoots = False

    for interval in boundingIntervals:
        interval.getFinalInterval()
        if interval.possibleExtraRoot:
            hasExtraRoots = True
        if len(interval.possibleDuplicateRoots) > 0:
            hasDupRoots = True

    #Warn if extra or duplicate roots
    if hasExtraRoots:
        warnings.warn(f"Might Have Extra Roots! See Bounding Boxes for details!")
    if hasDupRoots:
        warnings.warn(f"Might Have Duplicate Roots! See Bounding Boxes for details!")
    return boundingIntervals
