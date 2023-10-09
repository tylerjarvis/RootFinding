import numpy as np
from numba import njit, float64
from numba.types import UniTuple
from itertools import product
from scipy.spatial import HalfspaceIntersection, QhullError
from scipy.optimize import linprog
from yroots.QuadraticCheck import quadratic_check
from time import time
import copy
import warnings

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
        Maximum number of zooms allowed before subdividing (prevents infinite infintesimal shrinking)
    level : int
        Depth of subdivision for the given interval.
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
    
    def copy(self):
        return copy.copy(self) #Return shallow copy, everything should be a basic type

@njit
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

@njit
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
        #finalVal = alpha*arr2[i]
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

@njit
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
        #finalVal = alpha*arr2[i]
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
        # Move the current dimension to the dim 0 spot in the np array.
        order = np.array([dim] + [i for i in range(dim)] + [i for i in range(dim+1, coeffs.ndim)])
        # Then transpose with the inverted order after the transformation occurs.
        backOrder = np.zeros(coeffs.ndim, dtype = int)
        backOrder[order] = np.arange(coeffs.ndim)
        return TransformFunc(coeffs.transpose(order), alpha, beta).transpose(backOrder)

class TrackedInterval:
    """Tracks the properties of and changes to each interval as it passes through the solver.

    Parameters
    ----------
    topInterval: numpy array
        The original interval before any changes
    interval: numpy array
        The current interval (lower bound and upper bound for each dimension in order)
    transforms: list
        List of the alpha and beta values for all the transformations the interval has undergone
    ndim: int
        The number of dimensions of which the interval consists
    empty: bool
        Whether the interval is known to contain no roots
    finalStep: bool
        Whether the interval is in the final step (zooming in on the bounding box to a point at the end)
    canThrowOutFinalStep: bool
        Defaults to False. Whether or not the interval should be thrown out if empty in the final step
        of solving. Changed to True if subdivision occurs in the final step.
    possibleDuplicateRoots: list
        Any multiple roots found through subdivision in the final step that would have been
        returned as just one root before the final step
    possibleExtraRoot: bool
        Defaults to False. Whether or not the interval would have been thrown out during the final step.
    nextTransformPoints: numpy array
        Where the midpoint of the next subdivision should be for each dimension
    """
    def __init__(self, interval):
        self.topInterval = interval
        self.interval = interval
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

        Parameters:
        -----------
        subInterval : numpy array
            The subinterval to which the current interval is being reduced
        """
        #Ensure the interval has non zero size; mark it empty if it doesn't
        if np.any(subInterval[:,0] > subInterval[:,1]) and self.canThrowOut():
            self.empty = True
            return
        elif np.any(subInterval[:,0] > subInterval[:,1]):
            #If we can't throw the interval out, it should be bounded by [-1,1].
            subInterval[:,0] = np.minimum(subInterval[:,0], np.ones_like(subInterval[:,0]))
            subInterval[:,0] = np.maximum(subInterval[:,0], -np.ones_like(subInterval[:,0]))
            subInterval[:,1] = np.minimum(subInterval[:,1], np.ones_like(subInterval[:,0]))
            subInterval[:,1] = np.maximum(subInterval[:,1], subInterval[:,0])
        # Get the alpha and beta associated with the transformation in each dimension 
        a1,b1 = subInterval.T # all the lower bounds and upper bounds of the new interval, respectively
        a2,b2 = self.interval.T # all the lower bounds and upper bounds of the original interval
        alpha1, beta1 = (b1-a1)/2, (b1+a1)/2
        alpha2, beta2 = (b2-a2)/2, (b2+a2)/2
        self.transforms.append(np.array([alpha1, beta1]))
        #Update the lower and upper bounds of the current interval
        for dim in range(self.ndim):
            for i in range(2):
                x = subInterval[dim][i]
                #Be exact if x = +-1
                if x == -1.0:
                    self.interval[dim][i] = self.interval[dim][0]
                elif x == 1.0:
                    self.interval[dim][i] = self.interval[dim][1]
                else:
                    self.interval[dim][i] = alpha2[dim]*x+beta2[dim]
    
    def getLastTransform(self):
        """Gets the alpha and beta values of the last transformation the interval underwent."""
        return self.transforms[-1]
        
    def getFinalInterval(self):
        """Finds the interval that should be reported as containing a root.

        The final interval is calculated by applying all of the recorded transformations that
        occurred before the final step to topInterval, the original interval.

        Returns
        -------
        finalInterval: numpy array
            The final interval to be reported as containing a root
        """
        # TODO: Make this a seperate function so it can use njit.
        # Make these _NoNumba calls use floats so they call call the numba functions without a seperate compile
        finalInterval = self.topInterval.T
        finalIntervalError = np.zeros_like(finalInterval)
        transformsToUse = self.transforms if not self.finalStep else self.preFinalTransforms
        for alpha,beta in transformsToUse[::-1]: # Iteratively apply each saved transform
            finalInterval, temp = TwoProd_NoNumba(finalInterval, alpha)
            finalIntervalError = alpha * finalIntervalError + temp
            finalInterval, temp = TwoSum_NoNumba(finalInterval,beta)
            finalIntervalError += temp
            
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
        #TODO: Make this a seperate function so it can use njit.
        #Make these _NoNumba calls use floats so they call call the numba functions without a seperate compile
        if not self.finalStep: #If no final step, use the midpoint of the calculated final interval.
            self.root = (self.finalInterval[:,0] + self.finalInterval[:,1]) / 2
        else: #If using the final step, recalculate the final interval using post-final transforms.
            finalInterval = self.topInterval.T
            finalIntervalError = np.zeros_like(finalInterval)
            transformsToUse = self.transforms
            for alpha,beta in transformsToUse[::-1]:
                finalInterval, temp = TwoProd_NoNumba(finalInterval, alpha)
                finalIntervalError = alpha * finalIntervalError + temp
                finalInterval, temp = TwoSum_NoNumba(finalInterval,beta)
                finalIntervalError += temp
            finalInterval = finalInterval.T + finalIntervalError.T
            self.root = (finalInterval[:,0] + finalInterval[:,1]) / 2 # Return the midpoint
        return self.root
    
    def size(self):
        """Gets the volume of the current interval."""
        return np.product(self.interval[:,1] - self.interval[:,0])
    
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
        return np.all(point >= self.interval[:,0]) and np.all(point <= self.interval[:,1])

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
        return np.all(np.abs(self.interval[:,0] - self.interval[:,1]) < 1e-32)
        
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

def getLinearTerms(M):
    """Gets the linear terms of the Chebyshev coefficient tensor M.

    Uses the fact that the linear terms are located at 
    M[(0,0, ... ,0,1)]
    M[(0,0, ... ,1,0)]
    ...
    M[(0,1, ... ,0,0)]
    M[(1,0, ... ,0,0)]
    which are indexes
    1, M.shape[-1], M.shape[-1]*M.shape[-2], ... when looking at M.ravel().

    Parameters
    ----------
    M : numpy array
        The coefficient array to get the linear terms from

    Returns
    -------
    A: numpy array
        An array with the linear terms of M
    """
    A = []
    spot = 1
    for i in M.shape[::-1]:
        A.append(0 if i == 1 else M.ravel()[spot])
        spot *= i
    return A[::-1] # Return linear terms in dimension order.

def find_vertices(A_ub, b_ub, tol = .05):
    """
    This function calculates the feasible point that is most inside the halfspace. 
    It then feeds that feasible point into the halfspace intersection solver and finds the intersection of the hyper-polygon and the hyper-cube. 
    It returns these intersection points (which when we take the min and max of, gives the interval we should shrink down to).

    The algorithm for the first half of this function (the portion that calculates the feasible point) is found at:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.HalfspaceIntersection.html

    Parameters
    ----------
    A_ub : numpy array
        #This is created as: A_ub = np.vstack([A, -A]), where A is the matrix A from BoundingIntervalLinearSystem
    b_ub : numpy vector (1D)
        #This is created as: b_ub = np.hstack([err - consts, consts + err]).T.reshape(-1, 1), where err and consts are from BoundingIntervalLinearSystem

    Returns
    -------
    tell : int
        This is a variable that is used to tell how the linear programming proceeded:
            1 : The linear programming found no feasible point, so the interval should be thrown out.
            2 : The linear programming found a feasible point, but the halfspaces code said it was not "clearly inside" the halfspace.
                This happens when the coefficients and error are all really tiny. In this case we are zooming in on a root, so we should keep the entire interval.
            3 : This means the linear programming and halfspace code ran without error, and a new interval was found
            4 : This means the linear programming code failed in such a way that we cannot use this method and need to use Erik's linear linear code instead.
    intersects : numpy array (ND)
        An array of the vertices of the intersection of the hyper-polygon and hyper-cube 
    """
    
    #Get the shape of the A_ub matrix
    m, n = A_ub.shape

    #Create an array used as part of the halfspaces array below, and find its number of rows
    arr = np.hstack([np.vstack([np.identity(n, dtype = float), -np.identity(n, dtype = float)]), -np.ones(2 * n, dtype = float).reshape(2 * n, 1)])
    o = arr.shape[0]

    # Create the halfspaces array in the format the scipy halfspace solver needs it
    halfspaces = np.zeros((m + o, n + 1), dtype = float)
    halfspaces[:m, :n] = A_ub
    halfspaces[:m, n:] = -b_ub
    halfspaces[m:, :] = arr

    # Find a feasible point that is MOST inside the halfspace
    #The documentation for this algorithm is found at: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.HalfspaceIntersection.html
    norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1), (halfspaces.shape[0], 1))
    c = np.zeros((halfspaces.shape[1],), dtype = float)
    c[-1] = -1
    A = np.hstack((halfspaces[:, :-1], norm_vector))
    b = - halfspaces[:, -1:]

    #Run the linear programming code
    L = linprog(c, A, b, bounds = (-1+1e-12,1-1e-12), method = "highs")
    
    #If L.status == 0, it means the linear programming proceeded as normal and a feasible point was found
    if L.status == 0:
        #L.x is the vector that contains the feasible points.
        #The last element of the vector is not part of the feasible point (will be discussed below) so should not be included in what we return.
        feasible_point = L.x[:-1]
    else: 
        #If L.status is not 0, then something went wrong with the linear programming, so we should use Erik's code instead.
        return 4, None

    if L.x[-1] < 0:
        #We have halfspaces of the form Ax + b <= 0. However we want to solve for the feasible point that is MOST inside of the halfspace.
        #To do this, we solve the problem: maximize y subject to: Ax + y||Ai|| <= -b (where Ai are the rows of A)
        #This problem always has a solution even if the original problem (Ax + b) does not have a solution, because you can just make y negative as large as you need
        #Thus if y is negative, the optimization problem Ax + y||Ai|| <= -b has a solution, but the one we care about, Ax + b <= 0, does not have a solution.
        #L.x[-1] = y. Thus if it is negative, the optimization problem we care about has no solution, so we should throw out the entire interval.
        if np.isclose(np.min(norm_vector)*L.x[-1], 0 ):
            #HOWEVER, if y is negative but y*min(||Ai||) is extremely close to 0, it is likely that y is negative due to roundoff error and not because the problem Ax + b <= 0 has no solution.
            #Thus in this case we conclude that we cannot continue to run the halfspaces code due to this roundoff error, and we return a 4 which passes the problem to Erik's linear code instead.
            return 4, None
        else:
            #As explained above, when y is negative and y*min(||Ai||) is NOT close to zero, Ax + b <= 0 has no solution, and so we should throw out the interval by returning 1
            return 1, None
      
    #If the linear programming problem succeeded, try running the halfspaces code
    try:
        intersects = HalfspaceIntersection(halfspaces, feasible_point).intersections
    except QhullError as e:
        #If the feasible point is still not clearly inside the halfspaces, this error is thrown. 
        #In this case, the entire boundary is too close to the edge of the [-1,1]^n box so halfspaces will struggle. Thus we call Erik's linear programming code instead.
        return 4, None

    #If the problem can be solved, it means the interval was shrunk, so return the intersections
    return 3, intersects

@njit
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

def ReduceDimension(A, consts, errs, errors):
    """Reduces dimension of a region by 1 if it is intersected with at least 1 hyperplane

    Parameters
    ----------
    A : numpy array
        The matrix of linear terms
    consts : numpy vector (1D)
        The vector of constant terms
    errs : numpy vector (1D)
        The vector of maximum errors of linear chebyshev approximation
    errors : numpy vector (1D)
        The given errors of chebyshev approximation

    Returns
    -------
    new_A : numpy array
        Array of new linear terms
    new_consts : numpy vector (1D)
        Vector containing the new constant terms
    new_errs : numpy vector (1D)
        Vector containing new linear errors
    new_errors : numpy vector (1D)
        Vector containing new total errors
    row_idx : int
        The index of the row removed
    col_idx : int
        The index of the column removed
    del_row : numpy vector (1D)
        The normalized row taken out
    del_norm : float
        The largest value of the original del_row
    """

    #Get index of first row that has corresponding err less than 1e-11
    row_idx = np.where(errs <= 1e-11)[0][0]
    del_row = A[row_idx]
    col_idx = np.where(abs(del_row.flatten())==np.max(abs(del_row)))[0][0]
    del_norm = del_row[col_idx]
    consts[row_idx] /= del_norm
    errs[row_idx] /= del_norm
    errors[row_idx] /= del_norm
    del_row /= del_norm
    reduced_A = A-np.matmul(np.vstack(A[:,col_idx]),[del_row])
    new_A = np.vstack([np.hstack([reduced_A[:row_idx,:col_idx],reduced_A[:row_idx,col_idx+1:]]),np.hstack([reduced_A[row_idx+1:,:col_idx],reduced_A[row_idx+1:,col_idx+1:]])])
    reduced_consts = consts - consts[row_idx]*A[:,col_idx]
    new_consts = np.hstack([reduced_consts[:row_idx],reduced_consts[row_idx+1:]])
    reduced_errs = errs + abs(errs[row_idx]*A[:,col_idx])
    new_errs = np.hstack([reduced_errs[:row_idx],reduced_errs[row_idx+1:]])
    reduced_errors = errors + abs(errors[row_idx]*A[:,col_idx])
    new_errors = np.hstack([reduced_errors[:row_idx],reduced_errors[row_idx+1:]])
    return new_A,new_consts,new_errs,new_errors,row_idx,col_idx,del_row,del_norm

def RetrieveDimension(bounds, errs, consts, plane_consts, row_idx, col_idx, del_norm):
    """
    """
    plane_consts = np.hstack([plane_consts[:col_idx],plane_consts[col_idx+1:]])
    idxs = ((np.sign(plane_consts)+1)/2).astype(int)
    linear_terms = bounds[np.arange(len(idxs)),idxs]
    min = consts[row_idx]/del_norm - np.dot(linear_terms,plane_consts) - abs(errs[row_idx])
    idxs = 1-idxs
    linear_terms = bounds[np.arange(len(idxs)),idxs]
    max = consts[row_idx]/del_norm - np.dot(linear_terms,plane_consts) + abs(errs[row_idx])
    minmax = np.array([min,max])
    new_bounds = np.vstack([bounds[:row_idx,:],minmax,bounds[row_idx:,:]])
    return new_bounds

def BoundingIntervalLinearSystem(Ms, errors, finalStep, linear = False):
    """Finds a smaller region in which any root must be.

    Parameters
    ----------
    Ms : list of numpy arrays
        Each numpy array is the coefficient tensor of a chebyshev polynomials
    errors : iterable of floats
        The maximum error of chebyshev approximations
    finalStep : bool
        Whether we are in the final step of the algorithm

    Returns
    -------
    newInterval : numpy array
        The smaller interval where any root must be
    changed : bool
        Whether the interval has shrunk at all
    should_stop : bool
        Whether we should stop subdividing
    throwout :
        Whether we should throw out the interval entirely
    """
    if finalStep:
        errors = np.zeros_like(errors)

    dim = Ms[0].ndim
    #Some constants we use here
    widthToAdd = 1e-10 #Add this width to the new intervals we find to avoid rounding error throwing out roots
    minZoomForChange = 0.99 #If the volume doesn't shrink by this amount say that it hasn't changed
    minZoomForBaseCaseEnd = 0.4**dim #If the volume doesn't change by at least this amount when running with no error, stop
    #Get the matrix of the linear terms
    A = np.array([getLinearTerms(M) for M in Ms])
    #Get the Vector of the constant terms
    consts = np.array([M.ravel()[0] for M in Ms])
    #Get the Error of everything else combined.
    totalErrs = np.array([np.sum(np.abs(M)) + e for M,e in zip(Ms, errors)])
    linear_sums = np.sum(np.abs(A),axis=1)
    err = np.array([tE-abs(c)-l for tE,c,l in zip(totalErrs,consts,linear_sums)])

    #Scale all the polynomials relative to one another
    errors = errors.copy()
    for i in range(dim):
        scaleVal = np.max(np.abs(A[i]))
        if scaleVal > 0:
            s = 2.**int(np.floor(np.log2(abs(scaleVal))))
            A[i] /= s
            consts[i] /= s
            totalErrs[i] /= s
            linear_sums[i] /= s
            err[i] /= s
            errors[i] /= s
    #Precondition the columns. (AP)X = B -> A(PX) = B. So scale columns, solve, then scale the solution.
    colScaler = np.ones(dim)
    for i in range(dim):
        scaleVal = np.max(np.abs(A[:,i]))
        if scaleVal > 0:
            s = 2**(-np.floor(np.log2(abs(scaleVal))))
            colScaler[i] = s
            totalErrs += np.abs(A[:,i]) * (s - 1)
            A[:,i] *= s
    
    #Run Erik's code if the dimension of the problem is <= 4, OR if the variable "linear" equals True.
    #The variable "Linear" is False by default, but in the case where we tried to run the halfspaces code (meaning the problem has dimension 5 or greater)
    #And the function "find_vertices" failed (i.e. it returned a value of 4), it means that the halfspaces code cannot be run and we need to run Erik's linear code on that interval instead.
    if dim <= 4 or linear:
        #This loop will only execute the second time if the interval was not changed on the first iteration and it needs to run again with tighter errors
        for i in range(2):
            #Use the other interval shrinking method
            a, b = linearCheck1(totalErrs, A, consts)
            #Now do the linear solve check
            U, S, Vh = np.linalg.svd(A)
            wellConditioned = S[0] > 0 and S[-1]/S[0] > 1e-10
            #We use the matrix inverse to find the width, so might as well use it both spots. Should be fine as dim is small.
            if wellConditioned: #Make sure conditioning is ok.
                Ainv = (1/S * Vh.T) @ U.T
                center = -Ainv@consts

                #Ainv transforms the hyperrectangle of side lengths err into a parallelogram with these as the principal direction
                #So summing over them gets the farthest the parallelogram can reach in each dimension.
                width = np.sum(np.abs(Ainv*err),axis=1)
                #Bound with previous result
                a = np.maximum(center - width, a)
                b = np.minimum(center + width, b)
            #Undo the column preconditioning
            a *= colScaler
            b *= colScaler
            #Add error and bound
            a -= widthToAdd
            b += widthToAdd
            throwOut = np.any(a > b) or np.any(a > 1) or np.any(b < -1)
            a[a < -1] = -1
            b[b < -1] = -1
            a[a > 1] = 1
            b[b > 1] = 1

            forceShouldStop = finalStep and not wellConditioned
            # Calculate the "changed" variable
            newRatio = np.product(b - a) / 2**dim
            if throwOut:
                changed = True
            elif i == 0:
                changed = newRatio < minZoomForChange
            else:
                changed = newRatio < minZoomForBaseCaseEnd

            if i == 0 and changed:
                #If it is the first time through the loop and there was a change, return the interval it shrunk down to and set "is_done" to false
                return np.vstack([a,b]).T, changed, forceShouldStop, throwOut
            elif i == 0 and not changed:
                #If it is the first time through the loop and there was not a change, save the a and b as the original values to return,
                #and then try running through the loop again with a tighter error to see if we shrink then
                a_orig = a
                b_orig = b
                err = errors
            elif changed:
                #If it is the second time through the loop and it did change, it means we didn't change on the first time,
                #but that the interval did shrink with tighter errors. So return the original interval with changed = False and is_done = False 
                return np.vstack([a_orig, b_orig]).T, False, forceShouldStop, False
            else:
                #If it is the second time through the loop and it did NOT change, it means we will not shrink the interval even if we subdivide,
                #so return the original interval with changed = False and is_done = wellConditioned 
                return np.vstack([a_orig,b_orig]).T, False, wellConditioned or forceShouldStop, False

    #Use the halfspaces code in the case when dim >= 5

    num_dim_reductions = 0
    A_orig,consts_orig,err_orig,errors_orig = A,consts,err,errors
    if (np.any(err<1e-10)):
        num_dim_reductions+=1
        A,consts,err,errors,row_idx,col_idx,del_row,del_norm = ReduceDimension(A,consts,err,errors)

    #Define the A_ub matrix
    A_ub = np.vstack([A, -A])
    
    #This loop will only execute the second time if the interval was not changed on the first iteration and it needs to run again with tighter errors
    for i in range(2):

        #Bound err below by 1e-11 so approximations are not so tight that the halfspaces code doesn't work
        err_lb = 1e-11
        err = np.array(err)
        err = np.where(err<err_lb,err_lb,err)
        #Define the b_ub vector
        b_ub = np.hstack([err - consts, consts + err]).T.reshape(-1, 1)

        # Use the find_vertices function to return the vertices of the intersection of halfspaces
        tell, vertices = find_vertices(A_ub, b_ub)

        if tell == 4:
            return BoundingIntervalLinearSystem(Ms, errors_orig, finalStep, True)

        if i == 0 and tell == 1:
            #First time through and no feasible point so throw out the entire interval
            return np.vstack([[1.0]*len(A_orig),[-1.0]*len(A_orig)]).T, True, True, True

        elif i == 0 and tell != 1:
            if tell == 2:
                #This should never occur
                #This means we have zoomed up on a root and we should be done.
                return np.vstack([[-1.0]*len(A),[1.0]*len(A)]).T, False, True, False
            
            #Get the a and b arrays
            a = vertices.min(axis=0)
            b = vertices.max(axis=0)

            #Undo dimension reduction
            if (num_dim_reductions > 0):
                bounds = np.hstack([np.vstack(a),np.vstack(b)])
                new_bounds = RetrieveDimension(bounds,err_orig,consts_orig,del_row,row_idx,col_idx,del_norm)
                a = new_bounds[:,0]
                b = new_bounds[:,1]

            #Undo the column preconditioning
            a *= colScaler
            b *= colScaler
            #Add error and bound
            a -= widthToAdd
            b += widthToAdd
            throwOut = np.any(a > b) or np.any(a > 1) or np.any(b < -1)
            a[a < -1] = -1
            b[b < -1] = -1
            a[a > 1] = 1
            b[b > 1] = 1
            
            #Calculate the newRatio, the changed, and the throwout variables
            newRatio = np.product(b - a) / 2**dim
            changed = newRatio < minZoomForChange
            
            if changed:
                #If it is the first time through and there was a change, return the interval it shrunk down to and set "is_done" to false
                return np.vstack([a,b]).T, True, False, throwOut
            else:
                #If it is the first time through the loop and there was not a change, save the a and b as the original values to return.
                #Then try running through the loop again with a tighter error to see if we shrink with smaller error
                a_orig = a
                b_orig = b
                err = errors

        #If second time through:
        elif i == 1:

            if tell == 1 or tell == 2:
                #This means there was an issue when we ran the code with the smaller error, so some intervals may be good but some bad.
                #We will need to shrink down to find out. So return with is_done = False so that we subdivide.
                return np.vstack([a_orig, b_orig]).T, False, False, False

            else: #The system proceeded as normal
                
                #Get the a and b arrays
                a = vertices.min(axis=0) - widthToAdd
                b = vertices.max(axis=0) + widthToAdd

                if (num_dim_reductions > 0):
                    bounds = np.hstack([np.vstack(a),np.vstack(b)])
                    new_bounds = RetrieveDimension(bounds,err_orig,consts_orig,del_row,row_idx,col_idx,del_norm)
                    a = new_bounds[:,0]
                    b = new_bounds[:,1]

                #Adjust the a's and b's to account for slight error in the halfspaces code
                a -= widthToAdd
                b += widthToAdd
                a[a < -1] = -1
                b[b < -1] = -1
                a[a > 1] = 1
                b[b > 1] = 1
                
                #Calculate the newRatio, the changed, and the throwout variables
                newRatio = np.product(b - a) / 2**dim
                changed = newRatio < minZoomForBaseCaseEnd

                if changed:
                    #IThis means it didn't change the first time, but with tighter errors it did.
                    #Thus we should continue shrinking in, so set is_done = False.
                    return np.vstack([a_orig, b_orig]).T, False, False, False
                else:
                    #If it is the second time through the loop and it did NOT change, it means we will not shrink the interval 
                    #even if we subdivide, so return the original interval with changed = False and is_done = True
                    return np.vstack([a_orig, b_orig]).T, False, True, False
   
@njit(UniTuple(float64,2)(float64, float64))
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

@njit(UniTuple(float64,2)(float64))
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

@njit(UniTuple(float64,2)(float64, float64))
def TwoProd(a,b):
    """Returns x,y such that a*b=x+y exactly and a*b=x in floating point using numba."""
    x = a*b
    a1,a2 = Split(a)
    b1,b2 = Split(b)
    y=a2*b2-(((x-a1*b1)-a2*b1)-a1*b2)
    return x,y
def TwoProd_NoNumba(a,b):
    """Returns x,y such that a*b=x+y exactly and a*b=x in floating point without usin numba."""
    x = a*b
    a1,a2 = Split_NoNumba(a)
    b1,b2 = Split_NoNumba(b)
    y=a2*b2-(((x-a1*b1)-a2*b1)-a1*b2)
    return x,y

@njit(UniTuple(float64,2)(float64, float64, float64, float64))
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
    error = M.shape[dim] * machEps * np.sum(np.abs(M))
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
        max_length = np.max(dim_lengths)
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
        #Use slicing to look at a slice of the highest degree in the dimension we want to trim
        slices = [slice(None) for i in range(dim)] # equivalent to selecting everything
        for currDim in range(dim):
            slices[currDim] = -1 # Now look at just the last row of the current dimension's approximation
            lastSum = np.sum(np.abs(Ms[polyNum][tuple(slices)]))

            # Iteratively eliminate the highest degree row of the current dimension if
            # the sum of its approximation coefficients is of low error, but keep deg at least 2
            while lastSum < allowedErrorIncrease and Ms[polyNum].shape[currDim] > 3:
                # Trim the polynomial
                slices[currDim] = slice(None,-1)
                Ms[polyNum] = Ms[polyNum][tuple(slices)] 
                # Update the remaining error increase allowed an the error of the approximation.
                allowedErrorIncrease -= lastSum
                errors[polyNum] += lastSum
                # Reset for the next iteration with the next highest degree of the current dimension.
                slices[currDim] = -1
                lastSum = np.sum(np.abs(Ms[polyNum][tuple(slices)]))
            # Reset to select all of the current dimension when looking at the next dimension.
            slices[currDim] = slice(None)

def isExteriorInterval(originalInterval, trackedInterval):
    """Determines if the current interval is exterior to its original interval."""
    return np.any(trackedInterval.getIntervalForCombining() == originalInterval.getIntervalForCombining())

def solvePolyRecursive(Ms, trackedInterval, errors, solverOptions):
    """Recursively shrinks and subdivides the given interval to find the locations of all roots.

    Parameters
    ----------
    Ms : list of numpy arrays
        The chebyshev approximations of the functions
    trackedInterval : TrackedInterval
        The information about the interval we are solving on.
    errors : numpy array
        An upper bound for the error of the Chebyshev approximation of the function on the interval
    solverOptions : SolverOptions
        Desired settings for running interval checks, transformations, and subdivision.
    
    Returns
    -------
    boundingBoxesInterior : list of numpy arrays (optional)
        Each element of the list is an interval in which there may be a root. The interval is on the interior of the current
        interval
    boundingBoxesExterior : list of numpy arrays (optional)
        Each element of the list is an interval in which there may be a root. The interval is on the exterior of the current
        interval
    """    
    #TODO: Check if trackedInterval.interval has width 0 in some dimension, in which case we should get rid of that dimension.
    #If the interval is a point, return it
    if trackedInterval.isPoint():
        return [], [trackedInterval]
    
    #If we ever change the options in this function, we will need to do a copy here.
    #Should be cheap, but as we never change them for now just avoid the copy
    solverOptions = solverOptions.copy()
    solverOptions.level += 1
    
    #Constant term check, runs at the beginning of the solve and before each subdivision
    #If the absolute value of the constant term for any of the chebyshev polynomials is greater than the sum of the
    #absoulte values of any of the other terms, it will return that there are no zeros on that interval
    if solverOptions.constant_check:
        consts = np.array([M.ravel()[0] for M in Ms]) 
        err = np.array([np.sum(np.abs(M))-abs(c)+e for M,e,c in zip(Ms,errors,consts)])
        if np.any(np.abs(consts) > err):
            return [], []
    
    #Runs quadratic check after constant check, only for dimensions 2 and 3 by default
    #More expensive than constant term check, but testing show it saves time in lower dimensions
    if (solverOptions.low_dim_quadratic_check and Ms[0].ndim <= 3) or solverOptions.all_dim_quadratic_check:
        for i in range(len(Ms)):
            if quadratic_check(Ms[i], errors[i]):
                return [], []
    
    #Trim
    Ms = Ms.copy()
    originalMs = Ms.copy()
    trackedInterval = trackedInterval.copy()
    errors = errors.copy()
    tolerable_error = max(errors) * 1e-3
    trimMs(Ms, errors)

    #Solve
    dim = Ms[0].ndim
    changed = True
    zoomCount = 0
    originalInterval = trackedInterval.copy()
    originalIntervalSize = trackedInterval.size()
    #Zoom in while we can
    lastSizes = trackedInterval.dimSize()
    start_time = time()
    while changed and zoomCount <= solverOptions.maxZoomCount:
        #Zoom in until we stop changing or we hit machine epsilon
        Ms, errors, trackedInterval, changed, should_stop = zoomInOnIntervalIter(Ms, errors, trackedInterval, solverOptions.exact)
        if trackedInterval.empty: #Throw out the interval
            return [], []
        #Only count in towards the max is we don't cut the interval in half
        newSizes = trackedInterval.dimSize()
        if np.all(newSizes >= lastSizes / 2): #Check all dims and use >= to account for a dimension being 0.
            zoomCount += 1
        lastSizes = newSizes
    finish_time = time()
    if should_stop:
        #Start the final step if the is in the options and we aren't already in it.
        if trackedInterval.finalStep or not solverOptions.useFinalStep:
            if solverOptions.verbose:
                print("*",end="")
            if isExteriorInterval(originalInterval, trackedInterval):
                return [], [trackedInterval]
            else:
                return [trackedInterval], []
        else:
            trackedInterval.startFinalStep()
            return solvePolyRecursive(Ms, trackedInterval, errors, solverOptions)
    elif trackedInterval.finalStep:
        trackedInterval.canThrowOutFinalStep = True
        allMs, allErrors, allIntervals = getSubdivisionIntervals(Ms, errors, trackedInterval, solverOptions.exact, solverOptions.level)
        resultsAll = []
        for newMs, newErrs, newInt in zip(allMs, allErrors, allIntervals):
            newInterior, newExterior = solvePolyRecursive(newMs, newInt, newErrs, solverOptions)
            resultsAll += newInterior + newExterior
        if len(resultsAll) == 0:
            #Can't throw out final step! This might not actually be a root though!
            trackedInterval.possibleExtraRoot = True
            if isExteriorInterval(originalInterval, trackedInterval):
                return [], [trackedInterval]
            else:
                return [trackedInterval], []
        else:
            #Combine all roots that converged to the same point.
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
            #TODO: Don't subdivide in the final step in dimensions that are already points!
    else:       
        #Otherwise, Subdivide
        if solverOptions.level == 15:
            warnings.warn(f"High subdivision depth!\nSubdivision on the search interval has now reached" +
                          " at least depth 15. Runtime may be prolonged.")
        elif solverOptions.level == 25:
            warnings.warn(f"Extreme subdivision depth!\nSubdivision on the search interval has now reached" +
                          " at least depth 25, which is unusual. The solver may not finish running." +
                          "Ensure the input functions meet the requirements of being continuous, smooth," +
                          "and having only finitely many simple roots on the search interval.")
        resultInterior, resultExterior = [], []
        #Get the new intervals and polynomials
        allMs, allErrors, allIntervals = getSubdivisionIntervals(Ms, errors, trackedInterval, solverOptions.exact, solverOptions.level)
        #Run each interval
        for newMs, newErrs, newInt in zip(allMs, allErrors, allIntervals):
            newInterior, newExterior = solvePolyRecursive(newMs, newInt, newErrs, solverOptions)
            resultInterior += newInterior
            resultExterior += newExterior
        #Rerun the touching intervals
        idx1 = 0
        idx2 = 1
        #Combine any touching intervals and throw them at the end. Flip a bool saying rerun them
        #If changing this code, test it by defaulting the nextTransformationsInterals to 0, so roots lie on the boundary more.
        #TODO: Make the combining intervals it's own function!!!
        for tempInterval in resultExterior:
            tempInterval.reRun = False
        while idx1 < len(resultExterior):
            while idx2 < len(resultExterior):
                if resultExterior[idx1].overlapsWith(resultExterior[idx2]):
                    #Combine, throw at the back. Set reRun to true.
                    combinedInterval = originalInterval.copy()
                    if combinedInterval.finalStep:
                        combinedInterval.interval = combinedInterval.preFinalInterval.copy()
                        combinedInterval.transforms = combinedInterval.preFinalTransforms.copy()
                    newAs = np.min([resultExterior[idx1].getIntervalForCombining()[:,0], resultExterior[idx2].getIntervalForCombining()[:,0]], axis=0)
                    newBs = np.max([resultExterior[idx1].getIntervalForCombining()[:,1], resultExterior[idx2].getIntervalForCombining()[:,1]], axis=0)
                    final1 = resultExterior[idx1].getFinalInterval()
                    final2 = resultExterior[idx2].getFinalInterval()
                    newAsFinal = np.min([final1[:,0], final2[:,0]], axis=0)
                    newBsFinal = np.max([final1[:,1], final2[:,1]], axis=0)
                    oldAs = originalInterval.interval[:,0]
                    oldBs = originalInterval.interval[:,1]
                    oldAsFinal, oldBsFinal = originalInterval.getFinalInterval().T
                    #Find the final A and B values exactly. Then do the currSubinterval calculation exactly.
                    #Look at what was done on the example that's failing and see why.
                    equalMask = oldBsFinal == oldAsFinal
                    oldBsFinal[equalMask] = oldBsFinal[equalMask] + 1 #Avoid a divide by zero on the next line
                    currSubinterval = ((2*np.array([newAsFinal, newBsFinal]) - oldAsFinal - oldBsFinal)/(oldBsFinal - oldAsFinal)).T
                    #If the interval is exactly -1 or 1, make sure that shows up as exact.
                    currSubinterval[equalMask,0] = -1
                    currSubinterval[equalMask,1] = 1
                    currSubinterval[:,0][oldAs == newAs] = -1
                    currSubinterval[:,1][oldBs == newBs] = 1
                    #Update the current subinterval. Use the best transform we can get here, but use the exact combined
                    #interval for tracking
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
        #Rerun, check if still on exterior
        newResultExterior = []
        for tempInterval in resultExterior:
            if tempInterval.reRun:
                if np.all(tempInterval.interval == originalInterval.interval):
                    newResultExterior.append(tempInterval)
                else:
                    #Project the MS onto the interval, then recall the function.
                    #TODO: Instead of using the originalMs, use Ms, and then don't use the original interval, use the one
                    #we started subdivision with.
                    tempMs, tempErrors = transformChebToInterval(originalMs, *tempInterval.getLastTransform(), errors, solverOptions.exact)
                    tempResultsInterior, tempResultsExterior = solvePolyRecursive(tempMs, tempInterval, tempErrors, solverOptions)
                    #We can assume that nothing in these has to be recombined
                    resultInterior += tempResultsInterior
                    newResultExterior += tempResultsExterior
            elif isExteriorInterval(originalInterval, tempInterval):
                newResultExterior.append(tempInterval)
            else:
                resultInterior.append(tempInterval)
        return resultInterior, newResultExterior

def solveChebyshevSubdivision(Ms, errors, verbose = False, returnBoundingBoxes = False, exact = False, constant_check = True, low_dim_quadratic_check = True, all_dim_quadratic_check = False):
    """Initiates shrinking and subdivision recursion and returns the roots and bounding boxes.

    Parameters
    ----------
    Ms : list of numpy arrays
        The chebyshev approximations of the functions on the interval given to CombinedSolver
    errors : numpy array
        The max error of the chebyshev approximation from the function on the interval
    verbose : bool
        Defaults to False. Whether or not to output progress of solving to the terminal.
    returnBoundingBoxes : bool (Optional)
        Defaults to False. If True, returns the bounding boxes around each root as well as the roots.
    exact : bool
        Whether transformations should be done with higher precision to minimize error.
    constant_check : bool
        Defaults to True. Whether or not to run constant term check after each subdivision.
    low_dim_quadratic_check : bool
        Defaults to True. Whether or not to run quadratic check in dim 2, 3.
    all_dim_quadratic_check : bool
        Defaults to False. Whether or not to run quadratic check in dim >= 4.

    Returns
    -------
    roots : list
        The roots of the system of functions on the interval given to Combined Solver
    boundingBoxes : list of numpy arrays (optional)
        List of intervals for each root in which the root is bound to lie.
    """
    #Assert that we have n nD polys
    if np.any([M.ndim != len(Ms) for M in Ms]):
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

    if verbose:
        print("Finding roots...", end=' ')
    b1, b2 = solvePolyRecursive(Ms, originalInterval, errors, solverOptions)

    boundingIntervals = b1 + b2
    roots = []
    hasDupRoots = False
    hasExtraRoots = False
    for interval in boundingIntervals:
        #TODO: Figure out the best way to return the bounding intervals.
        #Right now interval.finalInterval is the interval where we say the root is.
        interval.getFinalInterval()
        if interval.possibleExtraRoot:
            hasExtraRoots = True
        if len(interval.possibleDuplicateRoots) > 0:
            roots += interval.possibleDuplicateRoots
            hasDupRoots = True
        else:
            roots.append(interval.getFinalPoint())
    #Warn if extra or duplicate roots
    if hasExtraRoots:
        warnings.warn(f"Might Have Extra Roots! See Bounding Boxes for details!")
    if hasDupRoots:
        warnings.warn(f"Might Have Duplicate Roots! See Bounding Boxes for details!")
    #Return
    roots = np.array(roots)
    if verbose:
        finish_string = '\n' + f"Found {len(roots)} roots"
        print((finish_string if len(roots) != 1 else finish_string[:-1]),end='\n\n')
    if returnBoundingBoxes:
        return roots, boundingIntervals
    else:
        return roots