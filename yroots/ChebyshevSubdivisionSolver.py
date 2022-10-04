import numpy as np
from numba import njit
from itertools import product
from scipy.spatial import HalfspaceIntersection
from scipy.optimize import linprog

@njit
def TransformChebInPlace1D(coeffs, alpha, beta):
    """Applies the transformation alpha*x + beta to the chebyshev polynomial coeffs.

    Written for 1D, but also works in ND to transform dimension 0.

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
    transformedCoeffs = np.zeros_like(coeffs)
    arr1 = np.zeros(len(coeffs))
    arr2 = np.zeros(len(coeffs))
    arr3 = np.zeros(len(coeffs))

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
        arr3[0] = -arr1[0] + alpha*arr2[1] + 2*beta*arr2[0]
        transformedCoeffs[0] += thisCoeff * arr3[0]

        #The 1 spot
        if maxRow > 2:
            arr3[1] = -arr1[1] + alpha*(2*arr2[0] + arr2[2]) + 2*beta*arr2[1]
            transformedCoeffs[1] += thisCoeff * arr3[1]
        
        #The middle spots
        for i in range(2, maxRow - 1):
            arr3[i] = -arr1[i] + alpha*(arr2[i-1] + arr2[i+1]) + 2*beta*arr2[i]
            transformedCoeffs[i] += thisCoeff * arr3[i]

        #The second to last spot
        i = maxRow - 1
        arr3[i] = -arr1[i] + (2 if i == 1 else 1)*alpha*(arr2[i-1]) + 2*beta*arr2[i]
        transformedCoeffs[i] += thisCoeff * arr3[i]

        #The last spot
        finalVal = alpha*arr2[i]
        if abs(finalVal) > 1e-16: #TODO: Justify this val!
            arr3[maxRow] = finalVal
            transformedCoeffs[maxRow] += thisCoeff * finalVal
            maxRow += 1
        
        arr = arr1
        arr1 = arr2
        arr2 = arr3
        arr3 = arr
    return transformedCoeffs[:maxRow]

@njit
def TransformChebInPlace1DErrorFree(coeffs, alpha, beta):
    """Applies the transformation alpha*x + beta to the chebyshev polynomial coeffs.

    Written for 1D, but also works in ND to transform dimension 0.
    Uses Error Free Transformations to minimize error in the matrix construction

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
    if alpha == 0.5:
        if beta == 0.5:
            return TransformChebInPlace1DErrorFreeSplit(coeffs, True)
        elif beta == -0.5:
            return TransformChebInPlace1DErrorFreeSplit(coeffs, False)
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
        #arr3[0] = -arr1[0] + alpha*arr2[1] + 2*beta*arr2[0]
        V1, E1 = TwoProdWithSplit(beta, 2*arr2[0], beta1, beta2)
        V2, E2 = TwoProdWithSplit(alpha, arr2[1], alpha1, alpha2)
        V3, E3 = TwoSum(V1, V2)
        V4, E4 = TwoSum(V3, -arr1[0])
        arr3[0] = V4
        arr3E[0] = -arr1E[0] + alpha*arr2E[1] + 2*beta*arr2E[0] + E1 + E2 + E3 + E4
        transformedCoeffs[0] += thisCoeff * (arr3[0] + arr3E[0])

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
def TransformChebInPlace1DErrorFreeSplit(coeffs, positiveBeta):
    #alpha = 0.5
    #beta = 0.5 if positiveBeta else -0.5
    
    betaSign = 1 if positiveBeta else -1
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
    #TODO: Could we calculate the allowed error beforehand and pass it in here?
    #TODO: Make this work for the power basis polynomials
    if alpha == 1.0 and beta == 0.0:
        return coeffs
    TransformFunc = TransformChebInPlace1DErrorFree if exact else TransformChebInPlace1D
    if dim == 0:
        return TransformFunc(coeffs, alpha, beta)
    else:
        order = np.array([dim] + [i for i in range(dim)] + [i for i in range(dim+1, coeffs.ndim)])
        backOrder = np.zeros(coeffs.ndim, dtype = int)
        backOrder[order] = np.arange(coeffs.ndim)
        return TransformFunc(coeffs.transpose(order), alpha, beta).transpose(backOrder)



class TrackedInterval:
    def __init__(self, interval):
        self.topInterval = interval
        self.interval = interval
        self.transforms = []
        self.ndim = len(self.interval)
        self.empty = False
    
    def addTransform(self, subInterval):
        #This function assumes the interval has non zero size.
        #Get the transformation in terms of alpha and beta
        if np.any(subInterval[:,0] > subInterval[:,1]):
            self.empty = True
            return
        a1,b1 = subInterval.T
        a2,b2 = self.interval.T
        alpha1, beta1 = (b1-a1)/2, (b1+a1)/2
        alpha2, beta2 = (b2-a2)/2, (b2+a2)/2
        self.transforms.append([alpha1, beta1])
        #Update the current interval
        for dim in range(self.ndim):
            for i in range(2):
                x = subInterval[dim][i]
                if x == -1.0 or x == 1.0:
                    continue #Don't change the current interval
                self.interval[dim][i] = alpha2[dim]*x+beta2[dim]
    
    def getLastTransform(self):
        return self.transforms[-1]
        
    def getFinalInterval(self):
        #Use the transformations and the topInterval
        #TODO: Make this a seperate function so it can use njit.
        finalInterval = self.topInterval.T
        finalIntervalError = np.zeros_like(finalInterval)
        for alpha,beta in self.transforms[::-1]:
            finalInterval, temp = TwoProd(finalInterval, alpha)
            finalIntervalError = alpha * finalIntervalError + temp
            finalInterval, temp = TwoSum(finalInterval,beta)
            finalIntervalError += temp
        finalInterval = finalInterval.T
        finalIntervalError = finalIntervalError.T
        self.finalInterval = finalInterval + finalIntervalError
        self.finalAlpha, alphaError = TwoSum(-finalInterval[:,0]/2,finalInterval[:,1]/2)
        self.finalAlpha += alphaError + (finalIntervalError[:,1] - finalIntervalError[:,0])/2
        self.finalBeta, betaError = TwoSum(finalInterval[:,0]/2,finalInterval[:,1]/2)
        self.finalBeta += betaError + (finalIntervalError[:,1] + finalIntervalError[:,0])/2
        return self.finalInterval
    
    def size(self):
        return np.product(self.interval[:,1] - self.interval[:,0])
    
    def copy(self):
        newone = TrackedInterval(self.topInterval)
        newone.interval = self.interval.copy()
        newone.transforms = self.transforms.copy()
        return newone

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



def find_vertices(A_ub, b_ub):
    # This calcualtes the feasible point that is MOST inside the halfspace
    #It then feeds that feasible point into a halfspace intersection solver and finds
    #the intersection of the hyper-polygon and the hyper-cube. It returns these intersections, which when
    #we take the min and max of, give the set of intervals that we should shrink down to.
    #I am going to document exactly what the function does, but wanted to get it pushed because I will be
    #working on finals and out of town for the next week.

    m, n = A_ub.shape

    arr = np.hstack([np.vstack([np.identity(n, dtype = float), -np.identity(n, dtype = float)]), -np.ones(2 * n, dtype = float).reshape(2 * n, 1)])

    o = arr.shape[0]

    # Create the halfspaces array in the format the scipy solver needs it
    halfspaces = np.zeros((m + o, n + 1), dtype = float)
    halfspaces[:m, :n] = A_ub
    halfspaces[:m, n:] = -b_ub
    halfspaces[m:, :] = arr

    # Find a feasible point that is MOST inside the halfspace
    norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1), (halfspaces.shape[0], 1))
    c = np.zeros((halfspaces.shape[1],), dtype = float)
    c[-1] = -1
    A = np.hstack((halfspaces[:, :-1], norm_vector))
    b = - halfspaces[:, -1:]
    
    L = linprog(c, A, b, bounds = (-1,1))
    
    #If L.status == 0, it means the linear programming proceeded as normal, so there is shrinkage that can occur in the interval
    if L.status == 0:
        feasible_point = L.x[:-1]
    else: 
        #If L.status is not 0, then there is no feasible point, meaning the entire interval can be thrown out
        return 1, None

    #If the last entry in the feasible point is negative, it also means there was not a suitable feasible point,
    #so the entire interval can be throw out
    if L.x[-1] < 0:
        return 1, None
      
    #Try solving the halfspace problem
    try:
        intersects = HalfspaceIntersection(halfspaces, feasible_point).intersections
    except:
        #If the halfspaces failed, it means the coefficnets were really tiny.
        #In this case it also means that we want to keep the entire interval because there is a root in this interval
        return 2, np.vstack([np.ones(n),-np.ones(n)])

    #If the problem can be solved, it means the interval was shrunk, so return the intersections
    return 3, intersects


def BoundingIntervalLinearSystem(Ms, errors):
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

    dim = A.shape[0]
    
    if dim <= 4:
        #Erik's method for shrinking the interval
        #Solve for the extrema
        #We use the matrix inverse to find the width, so might as well use it both spots. Should be fine as dim is small.
        try:
            Ainv = np.linalg.inv(A)
        except np.linalg.LinAlgError as e: #If it's not invertible we can't zoom in
            if len(A) == 1:
                return np.array([[-1.0,1.0]]), False
            else:
                return np.vstack([[-1.0]*len(A),[1.0]*len(A)]).T, False
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

    ##NEW CODE## I will document this much better, but wanted to get it pushed before finals/I leave town.
    #Define the A_ub and b_ub matrices in the correct form to feed into the linear programming problem.
    A_ub = np.vstack([A, -A])
    b_ub = np.hstack([err - consts, consts + err]).T.reshape(-1, 1)

    # Use the find_vertices function to return the vertices of the intersection of halfspaces
    tell, P = find_vertices(A_ub, b_ub)
    if tell == 1:
        #No feasible Point, throw out the entire interval
        return np.vstack([[1.0]*len(A),[-1.0]*len(A)]).T, True
    elif tell == 2:
        #No shrinkage possible, keep the entire interval
        return np.vstack([[-1.0]*len(A),[1.0]*len(A)]).T, False
    else:
        #Return the reduced interval
        a = P.min(axis=0)
        b = P.max(axis=0)     

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
    polyType = "C" #C for Chebyshev, P for Power
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
    maxRow = startValue - 1
    for j in range(startValue, n): #Loop over the columns starting at 2
        for i in range(j+1): #Loop over the rows on the upper diagonal
            maxRow = max(maxRow, i)
            val = 0
            aVal = 0
            bVal = 0
            if polyType == "C":
                if isValidSpot(i,j-2):
                    val -= M[i,j-2] #Adds no error
                if isValidSpot(i-1,j-1):
                    aVal += M[i-1,j-1] * (2 if i == 1 else 1) #Adds no error. Doubles magnitude of previous error.
                if isValidSpot(i,j-1):
                    bVal += 2*M[i,j-1] #Adds no error. Doubles magnitude of previous error.
                if isValidSpot(i+1,j-1):
                    aVal += M[i+1,j-1] #Could add machEps error
                M[i,j] = val + a*aVal + b*bVal #Could add 4 machEps error. Total of 5 machEps error added at most.
            elif polyType == "P":
                M[i,j] = a*M[i-1,j-1] + b*M[i,j-1]
            #TODO: Could we calculate the allowed error beforehand and pass it in here?
            #Not sure how to make that work when we want to store the matrix. Maybe
            #when we store the matrix we also store the max up to that row? That way we can easily grab a chunk of it.
            #TODO: Fix this. We could have a random 0 or small number anywhere if i > maxRow?
#             if i > 2 and abs(M[i,j]) < 1e-16: #Don't build the really small numbers into the matrix, assume it isn't worth it.
#                 break
    return M[:maxRow+1]

@njit
def TwoSum(a,b):
    x = a+b
    z = x-a
    y = (a-(x-z)) + (b-z)
    return x,y

@njit
def Split(a):
    c = (2**27 + 1) * a
    x = c-(c-a)
    y = a-x
    return x,y

@njit
def TwoProd(a,b):
    x = a*b
    a1,a2 = Split(a)
    b1,b2 = Split(b)
    y=a2*b2-(((x-a1*b1)-a2*b1)-a1*b2)
    return x,y

@njit
def TwoProdWithSplit(a,b,a1,a2):
    x = a*b
    b1,b2 = Split(b)
    y=a2*b2-(((x-a1*b1)-a2*b1)-a1*b2)
    return x,y

@njit
def makeMatrixErrorFree(n,a,b,subMatrix=None):
    polyType = "C" #C for Chebyshev, P for Power
    a1,a2 = Split(a)
    b1,b2 = Split(b)
    #Matrix creation with njit
    M = np.zeros(n*n)
    M = M.reshape(n,n)
    Me = np.zeros(n*n)
    Me = Me.reshape(n,n)
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
    maxRow = startValue - 1
    for j in range(startValue, n): #Loop over the columns starting at 2
        for i in range(j+1): #Loop over the rows on the upper diagonal
            maxRow = max(maxRow, i)
            if polyType == "C":
                #Sum the values for the As
                if isValidSpot(i-1,j-1) and isValidSpot(i+1,j-1):
                    AVal, AValE = TwoSum(M[i-1,j-1] * (2 if i == 1 else 1), M[i+1,j-1])
                    ErrorAVal = Me[i-1,j-1] * (2 if i == 1 else 1) + Me[i+1,j-1]
                elif isValidSpot(i-1,j-1):
                    AVal, AValE = M[i-1,j-1] * (2 if i == 1 else 1), 0
                    ErrorAVal = Me[i-1,j-1] * (2 if i == 1 else 1)
                elif isValidSpot(i+1,j-1):
                    AVal, AValE = M[i+1,j-1], 0
                    ErrorAVal = Me[i+1,j-1]
                else:
                    AVal, AValE = 0, 0
                    ErrorAVal = 0
                #Get the value for the b
                BVal = 0
                ErrorBVal = 0
                if isValidSpot(i,j-1):
                    BVal = 2*M[i,j-1]
                    ErrorBVal = 2*Me[i,j-1]
                #Get the final val
                Val = 0
                ErrorVal = 0
                if isValidSpot(i,j-2):
                    Val = -M[i,j-2]
                    ErrorVal = -Me[i,j-2]
                #TODO: Should I check for 0 values before these to speed it up?
                #Do the 2 multiplications
                P1, P1E = TwoProdWithSplit(a, AVal, a1, a2)
                P2, P2E = TwoProdWithSplit(b, BVal, b1, b2)
                #Do the final sum
                S1, S1E = TwoSum(P1, P2)
                S2, S2E = TwoSum(S1, Val)
                M[i,j] = S2
                Me[i,j] = ErrorVal + a*(ErrorAVal+AValE) + b*ErrorBVal + P1E + P2E + S1E + S2E
                if i > 2 and abs(M[i,j] + Me[i,j]) < 1e-32: #TODO: Justify This!
                    break
            elif polyType == "P":
                P1, P1E = TwoProdWithSplit(a, M[i-1,j-1], a1, a2)
                P2, P2E = TwoProdWithSplit(b, M[i,j-1], b1, b2)
                S1, S1E = TwoSum(P1, P2)
                M[i,j] = S1
                Me[i,j] = a*Me[i-1,j-1] + b*Me[i,j-1] + P1E + P2E + S1E     
                #TODO: Figure out what this should be got Power Basis
#                 if i > 2 and abs(M[i,j] + Me[i,j]) < 1e-320:
#                     break
    return (M + Me)[:maxRow+1]

def getTransformPoints(newInterval):
    """Given the new interval [a,b], gives c,d for reduction xHat = cx+d"""
    a,b = newInterval
    return (b-a)/2, (b+a)/2

def getTransformationError(M, dim):
    """Returns a bound on the error of transforming a chebyshev approximation M"""
    #The matrix is accurate to machEps, so the error is at most machEps * each number in the matrix
    #time the number of rows in the transformation, which is M.shape[dim]
    machEps = 2**-52
    error = M.shape[dim] * machEps * np.sum(np.abs(M))
    return error #TODO: Figure out a more rigurous bound!
    
def transformCheb(M, As, Bs, error):
    """Transforms the chebyshev coefficient matrix M to the new interval [As, Bs].

    Parameters
    ----------
    M : numpy array
        The chebyshev coefficient matrix
    As : iterable
        The min values of the interval we are transforming to
    Bs : iterable
        The max values of the interval we are transforming to
    error : float
        A bound on the error of the chebyshev approximation
    
    Returns
    -------
    M : numpy array
        The coefficient matrix on the new interval
    error : float
        A bound on the error of the new chebyshev approximation
    """
    #This just does the matrix multiplication on each dimension. Except it's by a tensor.
    exact = True
    for dim,n,a,b in zip(range(M.ndim),M.shape,As,Bs):
        error += getTransformationError(M, dim)
        M = TransformChebInPlaceND(M,dim,a,b,exact)
    return M, error

def transformChebToInterval(Ms, As, Bs, errors):
    """Transforms chebyshev coefficient matrices to a new interval.

    Parameters
    ----------
    Ms : list of numpy arrays
        The chebyshev coefficient matrices
    interval : numpy array
        The new interval to transform to
    originalInterval : numpy array
        The current interval on which Ms are valid
    errors : numpy array
        A bound on the error of each chebyshev approximation

    Returns
    -------
    newMs : list of numpy arrays
        The chebyshev coefficient matrices on the new interval
    newErrors : list of numpy arrays
        The errors of the newMs. This just adds the errors of applying the Chebyshev Transformation Matrix.
    """    
    #Transform the chebyshev polynomials
    newMs = []
    newErrors = []
    for M,e in zip(Ms, errors):
        newM, newE = transformCheb(M, As, Bs, e)
        newMs.append(newM)
        newErrors.append(newE)
    return newMs, np.array(newErrors)
    
def zoomInOnIntervalIter(Ms, errors, trackedInterval):
    """One iteration of the linear check and transforming to a new interval.

    Parameters
    ----------
    Ms : list of numpy arrays
        The chebyshev coefficient matrices
    errors : numpy array
        A bound on the error of each chebyshev approximation
    trackedInterval : TrackedInterval
        The current interval that the chebyshev approximations are valid for
    Returns
    -------
    Ms : list of numpy arrays
        The chebyshev coefficient matrices on the new interval
    errors : numpy array
        The errors of the new Ms. This just adds the errors of applying the Chebyshev Transformation Matrix.
    trackedInterval : TrackedInterval
        The new interval that the chebyshev approximations are valid for
    changed : bool
        Whether the interval changed as a result of this step.
    """    
    dim = len(Ms)
    #Zoom in on the current interval
    interval, changed = BoundingIntervalLinearSystem(Ms, errors)
    #Check if we can throw out the whole thing
    if np.any(interval[:,0] > interval[:,1]):
        trackedInterval.empty = True
        return Ms, errors, trackedInterval, True
    #Check if we are done interating
    if not changed:
        return Ms, errors, trackedInterval, changed
    #Transform the chebyshev polynomials
    trackedInterval.addTransform(interval)
    Ms, errors = transformChebToInterval(Ms, *trackedInterval.getLastTransform(), errors)
    return Ms, errors, trackedInterval, changed
    
def getTransposeDims(dim,transformDim):
    """Helper function for chebTransform1D"""
    return [i for i in range(transformDim,dim)] + [i for i in range(transformDim)]

def chebTransform1D(M, alpha, beta, transformDim):
    """Transform a chebyshev polynomial in a single dimension"""
    return TransformChebInPlaceND(M,transformDim,alpha, beta,True), getTransformationError(M, transformDim)

def getInverseOrder(order, dim):
    """Helper function to make the subdivide order match the subdivideInterval order"""
    order = 2**(len(order)-1 - order)
    newOrder = np.array([i@order for i in product([0,1],repeat=dim)])
    invOrder = np.zeros_like(newOrder)
    invOrder[newOrder] = np.arange(len(newOrder))
    return tuple(invOrder)
    
class Subdivider():
    #TODO: It might be better to just always do the transformations in place, and then get rid of this class.
        #Keep the functions but don't store anything.
    #This class handles subdividing and stores the precomputed matrices to save time.
    def __init__(self):
        #TODO: Make this 0.5. DO subdivide exactly in half. Requires combining intervals to work.
        self.RAND = 0.5139303900908738 #Don't subdivide exactly in half.
        self.precomputedArrayDeg = 1 #We don't compute the first 2
        self.subdivisionPoint =  self.RAND * 2 - 1
        self.transformPoints1 = getTransformPoints([-1., self.subdivisionPoint]) #TODO: This isn't always [-1,1]!!!
        self.transformPoints2 = getTransformPoints([self.subdivisionPoint,  1.])
        #Note that a transformation of a lower degree is just the submatrix of the higher degree transformation
        #So we can just store the highest degree transformation we have to save on space.
        #And we can compute the higher degree transformation starting at the degree we already have.
            
    def subdivideInterval(self, trackedInterval):
        #Get the new interval that will correspond with the new polynomials
        results = [trackedInterval]
        for thisDim in range(len(trackedInterval.interval)):
            newSubinterval = np.ones_like(trackedInterval.interval)
            newSubinterval[:,0] = -1.
            newResults = []
            for oldInterval in results:
                newInterval1 = oldInterval.copy()
                newInterval2 = oldInterval.copy()
                newSubinterval[thisDim] = [-1., self.subdivisionPoint]
                newInterval1.addTransform(newSubinterval)
                newSubinterval[thisDim] = [self.subdivisionPoint, 1.]
                newInterval2.addTransform(newSubinterval)
                newResults.append(newInterval1)
                newResults.append(newInterval2)
            results = newResults
        return results
        
    def subdivide(self, M, error):
        #Get the new Chebyshev approximations on the 2^n subintervals
        dim = M.ndim
        degs = M.shape
        order = np.argsort(degs)[::-1] #We want to subdivide from highest degree to lowest.
        #Iterate through the dimensions, highest degree first.
        resultMs = [[M,error]]
        for thisDim in order:
            thisDeg = M.shape[thisDim]
            newResults = []
            for T,E in resultMs:
                #Transform the polys
                P1, E1 = chebTransform1D(T, *self.transformPoints1, thisDim)
                P2, E2 = chebTransform1D(T, *self.transformPoints2, thisDim)
                newResults.append([P1, E1 + E])
                newResults.append([P2, E2 + E])
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
        print(errors) #DELETE THIS LINE
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
    intervals : list of lists of trackedIntervals
        Each list corresponds to one of the 2^n subintervals from the last step of subdivision, and contains a list of
        all the intervals in the subinterval in which there could be a root. Intervals in the same list will not be touching.
    
    Returns
    -------
    intervals : list of trackedInterval
        A new list of intervals such that each interval in the any of the old lists is contained in some interval in
        the new list, but none of the intervals in the new list are touching.
    """
    #TODO: WRITE THIS!!!
    #Since polishing doesn't seem to do any good, maybe we should just immediately run anything we combine.
    newIntervals = []
    for interval in intervals:
        newIntervals += interval
    return newIntervals

def shouldStopSubdivision(trackedInterval):
    #TODO: WRITE THIS!!!
    #In 1D a good check could be if the linear term is less than the error term (or close to it).
    #Not sure how to extend that to ND yet.
    #For now just checks if the interval is small. This won't work if there is a large error term.
    return np.all(trackedInterval.interval[:,1]-trackedInterval.interval[:,0] < 1e-10)

def solvePolyRecursive(Ms, trackedInterval, errors, trimErrorRelBound = 1e-16, trimErrorAbsBound = 1e-16):
    """Recursively finds regions in which any common roots of functions must be using subdivision

    Parameters
    ----------
    Ms : list of numpy arrays
        The chebyshev approximations of the functions
    trackedInterval : TrackedInterval
        The information about the interval we are solving on.
    errors : numpy array
        The max error of the chebyshev approximation from the function on the interval
    
    Returns
    -------
    boundingBoxes : list of numpy arrays (optional)
        Each element of the list is an interval in which there may be a root.
    """
    #The random numbers used below. TODO: Choose these better
    #Error For Trim trimMs
#     trimErrorAbsBound = 1e-16
#     trimErrorRelBound = 1e-16
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
    originalIntervalSize = trackedInterval.size()
    #The choosing when to stop zooming logic is really ugly. Needs more analysis.
    #Keep zooming while it's larger than minIntervalSize.
    while changed and np.max(trackedInterval.interval[:,1] - trackedInterval.interval[:,0]) > minIntervalSize:
        #If we've zoomed more than maxZoomCount1 and haven't shrunk the size by zoomRatioToZip, assume
        #we aren't making progress and subdivide. Once we get to zoomRatioToZip, assume we will just converge
        #quickly and zoom all the way by maxZoomCount2.
        if zoomCount > maxZoomCount1:
            newIntervalSize = trackedInterval.size()
            zoomRatio = (newIntervalSize / originalIntervalSize) ** (1/len(Ms))
            if zoomRatio >= zoomRatioToZip:
                break
            elif zoomCount > maxZoomCount2:
                break
        #Zoom in until we stop changing or we hit machine epsilon
        Ms, errors, trackedInterval, changed = zoomInOnIntervalIter(Ms, errors, trackedInterval)
        if trackedInterval.empty: #Throw out the interval
            return []
        zoomCount += 1
    if shouldStopSubdivision(trackedInterval):
        #Return the interval. Maybe we should return the linear approximation of the root here as well as the interval?
        #Might be better than just taking the midpoint later.
        #Or zoom in assuming no error and take the result of that.
            #If that throws it out, tag that root as being a possible root? It isn't a root of the approx but
            #may be a root of the function.
        return [trackedInterval]
    else:
        #Otherwise, Subdivide
        result = []
        #Get the new intervals and polynomials
        newInts = mySubdivider.subdivideInterval(trackedInterval)
        allMs = [mySubdivider.subdivide(M, e) for M,e in zip(Ms, errors)]
        #Run each interval
        for i in range(len(newInts)):
            result.append(solvePolyRecursive([allM[i][0] for allM in allMs], newInts[i], [allM[i][1] for allM in allMs], trimErrorRelBound, trimErrorAbsBound))
        return combineTouchingIntervals(result)

def solveChebyshevSubdivision(Ms, errors, returnBoundingBoxes = False, polish = False):
    """Finds regions in which any common roots of functions must be

    Parameters
    ----------
    Ms : list of numpy arrays
        The chebyshev approximations of the functions on the interval [-1,1]
    errors : numpy array
        The max error of the chebyshev approximation from the function on the interval
    returnBoundingBoxes : bool (Optional)
        Defaults to False. If True, returns the bounding boxes around each root as well as the roots.
    polish : bool (Optional)
        Defaults to True. Whether or not to polish the roots at the end by zooming all they way back in.
    
    Returns
    -------
    roots : list
        The roots
    boundingBoxes : list of numpy arrays (optional)
        Each element of the list is an interval in which there may be a root.
    """
    #Solve
    originalInterval = TrackedInterval(np.array([[-1.,1.]]*Ms[0].ndim))
    boundingIntervals = solvePolyRecursive(Ms, originalInterval, errors, 1e-16, 1e-16)
        
    #Polish. Testing seems to show no benefit for this.
    if polish:
        newIntervals = []
        for interval in boundingIntervals:
            finalInterval = interval.getFinalInterval()
            newInterval = interval.copy()
            newInterval.interval = finalInterval
            tempMs, tempErrors = transformChebToInterval(Ms, interval.finalAlpha, interval.finalBeta, errors)
            newInterval = solvePolyRecursive(tempMs, newInterval, tempErrors, 1e-16, 1e-16)
            newIntervals += newInterval
        boundingIntervals = newIntervals

    #TODO: Have an options to reRun all the bounding boxes with tight precision after a first run at lower precision.
    #For Example: 
    #boundingBoxes = [solvePolyRecursive(transformedMs, box, errors, <tigher tolerance params>) for box in boundingBoxes]
    #TODO: Don't return the midpoint, return the point this matrix converges to if we don't include any error.
    roots = []
    for interval in boundingIntervals:
        finalInterval = interval.getFinalInterval()
        roots.append((finalInterval[:,1] + finalInterval[:,0]) / 2)
    if returnBoundingBoxes:
        return roots, boundingIntervals
    else:
        return roots