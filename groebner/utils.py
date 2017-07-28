# A collection of functions used in the F4 and Macaulay solvers
import numpy as np
from scipy.linalg import lu, qr, solve_triangular

def divides(a,b):
    '''
    Returns True if the leading monomial of b divides the leading monomial of a.

    Args:
        a (MultiPower or MultiCheb)
        b (MultiPower or MultiCheb)

    Returns:
        bool

    '''

    # possibly more concise?
    # return all([(i-j) >= 0 for i,j in zip(a.lead_term, b.lead_term)])
    diff = tuple(i-j for i,j in zip(a.lead_term,b.lead_term))
    return all(i >= 0 for i in diff)


def sorted_polys_monomial(polys):
    '''
    Sorts the polynomials by the number of monomials they have, the ones with the least amount first.

    polys (list-like): !!! Is it a list or could it be any iterable?

    '''

    # A list to contain the number of monomials with non zero coefficients.
    num_monomials = []
    for p in polys:
        # Grab all the coefficients in the tensor that are non zero.
        # !!! Why are we only getting index 0?
        n = len(np.where(p.coeff != 0)[0])
        num_monomials.append(n)

    # Generate a sorted index based on num_monomials.
    # TODO: I'm pretty sure there is a numpy function that does this already.
    argsort_list = sorted(range(len(num_monomials)), key=num_monomials.__getitem__)[::]

    # Sort the polynomials according to the index argsort_list.
    sorted_polys = [polys(i) for i in argsort_list]

    return sorted_polys


def argsort(index_list):
    '''
    Returns an argsort list for the index, as well as sorts the list in place

    !!! This could be combined with sorted_polys_monomial to avoid repetitive code.

    '''

    argsort_list = sorted(range(len(index_list)), key=index_list.__getitem__)[::-1]
    return argsort_list, index_list.sort()[::-1]


def calc_r(m, sorted_polys):
    '''
    Finds the r polynomial that has a leading monomial m.
    Returns the polynomial.

    '''

    for p in sorted_polys:
        l = list(p.lead_term)
        # Check to see if l divides m
        if all([i<=j for i,j in zip(l,m)]) and len(l) == len(m):
            # !!! i-j is used in the divide() method, not j-i. Is this a problem?
            c = [j-i for i,j in zip(l,m)]
            if l != m: # Make sure c isn't all 0
                return p.mon_mult(c)


def row_swap_matrix(matrix):
    '''
    rearange the rows of matrix so it starts close to upper traingular

    '''

    rows, columns = np.where(matrix != 0)
    last_i = -1
    lms = list() # !!! What does lms stand for?
    for i,j in zip(rows,columns):
        if i != last_i:
            lms.append(j)
            last_i = i
    # !!! Repetition of previous code.
    argsort_list = sorted(range(len(lms)), key=lms.__getitem__)[::]
    return matrix[argsort_list]
 

def fill_size(bigMatrix, smallMatrix):
    '''
    Fits the small matrix inside of the big matrix and returns it.
    Returns just the coeff matrix as that is all we need in the Groebner create_matrix code.

    '''

    if smallMatrix.shape == bigMatrix.shape:
        return smallMatrix

    matrix = np.zeros_like(bigMatrix) #Even though bigMatrix is all zeros, use this because it makes a copy
    
    slices = list()
    for i in smallMatrix.shape:
        s = slice(0,i)
        slices.append(s)

    matrix[slices] = smallMatrix

    # The coefficient matrix
    return matrix
 

def clean_zeros_from_matrix(matrix, global_accuracy=10e-5):
    '''
    Sets all points in the matrix less than the gloabal accuracy to 0.

    '''

    matrix[np.where(np.abs(matrix) < global_accuracy)] = 0
    return matrix


def fullRank(matrix):
    '''
    Finds the full rank of a matrix.
    Returns independentRows - a list of rows that have full rank, and 
    dependentRows - rows that can be removed without affecting the rank
    Q - The Q matrix used in RRQR reduction in finding the rank

    '''
    
    height = matrix.shape[0]
    Q,R,P = qr(matrix, pivoting = True)
    diagonals = np.diagonal(R) #Go along the diagonals to find the rank
    rank = np.sum(np.abs(diagonals)>global_accuracy)
    numMissing = height - rank
    if numMissing == 0: #Full Rank. All rows independent
        return [i for i in range(height)],[],None
    else:
        #Find the rows we can take out. These are ones that are non-zero in the last rows of Q transpose, as QT*A=R.
        #To find multiple, we find the pivot columns of Q.T
        QMatrix = Q.T[-numMissing:]
        Q1,R1,P1 = qr(QMatrix, pivoting = True)
        independentRows = P1[R1.shape[0]:] #Other Columns
        dependentRows = P1[:R1.shape[0]] #Pivot Columns
        return independentRows,dependentRows,Q
 

def hasFullRank(matrix, global_accuracy=10e-5):
    """
    !!! fullRank() vs hasFullRank(): which one are we using?

    """

    height = matrix.shape[0]
    if height == 0:
        return True
    try:
        Q,R,P = qr(matrix, pivoting = True)
    except:
        raise ValueError("Problem with matrix %s" % str(matrix))

    diagonals = np.diagonal(R) #Go along the diagonals to find the rank
    rank = np.sum(np.abs(diagonals)>global_accuracy)

    if rank == height:
        return True
    else:
        print(rank,height)
        return False
 

def inverse_P(p):
    '''
    Takes in the one dimentional array of column switching.
    Returns the one dimentional array of switching it back.

    '''

    # The elementry matrix that flips the columns of given matrix.
    P = np.eye(len(p))[:,p]
    # This finds the index that equals 1 of each row of P.
    #(This is what we want since we want the index of 1 at each column of P.T)
    return np.where(P==1)[1]


def triangular_solve(self,matrix):
    """
    Reduces the upper block triangular matrix.

    """

    m,n = matrix.shape
    j = 0  # The row index.
    k = 0  # The column index.
    c = [] # It will contain the columns that make an upper triangular matrix.
    d = [] # It will contain the rest of the columns.
    order_c = [] # List to keep track of original index of the columns in c.
    order_d = [] # List to keep track of the original index of the columns in d.

    # Checks if the given matrix is not a square matrix.
    if m != n:
        # Makes sure the indicies are within the matrix.
        while j < m and k < n:
            if matrix[j,k]!= 0:
                c.append(matrix[:,k])
                order_c.append(k)
                # Move to the diagonal if the index is non-zero.
                j+=1
                k+=1
            else:
                d.append(matrix[:,k])
                order_d.append(k)
                # Check the next column in the same row if index is zero.
                k+=1
        # C will be the square matrix that is upper triangular with no zeros on the diagonals.
        C = np.vstack(c).T
        # If d is not empty, add the rest of the columns not checked into the matrix.
        if d:
            D = np.vstack(d).T
            D = np.hstack((D,matrix[:,k:]))
        else:
            D = matrix[:,k:]
        # Append the index of the rest of the columns to the order_d list.
        for i in range(n-k):
            order_d.append(k)
            k+=1

        # Solve for the CX = D
        X = solve_triangular(C,D)

        # Add I to X. [I|X]
        solver = np.hstack((np.eye(X.shape[0]),X))

        # Find the order to reverse the columns back.
        order = self.inverse_P(order_c+order_d)

        # Reverse the columns back.
        solver = solver[:,order]
        # Temporary checker. Plots the non-zero part of the matrix.
        #plt.matshow(~np.isclose(solver,0))

        return solver

    else:
    # The case where the matrix passed in is a square matrix
        return np.eye(m)


