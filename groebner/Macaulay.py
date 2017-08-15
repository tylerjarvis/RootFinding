from operator import itemgetter
import itertools
import numpy as np
import math
from scipy.linalg import lu, qr, solve_triangular, inv, solve, svd
from numpy.linalg import cond
from groebner.polynomial import Polynomial, MultiCheb, MultiPower
from scipy.sparse import csc_matrix, vstack
from groebner.utils import Term, row_swap_matrix, fill_size, clean_zeros_from_matrix, inverse_P, triangular_solve, divides, argsort_dec
import matplotlib.pyplot as plt
from collections import defaultdict

def Macaulay(initial_poly_list, global_accuracy = 1.e-10):
    """
    Macaulay will take a list of polynomials and use them to construct a Macaulay matrix.

    parameters
    --------
    initial_poly_list: A list of polynomials
    global_accuracy: How small we want a number to be before assuming it is zero.
    --------

    Returns
    -----------
    Reduced Macaulay matrix that can be passed into the root finder.
    -----------
    """
    Power = bool
    if all([type(p) == MultiPower for p in initial_poly_list]):
        Power = True
    elif all([type(p) == MultiCheb for p in initial_poly_list]):
        Power = False
    else:
        print([type(p) == MultiPower for p in initial_poly_list])
        raise ValueError('Bad polynomials in list')

    poly_coeff_list = []
    degree = find_degree(initial_poly_list)

    for i in initial_poly_list:
        poly_coeff_list = add_polys(degree, i, poly_coeff_list)

    matrix, matrix_terms = create_matrix(poly_coeff_list)
    
    #rrqr_reduce2 and rrqr_reduce same pretty matched on stability, though I feel like 2 should be better.
    matrix = rrqr_reduce2(matrix, global_accuracy = global_accuracy)
    matrix = clean_zeros_from_matrix(matrix)
    non_zero_rows = np.sum(np.abs(matrix),axis=1) != 0
    matrix = matrix[non_zero_rows,:] #Only keeps the non_zero_polymonials

    matrix = triangular_solve(matrix)
    matrix = clean_zeros_from_matrix(matrix)
    
    
    rows = get_good_rows(matrix, matrix_terms)
    final_polys = get_polys_from_matrix(rows,matrix,matrix_terms,Power)

    return final_polys

def get_polys_from_matrix(rows,matrix,matrix_terms,power):
    '''
    Takes a list of indicies corresponding to the rows of the reduced matrix and
    returns a list of polynomial objects
    '''
    shape = []
    p_list = []
    shape = np.maximum.reduce([term for term in matrix_terms])
    shape += np.ones_like(shape)
    spots = list()
    for dim in range(matrix_terms.shape[1]):
        spots.append(matrix_terms.T[dim])

    # Grabs each polynomial, makes coeff matrix and constructs object
    for i in rows:
        p = matrix[i]
        coeff = np.zeros(shape)
        coeff[spots] = p
        if power:
            poly = MultiPower(coeff)
        else:
            poly = MultiCheb(coeff)

        if poly.lead_term != None:
            p_list.append(poly)
    return p_list

def get_good_rows(matrix, matrix_terms):
    '''
    Gets the rows in a matrix whose leading monomial is not divisible by the leading monomial of any other row.
    Returns a list of rows.
    This function could probably be improved, but for now it is good enough.
    '''
    rowLMs = dict()
    already_looked_at = set()
    #Finds the leading terms of each row.
    for i, j in zip(*np.where(matrix!=0)):
        if i in already_looked_at:
            continue
        else:
            already_looked_at.add(i)
            rowLMs[i] = matrix_terms[j]
    keys= list(rowLMs.keys())
    keys = keys[::-1]
    spot = 0
    #Uses a sieve to find which of the rows to keep.
    while spot != len(keys):
        term1 = rowLMs[keys[spot]]
        toRemove = list()
        for i in range(spot+1, len(keys)):
            term2 = rowLMs[keys[i]]
            if divides(term1,term2):
                toRemove.append(keys[i])
        for i in toRemove:
            keys.remove(i)
        spot += 1
    return keys

def find_degree(poly_list):
    """
    Finds the degree needed for the Macaulay matrix
    -------
    Parameters:
        poly_list: polynomials that will be used to construct the matrix
    -------
    Returns:
        Integer value that is the degree needed.
    -------
    Example:
        For polynomials [P1,P2,P3] with degree [d1,d2,d3] the function returns d1+d2+d3-3+1
    -------
    """
    degree_needed = 0
    for poly in poly_list:
        degree_needed += poly.degree
    return ((degree_needed - len(poly_list)) + 1)

def mon_combos(mon, numLeft, spot = 0):
    '''
    This function finds all the monomials up to a given degree (here numLeft) and returns them.
    mon is a tuple that starts as all 0's and gets changed as needed to get all the monomials.
    numLeft starts as the dimension, but as the code goes is how much can still be added to mon.
    spot is the place in mon we are currently adding things to.
    Returns a list of all the possible monomials.
    '''
    answers = list()
    if len(mon) == spot+1: #We are at the end of mon, no more recursion.
        for i in range(numLeft+1):
            mon[spot] = i
            answers.append(mon.copy())
        return answers
    if numLeft == 0: #Nothing else can be added.
        answers.append(mon.copy())
        return answers
    temp = mon.copy() #Quicker than copying every time inside the loop.
    for i in range(numLeft+1): #Recursively add to mon further down.
        temp[spot] = i
        answers += mon_combos(temp, numLeft-i, spot+1)
    return answers

def add_polys(degree, poly, poly_coeff_list):
    """
    Take each polynomial and adds it to a poly_list
    Then uses monomial multiplication and adds all polynomials with degree less than
        or equal to the total degree needed.
    Returns a list of polynomials.
    """
    poly_coeff_list.append(poly.coeff)
    deg = degree - poly.degree
    dim = poly.dim
    mons = mon_combos(np.zeros(dim, dtype = int),deg)
    mons = mons[1:]
    for i in mons:
        poly_coeff_list.append(poly.mon_mult(i, returnType = 'Matrix'))
    return poly_coeff_list


def sort_matrix_terms(matrix_terms):
    '''Sorts the matrix_terms by term order.
    So the highest terms come first, the lowest ones last/.
    Parameters
    ----------
    matrix_terms : numpy array.
        Each row is one of the terms in the matrix.
    Returns
    -------
    matrix_terms : numpy array
        The sorted matrix_terms.
    '''
    termList = list()
    for term in matrix_terms:
        termList.append(Term(term))
    argsort_list, termList = argsort_dec(termList)
    return matrix_terms[argsort_list]

def coeff_slice(coeff):
    ''' Gets the n-d slices that corespond to the dimenison of a coeff matrix.
    Parameters
    ----------
    coeff : numpy matrix.
        The matrix of interest.
    Returns
    -------
    slices : list
        Each value of the list is a slice of the matrix in some dimension. It is exactly the size of the matrix.
    '''
    slices = list()
    for i in coeff.shape:
        slices.append(slice(0,i))
    return slices

def create_matrix(poly_coeffs):
    ''' Builds a Macaulay matrix.
        
    Parameters
    ----------
    poly_coeffs : list.
        Contains numpy arrays that hold the coefficients of the polynomials to be put in the matrix.
    Returns
    -------
    matrix : 2D numpy array
        The Macaulay matrix.
    '''
    bigShape = np.maximum.reduce([p.shape for p in poly_coeffs])

    #Finds the matrix terms.
    non_zeroSet = set()
    for coeff in poly_coeffs:
        for term in zip(*np.where(coeff != 0)):
            non_zeroSet.add(term)
    matrix_terms = np.zeros_like(bigShape)
    for term in non_zeroSet:
        matrix_terms = np.vstack((matrix_terms,term))
    matrix_terms = matrix_terms[1:]
        
    matrix_terms = sort_matrix_terms(matrix_terms)
    
    #Get the slices needed to pull the matrix_terms from the coeff matrix.
    matrix_term_indexes = list()
    for i in range(len(bigShape)):
        matrix_term_indexes.append(matrix_terms.T[i])
    
    #Adds the poly_coeffs to flat_polys, using added_zeros to make sure every term is in there.
    added_zeros = np.zeros(bigShape)
    flat_polys = list()
    for coeff in poly_coeffs:
        slices = coeff_slice(coeff)
        added_zeros[slices] = coeff
        flat_polys.append(added_zeros[matrix_term_indexes])
        added_zeros[slices] = np.zeros_like(coeff)
        
    #Make the matrix
    matrix = np.vstack(flat_polys[::-1])

    #Sorts the rows of the matrix so it is close to upper triangular.
    matrix = row_swap_matrix(matrix)
    return matrix, matrix_terms

def rrqr_reduce(matrix, clean = False, global_accuracy = 1.e-10):
    '''
    Reduces the matrix into row echelon form, so each row has a unique leading term.
    
    Parameters
    ----------
    matrix : (2D numpy array)
        The matrix of interest.
    clean: bool
        Defaults to False. If True then at certain points in the code all the points in the matrix
        that are close to 0 are set to 0.
    global_accuracy: float
        Defaults to 1.e-10. What is determined to be zero when searching for the pivot columns or setting
        things to zero.

    Returns
    -------
    matrix : (2D numpy array)
        The reduced matrix in row echelon form. It should look like this.
        a - - - - - - -
        0 b - - - - - -
        0 0 0 c - - - -
        0 0 0 0 d - - -
        0 0 0 0 0 0 0 e
    '''
    if matrix.shape[0]==0 or matrix.shape[1]==0:
        return matrix
    height = matrix.shape[0]
    A = matrix[:height,:height] #Get the square submatrix
    B = matrix[:,height:] #The rest of the matrix to the right
    Q,R,P = qr(A, pivoting = True) #rrqr reduce it
    PT = inverse_P(P)
    diagonals = np.diagonal(R) #Go along the diagonals to find the rank
    rank = np.sum(np.abs(diagonals)>global_accuracy)
    if rank == height: #full rank, do qr on it
        Q,R = qr(A)
        A = R #qr reduce A
        B = Q.T.dot(B) #Transform B the same way
    else: #not full rank
        A = R[:,PT] #Switch the columns back
        if clean:
            Q[np.where(abs(Q) < global_accuracy)]=0
        B = Q.T.dot(B) #Multiply B by Q transpose
        if clean:
            B[np.where(abs(B) < global_accuracy)]=0
        #sub1 is the top part of the matrix, we will recursively reduce this
        #sub2 is the bottom part of A, we will set this all to 0
        #sub3 is the bottom part of B, we will recursively reduce this.
        #All submatrices are then put back in the matrix and it is returned.
        sub1 = np.hstack((A[:rank,],B[:rank,])) #Takes the top parts of A and B
        result = rrqr_reduce(sub1) #Reduces it
        A[:rank,] = result[:,:height] #Puts the A part back in A
        B[:rank,] = result[:,height:] #And the B part back in B

        sub2 = A[rank:,]
        zeros = np.zeros_like(sub2)
        A[rank:,] = np.zeros_like(sub2)

        sub3 = B[rank:,]
        B[rank:,] = rrqr_reduce(sub3)

    reduced_matrix = np.hstack((A,B))
    return reduced_matrix

def rrqr_reduce2(matrix, clean = True, global_accuracy = 1.e-10):
    '''
    Reduces the matrix into row echelon form, so each row has a unique leading term.
    Note that it preforms the same function as rrqr_reduce, currently I'm not sure which is better.

    Parameters
    ----------
    matrix : (2D numpy array)
        The matrix of interest.
    clean: bool
        Defaults to True. If True then at certain points in the code all the points in the matrix
        that are close to 0 are set to 0.
    global_accuracy: float
        Defaults to 1.e-10. What is determined to be zero when searching for the pivot columns or setting
        things to zero.

    Returns
    -------
    matrix : (2D numpy array)
        The reduced matrix in row echelon form. It should look like this.
        a - - - - - - -
        0 b - - - - - -
        0 0 0 c - - - -
        0 0 0 0 d - - -
        0 0 0 0 0 0 0 e
    '''
    if matrix.shape[0] <= 1 or matrix.shape[0]==1 or  matrix.shape[1]==0:
        return matrix
    height = matrix.shape[0]
    A = matrix[:height,:height] #Get the square submatrix
    B = matrix[:,height:] #The rest of the matrix to the right
    independentRows, dependentRows, Q = fullRank(A, global_accuracy = global_accuracy)
    nullSpaceSize = len(dependentRows)
    if nullSpaceSize == 0: #A is full rank
        Q,R = qr(matrix)
        return clean_zeros_from_matrix(R)
    else: #A is not full rank
        #sub1 is the independentRows of the matrix, we will recursively reduce this
        #sub2 is the dependentRows of A, we will set this all to 0
        #sub3 is the dependentRows of Q.T@B, we will recursively reduce this.
        #We then return sub1 stacked on top of sub2+sub3
        if clean:
            Q[np.where(abs(Q) < global_accuracy)]=0
        bottom = matrix[dependentRows]
        BCopy = B.copy()
        sub3 = bottom[:,height:]
        sub3 = Q.T[-nullSpaceSize:]@BCopy
        if clean:
            sub3 = clean_zeros_from_matrix(sub3)
        sub3 = rrqr_reduce2(sub3)

        sub1 = matrix[independentRows]
        sub1 = rrqr_reduce2(sub1)

        sub2 = bottom[:,:height]
        sub2[:] = np.zeros_like(sub2)

        reduced_matrix = np.vstack((sub1,np.hstack((sub2,sub3))))
        if clean:
            return clean_zeros_from_matrix(reduced_matrix)
        else:
            return reduced_matrix
    pass


def matrixReduce(matrix, triangular_solve = False, global_accuracy = 1.e-10):
    '''
    Reduces the matrix into row echelon form, so each row has a unique leading term. If triangular_solve is
    True it is reduces to reduced row echelon form, so everything above the leading terms is 0.
    
    Parameters
    ----------
    matrix : (2D numpy array)
        The matrix of interest.
    triangular_solve: bool
        Defaults to False. If True then triangular solve is preformed.
    global_accuracy: float
        Defaults to 1.e-10. What is determined to be zero when searching for the pivot columns.

    Returns
    -------
    matrix : (2D numpy array)
        The reduced matrix. It should look like this if triangluar_solve is False.
        a - - - - - - -
        0 b - - - - - -
        0 0 0 c - - - -
        0 0 0 0 d - - -
        0 0 0 0 0 0 0 e

        If triangular solve is True it will look like this.
        a 0 - 0 0 - - 0
        0 b - 0 0 - - 0
        0 0 0 c 0 - - 0
        0 0 0 0 d - - 0
        0 0 0 0 0 0 0 e

    '''
    independentRows,dependentRows,Q = fullRank(matrix, global_accuracy = global_accuracy)
    matrix = matrix[independentRows]
    
    pivotColumnMatrix = findPivotColumns(matrix, global_accuracy = global_accuracy)    
    pivotColumns = list(np.where(pivotColumnMatrix == 1)[1])
    otherColumns = list()
    for i in range(matrix.shape[1]):
        if i not in pivotColumns:
            otherColumns.append(i)
        
    matrix = matrix[:,pivotColumns + otherColumns]
    
    Q,R = qr(matrix)
    if triangular_solve:
        R = clean_zeros_from_matrix(R)
        X = solve_triangular(R[:,:R.shape[0]],R[:,R.shape[0]:])
        reduced = np.hstack((np.eye(X.shape[0]),X))
    
    matrix = np.empty_like(reduced)
    matrix[:,pivotColumns + otherColumns] = reduced
    
    return matrix

def fullRank(matrix, global_accuracy = 1.e-10):
    '''
    Uses rank revealing QR to determine which rows of the given matrix are
    linearly independent and which ones are linearly dependent. (This
    function needs a name change).

    Parameters
    ----------
    matrix : (2D numpy array)
        The matrix of interest.
    global_accuracy: float
        Defaults to 1.e-10. What is determined to be zero when searching for the pivot columns.

    Returns
    -------
    independentRows : (list)
        The indexes of the rows that are linearly independent
    dependentRows : (list)
        The indexes of the rows that can be removed without affecting the rank
        (which are the linearly dependent rows).
    Q : (2D numpy array)
        The Q matrix used in RRQR reduction in finding the rank.
    '''
    height = matrix.shape[0]
    Q,R,P = qr(matrix, pivoting = True)
    diagonals = np.diagonal(R) #Go along the diagonals to find the rank
    rank = np.sum(np.abs(diagonals)>global_accuracy)
    numMissing = height - rank
    if numMissing == 0: # Full Rank. All rows independent
        return [i for i in range(height)],[],None
    else:
        # Find the rows we can take out. These are ones that are non-zero in
        # the last rows of Q transpose, since QT*A=R.
        # To find multiple, we find the pivot columns of Q.T
        QMatrix = Q.T[-numMissing:]
        Q1,R1,P1 = qr(QMatrix, pivoting = True)
        independentRows = P1[R1.shape[0]:] #Other Columns
        dependentRows = P1[:R1.shape[0]] #Pivot Columns
        return independentRows,dependentRows,Q

def findPivotColumns(matrix, global_accuracy = 1.e-10):
    ''' Finds the pivot columns of a matrix.
    Uses rank revealing QR to determine which columns in a matrix are the pivot columns. This is done using
    this function recursively.

    Parameters
    ----------
    matrix : (2D numpy array)
        The matrix of interest.
    global_accuracy: float
        Defaults to 1.e-10. What is determined to be zero when searching for the pivot columns.

    Returns
    -------
    matrix : (2D numpy array)
        A matrix of ones and zeros. Each row will have exactly one 1 in it, which will be a pivot column
        in the matrix. For example, a 5x8 matrix with pivot columns 1,2,4,5,8 will look like this
        1 0 0 0 0 0 0 0
        0 1 0 0 0 0 0 0
        0 0 0 1 0 0 0 0
        0 0 0 0 1 0 0 0
        0 0 0 0 0 0 0 1
    '''

    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return matrix
    elif matrix.shape[1] == 1:
        column = np.zeros_like(matrix)
        if np.sum(np.abs(matrix)) < global_accuracy:
            return column
        else:
            column[0] = 1
            return column
    
    height = matrix.shape[0]
    A = matrix[:height,:height] #Get the square submatrix
    B = matrix[:,height:] #The rest of the matrix to the right
    independentRows, dependentRows, Q = fullRank(A, global_accuracy = global_accuracy)
    nullSpaceSize = len(dependentRows)
    if nullSpaceSize == 0: #A is full rank
        #The columns of A are all pivot columns
        return np.hstack((np.eye(height),np.zeros_like(B)))
    else: #A is not full rank
        #sub1 is the independentRows of the matrix, we will recursively reduce this
        #sub2 is the dependentRows of A, we will set this all to 0
        #sub3 is the dependentRows of Q.T@B, we will recursively reduce this.
        #We then return sub1 stacked on top of sub2+sub3
        bottom = matrix[dependentRows]
        BCopy = B.copy()
        sub3 = bottom[:,height:]
        sub3 = Q.T[-nullSpaceSize:]@BCopy #I think this line can be taked out for this code.
        sub3 = findPivotColumns(sub3)

        sub1 = matrix[independentRows]
        sub1 = findPivotColumns(sub1)

        sub2 = bottom[:,:height]
        sub2[:] = np.zeros_like(sub2)

        pivot_columns = np.vstack((sub1,np.hstack((sub2,sub3))))
        return pivot_columns
    pass
