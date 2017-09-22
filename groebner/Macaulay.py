import numpy as np
from scipy.linalg import qr, solve_triangular
from groebner.polynomial import MultiCheb, MultiPower
from groebner.utils import Term, clean_zeros_from_matrix, triangular_solve, divides, slice_top, mon_combos, mon_combosHighest
import groebner.utils as utils
import matplotlib.pyplot as plt
import sympy

def Macaulay(initial_poly_list, global_accuracy = 1.e-10):
    """
    Accepts a list of polynomials and use them to construct a Macaulay matrix.

    parameters
    --------
    initial_poly_list: list
        Polynomials for Macaulay construction.
    global_accuracy : float
        Round-off parameter: values within global_accuracy of zero are rounded to zero. Defaults to 1e-10.

    Returns
    -------
    final_polys : list
        Reduced Macaulay matrix that can be passed into the root finder.
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
    matrix = utils.rrqr_reduce2(matrix, global_accuracy = global_accuracy)
    matrix = clean_zeros_from_matrix(matrix)
    non_zero_rows = np.sum(np.abs(matrix),axis=1) != 0
    matrix = matrix[non_zero_rows,:] #Only keeps the non_zero_polymonials
    
    matrix = triangular_solve(matrix)
    matrix = clean_zeros_from_matrix(matrix)
    
    #The other reduction option. I thought it would be really stable but seems to be the worst of the three.
    #matrix = matrixReduce(matrix, triangular_solve = True, global_accuracy = global_accuracy)
    
    rows = get_good_rows(matrix, matrix_terms)
    final_polys = get_polys_from_matrix(matrix, matrix_terms, rows, Power)

    return final_polys

def get_polys_from_matrix(matrix, matrix_terms ,rows, power):
    '''Creates polynomial objects from the specified rows of the given matrix.

    Parameters
    ----------
    matrix : (M,N) ndarray
        The matrix with rows corresponding to polynomials, columns corresponding
        to monomials, and entries corresponding to coefficients.
    matrix_terms : array-like
        The column labels for matrix in order. Contains Term objects.
    rows : iterable
        The rows for which to create polynomial objects. Contains integers.
    power : bool
        If true, the polynomials returned will be MultiPower objects.
        Otherwise, they will be MultiCheb.
        
    Returns
    -------
    poly_list : list
        Polynomial objects corresponding to the specified rows.
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
    
    Parameters
    ----------
    matrix : (M,N) ndarray
        Input matrix.
    matrix_terms : array-like
        The column labels for matrix in order. Contains Term objects.
        
    Returns
    -------
    keys : list
        Rows indicies satisfying the divisibility condition.
        
    Notes
    -----
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

    Parameters
    ----------
    poly_list : list
        Polynomials that will be used to construct the matrix.
    
    Returns
    -------
    int
        Needed degree for Macaulay matrix.
    
    Notes
    -------
        For polynomials [P1,P2,P3] with degree [d1,d2,d3] the function returns d1+d2+d3-3+1
    """
    degree_needed = 0
    for poly in poly_list:
        degree_needed += poly.degree
    return ((degree_needed - len(poly_list)) + 1)

def add_polys(degree, poly, poly_coeff_list):
    """Adds polynomials to a Macaulay Matrix.
    
    This function is called on one polynomial and adds all monomial multiples of it to the matrix.
    
    Parameters
    ----------
    degree : int
        The degree of the TVB Matrix
    poly : Polynomial
        One of the polynomials used to make the matrix. 
    poly_coeff_list : list
        A list of all the current polynomials in the matrix.
    Returns
    -------
    poly_coeff_list : list
        The original list of polynomials in the matrix with the new monomial multiplications of poly added.
    """
    poly_coeff_list.append(poly.coeff)
    deg = degree - poly.degree
    dim = poly.dim
    mons = mon_combos([0]*dim,deg)
    mons = mons[1:]
    for i in mons:
        poly_coeff_list.append(poly.mon_mult(i, returnType = 'Matrix'))
    return poly_coeff_list


def sort_matrix_terms(matrix_terms):
    '''Sorts the matrix_terms by term order.
    So the highest terms come first, the lowest ones last.
    
    Parameters
    ----------
    matrix_terms : ndarray
        Each row is one of the terms in the matrix.
    
    Returns
    -------
    matrix_terms : ndarray
        The sorted matrix_terms.
    '''
    termList = list()
    for term in matrix_terms:
        termList.append(Term(term))
    argsort_list = np.argsort(termList)[::-1]
    return matrix_terms[argsort_list]

def create_matrix(poly_coeffs):
    ''' Builds a Macaulay matrix.

    Parameters
    ----------
    poly_coeffs : list
        Contains numpy arrays that hold the coefficients of the polynomials to be put in the matrix.
    Returns
    -------
    matrix : (M,N) ndarray
        The Macaulay matrix.
    '''
    bigShape = np.maximum.reduce([p.shape for p in poly_coeffs])

    #Finds the matrix terms.
    non_zeroSet = set()
    for coeff in poly_coeffs:
        for term in zip(*np.where(coeff != 0)):
            non_zeroSet.add(term)
    matrix_terms = np.array(non_zeroSet.pop())
    for term in non_zeroSet:
        matrix_terms = np.vstack((matrix_terms,term))

    matrix_terms = sort_matrix_terms(matrix_terms)

    #Get the slices needed to pull the matrix_terms from the coeff matrix.
    matrix_term_indexes = list()
    for i in range(len(bigShape)):
        matrix_term_indexes.append(matrix_terms.T[i])

    #Adds the poly_coeffs to flat_polys, using added_zeros to make sure every term is in there.
    added_zeros = np.zeros(bigShape)
    flat_polys = list()
    for coeff in poly_coeffs:
        slices = slice_top(coeff)
        added_zeros[slices] = coeff
        flat_polys.append(added_zeros[matrix_term_indexes])
        added_zeros[slices] = np.zeros_like(coeff)

    #Make the matrix
    matrix = np.vstack(flat_polys[::-1])

    #Sorts the rows of the matrix so it is close to upper triangular.
    matrix = utils.row_swap_matrix(matrix)
    return matrix, matrix_terms

def matrixReduce(matrix, triangular_solve = False, global_accuracy = 1.e-10):
    '''
    Reduces the matrix into row echelon form, so each row has a unique leading term. If triangular_solve is
    True it is reduces to reduced row echelon form, so everything above the leading terms is 0.

    Parameters
    ----------
    matrix : (M,N) ndarray
        The matrix of interest.
    triangular_solve : bool
        Defaults to False. If True then triangular solve is preformed.
    global_accuracy : float
        Defaults to 1.e-10. What is determined to be zero when searching for the pivot columns.

    Returns
    -------
    matrix : (M,N) ndarray
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
    independentRows,dependentRows,Q = utils.row_linear_dependencies(matrix, accuracy = global_accuracy)
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
    else:
        reduced = R

    matrix = reduced[:,utils.inverse_P(pivotColumns + otherColumns)]

    matrix = clean_zeros_from_matrix(matrix)
    return matrix

def findPivotColumns(matrix, global_accuracy = 1.e-10):
    ''' Finds the pivot columns of a matrix.
    Uses rank revealing QR to determine which columns in a matrix are the pivot columns. This is done using
    this function recursively.

    Parameters
    ----------
    matrix : (M,N) ndarray
        The matrix of interest.
    global_accuracy : float
        Defaults to 1.e-10. What is determined to be zero when searching for the pivot columns.

    Returns
    -------
    matrix : (M,N) ndarray
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
    independentRows, dependentRows, Q = utils.row_linear_dependencies(A, accuracy = global_accuracy)
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

