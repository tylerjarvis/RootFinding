import numpy as np
import itertools
from groebner.utils import clean_zeros_from_matrix, get_var_list, memoize
from groebner.TelenVanBarel import create_matrix, add_polys
from groebner.root_finder import _random_poly
from scipy.linalg import solve_triangular, eig
from scipy.stats import mode
from scipy.linalg import qr

def division_cheb(polys):
    '''Calculates the common zeros of polynomials using a division matrix.
    
    Parameters
    --------
    polys: MultiCheb Polynomials
        The polynomials for which the common roots are found.

    Returns
    -----------
    zeros : list
        The common zeros of the polynomials. Each list element is a numpy array of complex entries
        that contains the coordinates in each dimension of the zero.
    '''
    dim = polys[0].dim
    
    #This first section creates the Macaulay Matrix with the monomials that only have ys first.
    matrix_degree = np.sum(poly.degree for poly in polys) - len(polys) + 1
    poly_coeff_list = []
    for i in polys:
        poly_coeff_list = add_polys(matrix_degree, i, poly_coeff_list)
    matrix, matrix_terms, matrix_shape_stuff = create_matrix(poly_coeff_list, matrix_degree, dim)
    #perm is a permutation to reorder the matrix columns to put the ys first.
    #cut is the spot in the matrix that divides between the y terms and the other terms.
    perm, cut = matrix_term_perm(matrix_terms)
    matrix = matrix[:,perm]
    matrix_terms = matrix_terms[perm]
    #Reduces the Macaulay matrix like normal.
    A,B = matrix[:,:cut], matrix[:,cut:]
    Q,A,P = qr(A, pivoting=True)
    matrix_terms[:cut] = matrix_terms[:cut][P]
    B = Q.T@B
    C,D = B[:cut], B[cut:]
    Q0,D,P0 = qr(D, pivoting=True)
    C = C[:,P0]
    matrix_terms[cut:] = matrix_terms[cut:][P0]
    matrix[:,:cut] = A
    matrix[:cut,cut:] = C
    matrix[cut:,cut:] = D
    rows,columns = matrix.shape
    
    VB = matrix_terms[matrix.shape[0]:]
    matrix = np.hstack((np.eye(rows),solve_triangular(matrix[:,:rows],matrix[:,rows:])))

    #Builds the inverse matrix. The terms are the vector basis as well as y^k/x terms for all k. Reducing
    #this matrix allows the y^k/x terms to be reduced back into the vector basis.
    inverses = np.vstack((-np.ones(cut), np.arange(cut))).T
    inv_matrix_terms = np.vstack((inverses, VB))
    inv_matrix = np.zeros([len(inverses),len(inv_matrix_terms)])

    #A bunch of different dictionaries are used below for speed purposes and to prevent repeat calculations.
    
    #A dictionary of term in inv_matrix_terms to their spot in inv_matrix_terms.
    inv_spot_dict = dict()
    spot = 0
    for term in inv_matrix_terms:
        inv_spot_dict[tuple(term)] = spot
        spot+=1

    #A dictionary of terms on the diagonal to their reduction in the vector basis.
    diag_reduction_dict = dict()
    for i in range(matrix.shape[0]):
        term = matrix_terms[i]
        diag_reduction_dict[tuple(term)] = matrix[i][-len(VB):]
    
    #A dictionary of terms to the terms in their quotient when divided by x.
    divisor_terms_dict = dict()
    for term in matrix_terms:
        divisor_terms_dict[tuple(term)] = get_divisor_terms(term)
    
    #A dictionary of terms to their quotient when divided by x.
    term_divide_dict = dict()
    for term in matrix_terms[-len(VB):]:
        term_divide_dict[tuple(term)] = divide_term_x(term, inv_matrix_terms, inv_spot_dict, diag_reduction_dict,
                                                      len(VB), divisor_terms_dict)

    #Builds the inv_matrix by dividing the rows of matrix by x.
    for i in range(cut):
        spot = matrix_terms[i]
        y = spot[1]
        inv_matrix[i] = divide_row_x(matrix[i][-len(VB):], matrix_terms[-len(VB):], term_divide_dict, len(inv_matrix_terms))
        inv_matrix[i][y] += 1

    #Reduces the inv_matrix to solve for the y^k/x terms in the vector basis.
    Q,R = qr(inv_matrix)
    inv_solutions = np.hstack((np.eye(R.shape[0]),solve_triangular(R[:,:R.shape[0]], R[:,R.shape[0]:])))

    #A dictionary of term in the vector basis to their spot in the vector basis.
    VB_spot_dict = dict()
    spot = 0
    for row in VB:
        VB_spot_dict[tuple(row)] = spot
        spot+=1

    #A dictionary of terms of type y^k/x to their reduction in the vector basis.
    inv_reduction_dict = dict()
    for i in range(len(inv_solutions)):
        inv_reduction_dict[tuple(inv_matrix_terms[i])] = inv_solutions[i][len(inv_solutions):]

    #Builds the division matrix and finds the eigenvalues and eigenvectors.
    division_matrix = build_division_matrix(VB, VB_spot_dict, diag_reduction_dict, inv_reduction_dict, divisor_terms_dict)
    vals, vecs = eig(division_matrix.T)

    #Finds two spots in the vector basis with the same x value and y values of 0 and 1 so we can use
    #those spots on the eigenvectors to calculate the y values of the zeros.
    for spot in range(len(VB)):
        term = VB[spot]
        if term[1] == 1:
            if tuple(term - [0,1]) in VB_spot_dict:
                ys = [spot, VB_spot_dict[tuple(term - [0,1])]]
                break
    
    #Calculates the zeros, the x values from the eigenvalues and the y values from the eigenvectors.
    zeroY = vecs[ys[0]]/vecs[ys[1]]
    zeroX = 1/vals
    zeros = list()
    for i in range(len(zeroX)):
        zeros.append(np.array([zeroX[i], zeroY[i]]))
    
    return zeros
    
def matrix_term_perm(matrix_terms):
    '''Finds the needed column permutation to have all the y terms first in the Macaulay Matrix.
    
    Parameters
    --------
    matrix_terms: numpy array
        The current order of the terms in the matrix.

    Returns
    -----------
    perm : numpy array
        The desired column permutation.
    cut : The number of y terms in the matrix. This is where the matrix is cut in the matrix reduction,
          pivoting past this point is not allowed.
    '''    
    boundary = np.where(matrix_terms[:,0] == 0)[0]
    bSet = set(boundary)
    other = np.arange(len(matrix_terms))
    mask = [num not in bSet for num in other]
    other = other[mask]
    return np.hstack((boundary,other)), len(boundary)

def divide_row_x(coeffs, terms, term_divide_dict, length):
    """Divides a row of the matrix by x.
    
    Parameters
    ----------
    coeffs : numpy array.
        The numerical values of the terms we want to divide by x.
    terms: numpy array
        The terms corresponding to the numerical values.
    term_divide_dict: dictionary
        Maps each term as a tuple to a numpy array representing that term divided by x.
    length : int
        The length of the rows in the inv_matrix.
    Returns
    -------
    new_row : numpy array
        The row we get in the inverse_matrix by dividing the first row by x.
    """    
    new_row = np.zeros(length)
    for i in range(len(coeffs)):
        new_row+=coeffs[i]*term_divide_dict[tuple(terms[i])]
    return new_row

def divide_term_x(term, inv_matrix_terms, inv_spot_dict, diag_reduction_dict, VB_size, divisor_terms_dict):
    """Divides a term of the matrix by x.
    
    Parameters
    ----------
    term: numpy array
        The term to divide.
    inv_matrix_terms: numpy array
        The terms in the inverse matrix.
    inv_spot_dict : dictionary
        A dictionary of term in inv_matrix_terms to their spot in inv_matrix_terms.
    diag_reduction_dict : dictionary
        A dictionary of terms on the diagonal to their reduction in the vector basis.
    VB_size : int
        The number of elements in the vector basis.
    divisor_terms_dict : dictionary
        A dictionary of terms to the terms in their dividend when divided by x.
    
    Returns
    -------
    row : numpy array
        The row we get in the inverse_matrix by dividing the term by x.
    """
    row = np.zeros(len(inv_matrix_terms))
    divisor_terms = divisor_terms_dict[tuple(term)]
    parity = 1
    for spot in divisor_terms[:-1]:
        if tuple(spot) in inv_spot_dict:
            row[inv_spot_dict[tuple(spot)]] += parity*2
        else:
            row[-VB_size:] -= 2*parity*diag_reduction_dict[tuple(spot)]
        parity*=-1
    spot = divisor_terms[-1]
    if tuple(spot) in inv_spot_dict:
        row[inv_spot_dict[tuple(spot)]] += parity
    else:
        row[-VB_size:] -= parity*diag_reduction_dict[tuple(spot)]
    return row

def get_divisor_terms(term):
    """Finds the terms that will be present when dividing a given term by x.
    
    Parameters
    ----------
    term: numpy array
        The term to divide.
    
    Returns
    -------
    terms : numpy array
        Each row is a term that will be in the quotient.
    """
    initial = term - [1,0]
    height = term[0]//2+1
    terms = np.vstack([initial.copy() for i in range(height)])
    dec = 0
    for i in range(terms.shape[0]):
        terms[i][0]-=2*dec
        dec+=1
    return terms

def build_division_matrix(VB, VB_spot_dict, diag_reduction_dict, inv_reduction_dict, divisor_terms_dict):
    """Builds the division matrix.
    
    Parameters
    ----------
    VB: numpy array
        The vector basis.
    VB_spot_dict : dictionary
        A dictionary of term in the vector basis to their spot in the vector basis.
    diag_reduction_dict : dictionary
        A dictionary of terms on the diagonal to their reduction in the vector basis.
    inv_reduction_dict : dictionary
        A dictionary of terms of type y^k/x to their reduction in the vector basis.
    divisor_terms_dict : dictionary
        A dictionary of terms to the terms in their dividend when divided by x.
    
    Returns
    -------
    row : numpy array
        The row we get in the inverse_matrix by dividing the term by x.
    """

    div_matrix = np.zeros((len(VB), len(VB)))
    for i in range(len(VB)):
        term = VB[i]
        terms = divisor_terms_dict[tuple(term)]
        parity = 1
        for spot in terms[:-1]:
            if tuple(spot) in VB_spot_dict:
                div_matrix[VB_spot_dict[tuple(spot)]][i]+=2*parity
            else:
                div_matrix[:,i] -= 2*parity*diag_reduction_dict[tuple(spot)][-len(VB):]
            parity *= -1
        spot = terms[-1]
        if tuple(spot) in diag_reduction_dict:
            div_matrix[:,i] -= parity*diag_reduction_dict[tuple(spot)][-len(VB):]
        else:
            div_matrix[:,i] -= parity*inv_reduction_dict[tuple(spot)]
    return div_matrix
