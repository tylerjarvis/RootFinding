import numpy as np
import itertools
from scipy.linalg import qr, solve_triangular, qr_multiply
from yroots.polynomial import Polynomial, MultiCheb, MultiPower
from yroots.utils import row_swap_matrix, MacaulayError, slice_top, mon_combos, \
                              num_mons_full, memoized_all_permutations, mons_ordered, \
                              all_permutations_cheb
from matplotlib import pyplot as plt
from scipy.linalg import svd

def add_polys(degree, poly, poly_coeff_list):
    """Adds polynomials to a Macaulay Matrix.

    This function is called on one polynomial and adds all monomial multiples of
     it to the matrix.

    Parameters
    ----------
    degree : int
        The degree of the Macaulay Matrix
    poly : Polynomial
        One of the polynomials used to make the matrix.
    poly_coeff_list : list
        A list of all the current polynomials in the matrix.
    Returns
    -------
    poly_coeff_list : list
        The original list of polynomials in the matrix with the new monomial
        multiplications of poly added.
    """

    poly_coeff_list.append(poly.coeff)
    deg = degree - poly.degree
    dim = poly.dim

    mons = mon_combos([0]*dim,deg)

    for mon in mons[1:]: #skips the first all 0 mon
        poly_coeff_list.append(poly.mon_mult(mon, returnType = 'Matrix'))
    return poly_coeff_list

def find_degree(poly_list, verbose=False):
    '''Finds the appropriate degree for the Macaulay Matrix.

    Parameters
    --------
    poly_list: list
        The polynomials used to construct the matrix.
    verbose : bool
        If True prints the degree
    Returns
    -----------
    find_degree : int
        The degree of the Macaulay Matrix.

    '''
    if verbose:
        print('Degree of Macaulay Matrix:', sum(poly.degree for poly in poly_list) - len(poly_list) + 1)
    return sum(poly.degree for poly in poly_list) - len(poly_list) + 1

def rrqr_reduceMacaulay(matrix, matrix_terms, cuts, accuracy = 1.e-10):
    ''' Reduces a Macaulay matrix, BYU style.

    The matrix is split into the shape
    A B C
    D E F
    Where A is square and contains all the highest terms, and C contains all the x,y,z etc. terms. The lengths
    are determined by the matrix_shape_stuff tuple. First A and D are reduced using rrqr without pivoting, and then the rest of
    the matrix is multiplied by Q.T to change it accordingly. Then E is reduced by rrqr with pivoting, the rows of B are shifted
    accordingly, and F is multipled by Q.T to change it accordingly. This is all done in place to save memory.

    Parameters
    ----------
    matrix : numpy array.
        The Macaulay matrix, sorted in BYU style.
    matrix_terms: numpy array
        Each row of the array contains a term in the matrix. The i'th row corresponds to
        the i'th column in the matrix.
    cuts : tuple
        When the matrix is reduced it is split into 3 parts with restricted pivoting. These numbers indicate
        where those cuts happen.
    accuracy : float
        Throws an error if the condition number of the backsolve is more than 1/accuracy.
    Returns
    -------
    matrix : numpy array
        The reduced matrix.
    matrix_terms: numpy array
        The resorted matrix_terms.
    '''
    #RRQR reduces A and D without pivoting sticking the result in it's place.
    Q1,matrix[:,:cuts[0]] = qr(matrix[:,:cuts[0]])

    #Multiplying BCEF by Q.T
    matrix[:,cuts[0]:] = Q1.T@matrix[:,cuts[0]:]
    del Q1 #Get rid of Q1 for memory purposes.

    #RRQR reduces E sticking the result in it's place.
    Q,matrix[cuts[0]:,cuts[0]:cuts[1]],P = qr(matrix[cuts[0]:,cuts[0]:cuts[1]], pivoting = True)

    #Multiplies F by Q.T.
    matrix[cuts[0]:,cuts[1]:] = Q.T@matrix[cuts[0]:,cuts[1]:]
    del Q #Get rid of Q for memory purposes.

    #Permute the columns of B
    matrix[:cuts[0],cuts[0]:cuts[1]] = matrix[:cuts[0],cuts[0]:cuts[1]][:,P]

    #Resorts the matrix_terms.
    matrix_terms[cuts[0]:cuts[1]] = matrix_terms[cuts[0]:cuts[1]][P]

    #eliminate zero rows from the bottom of the matrix.
    matrix = row_swap_matrix(matrix)
    for row in matrix[::-1]:
        if np.allclose(row, 0,atol=accuracy):
            matrix = matrix[:-1]
        else:
            break

    #set very small values in the matrix to zero before backsolving
    matrix[np.isclose(matrix, 0, atol=accuracy)] = 0

    #SVD conditioning check
    D = np.linalg.svd(matrix[:,:matrix.shape[0]], compute_uv=False)
    if D[0]/D[-1] > 1/accuracy:
        return -1, -1

    #backsolve
    height = matrix.shape[0]
    matrix[:,height:] = solve_triangular(matrix[:,:height],matrix[:,height:])
    matrix[:,:height] = np.eye(height)

    return matrix, matrix_terms

def rrqr_reduceMacaulay2(matrix, matrix_terms, cuts, accuracy = 1.e-10):
    ''' Reduces a Macaulay matrix, BYU style

    This function does the same thing as rrqr_reduceMacaulay but uses
    qr_multiply instead of qr and a multiplication
    to make the function faster and more memory efficient.

    This function only works properly if the bottom left (D) part of the matrix is zero

    Parameters
    ----------
    matrix : numpy array.
        The Macaulay matrix, sorted in BYU style.
    matrix_terms: numpy array
        Each row of the array contains a term in the matrix. The i'th row corresponds to
        the i'th column in the matrix.
    cuts : tuple
        When the matrix is reduced it is split into 3 parts with restricted pivoting. These numbers indicate
        where those cuts happen.
    accuracy : float
        What is determined to be 0.
    Returns
    -------
    matrix : numpy array
        The reduced matrix.
    matrix_terms: numpy array
        The resorted matrix_terms.
    '''
    #RRQR reduces A and multiplies BC.T by Q
    product1,matrix[:cuts[0],:cuts[0]] = qr_multiply(matrix[:,:cuts[0]], matrix[:,cuts[0]:].T, mode = 'right')
    #BC is now Q.T @ BC
    matrix[:cuts[0],cuts[0]:] = product1.T
    del product1 #remove ]for memory purposes

    #check if there are zeros along the diagonal of R1
    if any(np.isclose(np.diag(matrix[:,:cuts[0]]),0, atol=accuracy)):
        raise MacaulayError("R1 IS NOT FULL RANK")

    #set small values to zero before backsolving
    matrix[np.isclose(matrix, 0, atol=accuracy)] = 0
    #backsolve top of matrix (solve triangular on B and C)
    matrix[:cuts[0],cuts[0]:] = solve_triangular(matrix[:cuts[0],:cuts[0]],matrix[:cuts[0],cuts[0]:])
    matrix[:cuts[0],:cuts[0]] = np.eye(cuts[0]) #A is now the identity after backsolving
    #E and F                            D                           BC
    matrix[cuts[0]:,cuts[0]:] -= (matrix[cuts[0]:,:cuts[0]])@matrix[:cuts[0],cuts[0]:] #?

    #QRP on E, multiply that onto F
    product2,R,P = qr_multiply(matrix[cuts[0]:,cuts[0]:cuts[1]], matrix[cuts[0]:,cuts[1]:].T, mode = 'right', pivoting = True)
    #get rid of zero rows
    matrix = matrix[:R.shape[0]+cuts[0]]
    #set D to zero
    matrix[cuts[0]:,:cuts[0]] = np.zeros_like(matrix[cuts[0]:,:cuts[0]])
    #fill E in with R
    matrix[cuts[0]:,cuts[0]:cuts[0]+R.shape[1]] = R
    #fill F in with product2.T
    matrix[cuts[0]:,cuts[0]+R.shape[1]:] = product2.T
    del product2,R

    #raise error if E is not full rank
    if any(np.isclose(np.diag(matrix),0, atol=accuracy)):
        raise MacaulayError("FULL MATRIX IS NOT FULL RANK")

    #Permute the columns of B, since E already got permuted implicitly
    matrix[:cuts[0],cuts[0]:cuts[1]] = matrix[:cuts[0],cuts[0]:cuts[1]][:,P]
    matrix_terms[cuts[0]:cuts[1]] = matrix_terms[cuts[0]:cuts[1]][P]
    del P

    #set small values in the matrix to zero now, after the QR reduction
    matrix[np.isclose(matrix, 0, atol=accuracy)] = 0
    #eliminate zero rows from the bottom of the matrix.
    matrix = row_swap_matrix(matrix)
    for row in matrix[::-1]:
        if np.allclose(row, 0,atol=accuracy):
            matrix = matrix[:-1]
        else:
            break

    #backsolve
    height = matrix.shape[0]
    matrix[:,height:] = solve_triangular(matrix[:,:height],matrix[:,height:])
    matrix[:,:height] = np.eye(height)

    return matrix, matrix_terms
