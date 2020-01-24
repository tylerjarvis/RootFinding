import numpy as np
import itertools
from scipy.linalg import qr, solve_triangular, qr_multiply, svd
from yroots.polynomial import Polynomial, MultiCheb, MultiPower
from yroots.utils import row_swap_matrix, MacaulayError, slice_top, mon_combos, \
                              num_mons_full, memoized_all_permutations, mons_ordered, \
                              all_permutations_cheb, ConditioningError
from matplotlib import pyplot as plt

macheps = 2.220446049250313e-16
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

def reduce_macaulay(matrix, cut, max_cond=1e6):
    """Reduces the Macaulay matrix using the Transposed QR method.

    Parameters:
    -----------
    matrix : 2d ndarray
        The Macaulay matrix
    cut : int
        Number of columns of max degree
    max_cond : int or float
        Max condition number for the two condition number checks

    Returns:
    --------
    E : 2d ndarray
        The columns of the reduced Macaulay matrix corresponding to the quotient basis
    Q2 : 2d ndarray
        Matrix giving the quotient basis in terms of the monomial basis. Q2[:,i]
        being the coefficients for the ith basis element
    """

    M = matrix.copy()
    # Check condition number before first QR
    cond_num = np.linalg.cond(M[:,:cut])
    if cond_num > max_cond:
        raise ConditioningError("Condition number of the Macaulay high-degree columns is {}".format(cond_num))

    # QR reduce the highest-degree columns
    Q,M[:,:cut] = qr(M[:,:cut])
    M[:,cut:] = Q.T @ M[:,cut:]
    del Q

    # If the matrix is "tall", compute an orthogonal transformation of the remaining
    # columns, generating a new polynomial basis
    if cut < M.shape[0]:
        Q = qr(M[cut:,cut:].T,pivoting=True)[0]
        M[:cut,cut:] = M[:cut,cut:] @ Q # Apply column transform

    # Compute numerical rank
    s = svd(M, compute_uv=False)
    tol = max(M.shape)*s[0]*macheps
    rank = len(s[s>tol])

    # Check condition number before backsolve
    cond_num = np.linalg.cond(M[:,:cut])
    if cond_num > max_cond:
        raise ConditioningError("Condition number of the Macaulay primary submatrix is {}".format(cond_sum))

    # Return the backsolved columns and coefficient matrix for the quotient basis
    return solve_triangular(M[:cut,:cut],M[:cut,rank:]),Q[:,rank-M.shape[1]:]

def reduce_macaulay_tvb(matrix, cut, max_cond=1e6):

    M = matrix.copy()
    # Check condition number before first QR
    cond_num = np.linalg.cond(M[:,:cut])
    if cond_num > max_cond:
        raise ConditioningError(f"Condition number of the Macaulay high-degree columns is {cond_num}")

    # QR reduce the highest-degree columns
    Q,M[:,:cut] = qr(M[:,:cut])
    M[:,cut:] = Q.T @ M[:,cut:]
    del Q

    # If the matrix is "tall", compute an orthogonal transformation of the remaining
    # columns, generating a new polynomial basis
    if cut < M.shape[0]:
        Q,M[cut:,cut:],P = qr(M[cut:,cut:],pivoting=True)
        del Q
        M[:cut,cut:] = M[:cut,cut:][:,P] # Permute columns

    # Compute numerical rank
    s = svd(M, compute_uv=False)
    tol = max(M.shape)*s[0]*macheps
    rank = len(s[s>tol])

    # Check condition number before backsolve
    cond_num = np.linalg.cond(M[:,:cut])
    if cond_num > max_cond:
        raise ConditioningError(f"Condition number of the Macaulay primary submatrix is {cond_num}")

    return solve_triangular(M[:rank,:rank],M[:rank,rank:]),P

def reduce_macaulay_p(matrix, cut, P, max_cond=1e6):
    M = matrix.copy()
    # Check condition number before first QR
    cond_num = np.linalg.cond(M[:,:cut])
    if cond_num > max_cond:
        raise ConditioningError(f"Condition number of the Macaulay high-degree columns is {cond_num}")

    # QR reduce the highest-degree columns
    Q,M[:,:cut] = qr(M[:,:cut])
    M[:,cut:] = (Q.T @ M[:,cut:])[:,P]
    del Q

    # If the matrix is "tall", compute an orthogonal transformation of the remaining
    # columns, generating a new polynomial basis
    if cut < M.shape[0]:
        Q,M[cut:,cut:] = qr(M[cut:,cut:])
        del Q

    # Compute numerical rank
    s = svd(M, compute_uv=False)
    tol = max(M.shape)*s[0]*macheps
    rank = len(s[s>tol])

    # Check condition number before backsolve
    cond_num = np.linalg.cond(M[:,:cut])
    if cond_num > max_cond:
        raise ConditioningError(f"Condition number of the Macaulay primary submatrix is {cond_num}")

    return solve_triangular(M[:rank,:rank],M[:rank,rank:]),P

def reduce_macaulay_byu(matrix, cut, dim,max_cond=1e6):

    M = matrix.copy()
    # Check condition number before first QR
    cond_num = np.linalg.cond(M[:,:cut])
    if cond_num > max_cond:
        raise ConditioningError(f"Condition number of the Macaulay high-degree columns is {cond_num}")

    # QR reduce the highest-degree columns
    Q,M[:,:cut] = qr(M[:,:cut])
    M[:,cut:] = Q.T @ M[:,cut:]
    del Q

    # If the matrix is "tall", compute an orthogonal transformation of the remaining
    # columns, generating a new polynomial basis
    if cut < M.shape[0] and cut != M.shape[1]-(dim+1):
        Q,M[cut:,cut:-(dim+1)],P = qr(M[cut:,cut:-(dim+1)],pivoting=True)
        M[cut:,-(dim+1):] = Q.T @ M[cut:,-(dim+1):]
        del Q
        M[:cut,cut:-(dim+1)] = M[:cut,cut:-(dim+1)][:,P] # Permute columns

        # update P to be correct length as an array for all columns less than max degree
        P = np.hstack((P,np.arange(len(P),len(P)+dim+1)))

    # Compute numerical rank
    s = svd(M, compute_uv=False)
    tol = max(M.shape)*s[0]*macheps
    rank = len(s[s>tol])

    # Check condition number before backsolve
    cond_num = np.linalg.cond(M[:,:cut])
    if cond_num > max_cond:
        raise ConditioningError(f"Condition number of the Macaulay primary submatrix is {cond_num}")

    return solve_triangular(M[:rank,:rank],M[:rank,rank:]),P

def rrqr_reduceMacaulay(matrix, matrix_terms, cuts, max_cond_num, macaulay_zero_tol, return_perm=False):
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
    max_cond_num : float
        Throws an error if the condition number of the backsolve is more than max_cond_num.
    macaulay_zero_tol : float
        What is considered to be 0 after the reduction. Specifically, rows where every element has
        magnitude less that macaulay_zero_tol are removed.
    return_perm : bool
        If True, also returns the permutation done by the pivoting.
    Returns
    -------
    matrix : numpy array
        The reduced matrix.
    matrix_terms: numpy array
        The resorted matrix_terms.
    perm : numpy array
        The permutation of the rows from the original. Returned only if return_perm is True.
    Raises
    ------
    ConditioningError if the conditioning number of the Macaulay matrix after
    QR is greater than max_cond_num.
    '''
    #controller variables for each part of the matrix
    AD = matrix[:,:cuts[0]]

    BCEF = matrix[:,cuts[0]:]
    # A = matrix[:cuts[0],:cuts[0]]
    B = matrix[:cuts[0],cuts[0]:cuts[1]]
    # C = matrix[:cuts[0],cuts[1]:]
    # D = matrix[cuts[0]:,:cuts[0]]
    E = matrix[cuts[0]:,cuts[0]:cuts[1]]
    F = matrix[cuts[0]:,cuts[1]:]

    #RRQR reduces A and D without pivoting sticking the result in its place.
    Q1,matrix[:,:cuts[0]] = qr(AD)
    #Conditioning check
    cond_num = np.linalg.cond(matrix[:,:cuts[0]])
    if cond_num > max_cond_num:
        raise ConditioningError("Conditioning number of the Macaulay matrix "\
                                + "after first QR is: " + str(cond_num))

    #Multiplying BCEF by Q.T
    BCEF[...] = Q1.T @ BCEF
    del Q1 #Get rid of Q1 for memory purposes.

    #Check to see if E exists
    if cuts[0] != cuts[1] and cuts[0] < matrix.shape[0]:
        #RRQR reduces E sticking the result in it's place.
        Q,E[...],P = qr(E, pivoting = True)

        #Multiplies F by Q.T.
        F[...] = Q.T @ F
        del Q #Get rid of Q for memory purposes.

        #Permute the columns of B
        B[...] = B[:,P]

        #Resorts the matrix_terms.
        matrix_terms[cuts[0]:cuts[1]] = matrix_terms[cuts[0]:cuts[1]][P]

    #use the numerical rank to determine how many rows to keep
    matrix = row_swap_matrix(matrix)[:cuts[1]]
    s = svd(matrix,compute_uv=False)
    tol = max(matrix.shape)*s[0]*macheps
    rank = len(s[s>tol])
    matrix = matrix[:rank]

    #find the condition number of the backsolve
    s = svd(matrix[:,:rank],compute_uv=False)
    cond_num = s[0]/s[-1]
    if cond_num > max_cond_num:
        raise ConditioningError("Conditioning number of backsolving the Macaulay is: " + str(cond_num))

    #backsolve
    height = matrix.shape[0]
    matrix[:,height:] = solve_triangular(matrix[:,:height],matrix[:,height:])
    matrix[:,:height] = np.eye(height)

    if return_perm:
        perm = np.arange(matrix.shape[1])
        perm[cuts[0]:cuts[1]] = perm[cuts[0]:cuts[1]][P]
        return matrix, matrix_terms, perm

    return matrix, matrix_terms
