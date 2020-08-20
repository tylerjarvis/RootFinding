import numpy as np
import itertools
from scipy.linalg import qr, solve_triangular, qr_multiply, svd
from yroots.polynomial import Polynomial, MultiCheb, MultiPower
from yroots.utils import row_swap_matrix, MacaulayError, slice_top, mon_combos, \
                              num_mons_full, memoized_all_permutations, mons_ordered, \
                              all_permutations_cheb, ConditioningError, TooManyRoots
from matplotlib import pyplot as plt
from warnings import warn

macheps = 2.220446049250313e-16

def plot_scree(s,tol):
    plt.semilogy(s,marker='.')
    plt.plot(np.ones(len(s))*tol)
    plt.show()

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

def reduce_macaulay_qrt(M, cut, bezout_bound, max_cond=1e6):
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
    # Compute numerical rank
    s = svd(M, compute_uv=False)
    tol = max(M.shape)*s[0]*macheps
    rank = len(s[s>tol])
    # Check if numerical rank doesn't match bezout bound
    bezout_rank = M.shape[1]-bezout_bound
    if rank < bezout_rank:
        warn("Rank of Macaulay Matrix does not match the Bezout bound. Expected rank {}, found rank {}. System potentially has infinitely many solutions.".format(bezout_rank,rank))
    elif rank > bezout_rank:
        warn('Rank of Macaulay Matrix does not match the Bezout bound. Expected rank {}, found rank {}.'.format(bezout_rank,rank))

    # Check condition number before first QR
    cond_num = np.linalg.cond(M[:,:cut])
    if cond_num > max_cond:
        return None, "Condition number of the Macaulay high-degree columns is {}".format(cond_num)

    # QR reduce the highest-degree columns
    Q,M[:,:cut] = qr(M[:,:cut])
    M[:,cut:] = Q.T @ M[:,cut:]
    Q = None
    del Q

    # If the matrix is "tall", compute an orthogonal transformation of the remaining
    # columns, generating a new polynomial basis
    if cut < M.shape[0]:
        Q = qr(M[cut:,cut:].T,pivoting=True)[0]
        M[:cut,cut:] = M[:cut,cut:] @ Q # Apply column transform

    # Return the backsolved columns and coefficient matrix for the quotient basis
    return solve_triangular(M[:cut,:cut],M[:cut,bezout_rank:]),Q[:,-bezout_bound:]

def reduce_macaulay_svd(M, cut, bezout_bound, max_cond=1e6):
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
    # Compute numerical rank
    s = svd(M, compute_uv=False)
    tol = max(M.shape)*s[0]*macheps
    rank = len(s[s>tol])
    # Check if numerical rank doesn't match bezout bound
    bezout_rank = M.shape[1]-bezout_bound
    if rank < bezout_rank:
        warn("Rank of Macaulay Matrix does not match the Bezout bound. Expected rank {}, found rank {}. System potentially has infinitely many solutions.".format(bezout_rank,rank))
    elif rank > bezout_rank:
        warn('Rank of Macaulay Matrix does not match the Bezout bound. Expected rank {}, found rank {}.'.format(bezout_rank,rank))

    # Check condition number before first QR
    cond_num = np.linalg.cond(M[:,:cut])
    if cond_num > max_cond:
        return None, "Condition number of the Macaulay high-degree columns is {}".format(cond_num)

    # QR reduce the highest-degree columns
    Q,M[:,:cut] = qr(M[:,:cut])
    M[:,cut:] = Q.T @ M[:,cut:]
    Q = None
    del Q

    # If the matrix is "tall", compute an orthogonal transformation of the remaining
    # columns, generating a new polynomial basis
    if cut < M.shape[0]:
        Q = svd(M[cut:,cut:])[2].conj().T
        M[:cut,cut:] = M[:cut,cut:] @ Q # Apply column transform

    # Return the backsolved columns and coefficient matrix for the quotient basis
    return solve_triangular(M[:cut,:cut],M[:cut,bezout_rank:]),Q[:,-bezout_bound:]

def reduce_macaulay_tvb(M, cut, bezout_bound, max_cond=1e6):
    # Compute numerical rank
    s = svd(M, compute_uv=False)
    tol = max(M.shape)*s[0]*macheps
    rank = len(s[s>tol])
    # Check if numerical rank doesn't match bezout bound
    bezout_rank = M.shape[1]-bezout_bound
    if rank < bezout_rank:
        warn("Rank of Macaulay Matrix does not match the Bezout bound. Expected rank {}, found rank {}. System potentially has infinitely many solutions.".format(bezout_rank,rank))
    elif rank > bezout_rank:
        warn('Rank of Macaulay Matrix does not match the Bezout bound. Expected rank {}, found rank {}.'.format(bezout_rank,rank))

    # Check condition number before first QR
    cond_num = np.linalg.cond(M[:,:cut])
    if cond_num > max_cond:
        return None, "Condition number of the Macaulay high-degree columns is {}".format(cond_num)

    # QR reduce the highest-degree columns
    Q,M[:,:cut] = qr(M[:,:cut])
    M[:,cut:] = Q.T @ M[:,cut:]
    Q = None
    del Q

    # If the matrix is "tall", compute an orthogonal transformation of the remaining
    # columns, generating a new polynomial basis
    if cut < M.shape[0]:
        M[cut:,cut:],P = qr(M[cut:,cut:], mode='r', pivoting=True)
        M[:cut,cut:] = M[:cut,cut:][:,P] # Permute columns

    # Check condition number before backsolve
    cond_num_back = np.linalg.cond(M[:bezout_rank,:bezout_rank])
    if cond_num_back > max_cond:
        return None, "Condition number of the Macaulay primary submatrix is {}".format(cond_num)

    return solve_triangular(M[:bezout_rank,:bezout_rank],M[:bezout_rank,bezout_rank:]),P

def reduce_macaulay_p(M, cut, P, max_cond=1e6):
    # Compute numerical rank
    s = svd(M, compute_uv=False)
    tol = max(M.shape)*s[0]*macheps
    rank = len(s[s>tol])
    # Check if numerical rank doesn't match bezout bound
    bezout_rank = M.shape[1]-bezout_bound
    if rank < bezout_rank:
        warn("Rank of Macaulay Matrix does not match the Bezout bound. Expected rank {}, found rank {}. System potentially has infinitely many solutions.".format(bezout_rank,rank))
    elif rank > bezout_rank:
        warn('Rank of Macaulay Matrix does not match the Bezout bound. Expected rank {}, found rank {}.'.format(bezout_rank,rank))

    # Check condition number before first QR
    cond_num = np.linalg.cond(M[:,:cut])
    if cond_num > max_cond:
        return None, "Condition number of the Macaulay high-degree columns is {}".format(cond_num)

    # QR reduce the highest-degree columns
    Q,M[:,:cut] = qr(M[:,:cut])
    M[:,cut:] = (Q.T @ M[:,cut:])[:,P]
    Q = None
    del Q

    # If the matrix is "tall", compute an orthogonal transformation of the remaining
    # columns, generating a new polynomial basis
    if cut < M.shape[0]:
        M[cut:,cut:] = qr(M[cut:,cut:])[1:]

    # Check condition number before backsolve
    cond_num_back = np.linalg.cond(M[:,:cut])
    if cond_num_back > max_cond:
        return None, "Condition number of the Macaulay primary submatrix is {}".format(cond_num)

    return solve_triangular(M[:bezout_rank,:bezout_rank],M[:bezout_rank,bezout_rank:]),P
