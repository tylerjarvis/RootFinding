import numpy as np
import itertools
from scipy.linalg import solve_triangular, eig, qr
from yroots import LinearProjection
from yroots.polynomial import MultiCheb, MultiPower, is_power
from yroots.MacaulayReduce import add_polys, rrqr_reduceMacaulay
from yroots.utils import get_var_list, slice_top, row_swap_matrix, \
                              mon_combos, newton_polish, MacaulayError
from yroots.Division import create_matrix

def divisionNew(polys, divisor_var=0, tol=1.e-12, verbose=False, polish=False, return_all_roots=True):
    '''Calculates the common zeros of polynomials using a division matrix.

    Parameters
    --------
    polys: list of MultiCheb Polynomials
        The polynomials for which the common roots are found.
    divisor_var : int
        What variable is being divided by. 0 is x, 1 is y, etc. Defaults to x.
    tol : float
        The tolerance parameter for the Macaulay Reduce.
    verbose : bool
        If True prints information about the solve.
    polish: bool
        If True runs a newton polish on the zeros before returning.

    Returns
    -----------
    zeros : numpy array
        The common roots of the polynomials. Each row is a root.
    '''
    #This first section creates the Macaulay Matrix with the monomials that don't have
    #the divisor variable in the first columns.
    polys, transform, is_projected = LinearProjection.remove_linear(polys, 1e-4, 1e-8)
    if len(polys) == 1:
        from yroots.OneDimension import solve
        return transform(solve(polys[0], MSmatrix=0))
    power = is_power(polys)
    if power:
        raise ValueError("This only works for Chebyshev polynomials")

    dim = polys[0].dim
    matrix_degree = np.sum([poly.degree for poly in polys]) - len(polys) + 1
    poly_coeff_list = []
    for poly in polys:
        poly_coeff_list = add_polys(matrix_degree, poly, poly_coeff_list)

    matrix, matrix_terms, cuts = create_matrix(poly_coeff_list, matrix_degree,\
                                                   dim, divisor_var)

    x_pows_over_y = matrix_terms[:cuts[0]].copy()
    x_pows_over_y[:,divisor_var] = -np.ones(cuts[0], dtype = 'int')
    inv_matrix_terms = np.vstack((x_pows_over_y, matrix_terms))

    num_rows = matrix.shape[0]
    inv_matrix = np.zeros([num_rows + matrix.shape[0], len(inv_matrix_terms)])
    cuts = tuple([cuts[0], cuts[0]+cuts[1]])

    #A dictionary of term in inv_matrix_terms to their spot in inv_matrix_terms.
    inv_spot_dict = dict()
    spot = 0
    for term in inv_matrix_terms:
        inv_spot_dict[tuple(term)] = spot
        spot+=1

    #A dictionary of terms to the terms in their quotient when divided by x. (symbolically)
    divisor_terms_dict = dict()
    for term in matrix_terms:
        divisor_terms_dict[tuple(term)] = get_divisor_terms(term, divisor_var)

    #A dictionary of terms to their quotient when divided by x. (in the vector basis)
    term_divide_dict = dict()
    for term in matrix_terms:
        term_divide_dict[tuple(term)] = divide_term(term, inv_matrix_terms, inv_spot_dict,
                                                    divisor_terms_dict)

    #Builds the inv_matrix by dividing the rows of matrix by x.
    for i in range(num_rows):
        inv_matrix[i] = divide_row(matrix[i], matrix_terms, term_divide_dict,
                                   len(inv_matrix_terms))

    inv_matrix[num_rows:,cuts[0]:] = matrix

    matrix, matrix_terms, perm = rrqr_reduceMacaulay(inv_matrix, inv_matrix_terms, cuts,
                                               accuracy=tol, return_perm=True)
    if isinstance(matrix, int):
        return -1

    for term in term_divide_dict:
        term_divide_dict[tuple(term)] = term_divide_dict[tuple(term)][perm]

    VB = matrix_terms[matrix.shape[0]:]
    basisDict = makeBasisDict2(matrix, matrix_terms)

#     print(len(VB))

    #Dictionary of terms in the vector basis their spots in the matrix.
    VBdict = {}
    spot = 0
    for row in VB:
        VBdict[tuple(row)] = spot
        spot+=1

    #Builds the division matrix and finds the eigenvalues and eigenvectors.
    division_matrix = build_division_matrix(VB, VBdict, basisDict, term_divide_dict,
                                           matrix_terms)

    vals, vecs = eig(division_matrix,left=True,right=False)
    #conjugate because scipy gives the conjugate eigenvector
    vecs = vecs.conj()

    if len(vals) > len(np.unique(np.round(vals, 10))):
        return -1

    vals2, vecs2 = eig(vecs)
    sorted_vals2 = np.sort(np.abs(vals2)) #Sorted smallest to biggest
    if sorted_vals2[0] < sorted_vals2[-1]*tol:
        return -1

    if verbose:
        print("\nDivision Matrix\n", np.round(division_matrix[::-1,::-1], 2))
        print("\nLeft Eigenvectors (as rows)\n", vecs.T)

    if np.max(np.abs(vals)) > 1.e6:
        return -1

    #Calculates the zeros, the x values from the eigenvalues and the y values from the eigenvectors.
    zeros = list()

    for i in range(len(vals)):
        if  np.abs(vals[i]) < 1.e-5:
            continue
        root = np.zeros(dim, dtype=complex)
        for spot in range(0,divisor_var):
            root[spot] = vecs[-(2+spot)][i]/vecs[-1][i]
        for spot in range(divisor_var+1,dim):
            root[spot] = vecs[-(1+spot)][i]/vecs[-1][i]

        root[divisor_var] = 1/vals[i]

        if polish:
            root = newton_polish(polys,root,tol = tol)

        #throw out bad roots in cheb
        if np.any([abs(poly(root)) > 1.e-1 for poly in polys]):
            continue

        zeros.append(root)

    if return_all_roots:
        return transform(np.array(zeros))
    else:
        # only return roots in the unit complex hyperbox
        zeros = transform(np.array(zeros))
        return zeros[np.all(np.abs(zeros) <= 1,axis = 0)]

def divide_row(coeffs, terms, term_divide_dict, length):
    """Divides a row of the matrix by the divisor variable..

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

def divide_term(term, inv_matrix_terms, inv_spot_dict, divisor_terms_dict):
    """Divides a term of the matrix by the divisor variable.

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
        row[inv_spot_dict[tuple(spot)]] += parity*2
        parity*=-1
    spot = divisor_terms[-1]
    row[inv_spot_dict[tuple(spot)]] += parity
    return row

def get_divisor_terms(term, divisor_var):
    """Finds the terms that will be present when dividing a given term by x.

    Parameters
    ----------
    term: numpy array
        The term to divide.
    divisor_var : int
        What variable is being divided by. 0 is x, 1 is y, etc.

    Returns
    -------
    terms : numpy array
        Each row is a term that will be in the quotient.
    """
    dim = len(term)
    diff = np.zeros(dim)
    diff[divisor_var] = 1
    initial = term - diff
    height = term[divisor_var]//2+1
    terms = np.vstack([initial.copy() for i in range(height)])
    dec = 0
    for i in range(terms.shape[0]):
        terms[i][divisor_var]-=2*dec
        dec+=1
    return terms

def build_division_matrix(VB, VBdict, basisDict, term_divide_dict,
                          matrix_terms):
    """Builds the division matrix.

    Parameters
    ----------
    VB: numpy array
        The vector basis.
    VBdict : dictionary
        A dictionary of term in the vector basis to their spot in the vector basis.
    diag_reduction_dict : dictionary
        A dictionary of terms on the diagonal to their reduction in the vector basis.
    inv_reduction_dict : dictionary
        A dictionary of terms of type y^k/x to their reduction in the vector basis.
    divisor_terms_dict : dictionary
        A dictionary of terms to the terms in their dividend when divided by x.

    Returns
    -------
    div_matrix : numpy array
        The division matrix.
    """
    div_matrix = np.zeros((len(VB), len(VB)))
    for i in range(len(VB)):
        term = VB[i]
        row = term_divide_dict[tuple(term)]
        for spot, val in enumerate(row):
            if val != 0:
                term = tuple(matrix_terms[spot])
                if term in VBdict:
                    div_matrix[VBdict[term]][i] += val
                else:
                    div_matrix[:,i] -= val*basisDict[term]

    return div_matrix

def makeBasisDict2(matrix, matrix_terms):
    '''Calculates and returns the basisDict.

    This is a dictionary of the terms on the diagonal of the reduced Macaulay
    matrix to the terms in the Vector Basis.
    It is used to create the multiplication matrix in root_finder.

    Parameters
    --------
    matrix: numpy array
        The reduced Macaulay matrix.
    matrix_terms : numpy array
        The terms in the matrix. The i'th row is the term represented by the i'th
        column of the matrix.
    VB : numpy array
        Each row is a term in the vector basis.

    Returns
    -----------
    basisDict : dict
        Maps terms on the diagonal of the reduced Macaulay matrix (tuples) to numpy
        arrays of the shape remainder_shape
        that represent the terms reduction into the Vector Basis.
    '''
    basisDict = {}
    for i in range(matrix.shape[0]):
        term = tuple(matrix_terms[i])
        basisDict[term] = matrix[i][matrix.shape[0]:]
    return basisDict
