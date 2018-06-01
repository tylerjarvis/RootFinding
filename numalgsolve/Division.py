import numpy as np
import itertools
from numalgsolve.polynomial import MultiCheb, MultiPower, is_power
from numalgsolve.utils import get_var_list, slice_top, row_swap_matrix, mon_combos, newton_polish
from numalgsolve.TVBCore import add_polys, rrqr_reduceTelenVanBarel
from scipy.linalg import solve_triangular, eig, qr

def division(polys, divisor_var = 0, tol = 1.e-12):
    '''Calculates the common zeros of polynomials using a division matrix.

    Parameters
    --------
    polys: MultiCheb Polynomials
        The polynomials for which the common roots are found.
    divisor_var : int
        What variable is being divided by. 0 is x, 1 is y, etc. Defaults to x.

    Returns
    -----------
    zeros : list
        The common zeros of the polynomials. Each list element is a numpy array of complex entries
        that contains the coordinates in each dimension of the zero.
    '''
    #This first section creates the Macaulay Matrix with the monomials that don't have
    #the divisor variable in the first columns.
    power = is_power(polys)
    dim = polys[0].dim
    matrix_degree = np.sum(poly.degree for poly in polys) - len(polys) + 1

    poly_coeff_list = []
    for poly in polys:
        poly_coeff_list = add_polys(matrix_degree, poly, poly_coeff_list)

    matrix, matrix_terms, cuts = create_matrix(poly_coeff_list, matrix_degree, dim, divisor_var)

    #Reduces the Macaulay matrix like normal.
    matrix, matrix_terms = rrqr_reduceTelenVanBarel(matrix, matrix_terms, cuts, tol)

    rows,columns = matrix.shape
    VB = matrix_terms[matrix.shape[0]:]
    matrix = np.hstack((np.eye(rows),solve_triangular(matrix[:,:rows],matrix[:,rows:])))


    #------------> chebyshev
    if not power:
        #Builds the inverse matrix. The terms are the vector basis as well as y^k/x terms for all k. Reducing
        #this matrix allows the y^k/x terms to be reduced back into the vector basis.
        inverses = matrix_terms[np.where(matrix_terms[:,divisor_var] == 0)[0]]
        inverses[:,divisor_var] = -np.ones(inverses.shape[0], dtype = 'int')
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
            divisor_terms_dict[tuple(term)] = get_divisor_terms(term, divisor_var)

        #A dictionary of terms to their quotient when divided by x.
        term_divide_dict = dict()
        for term in matrix_terms[-len(VB):]:
            term_divide_dict[tuple(term)] = divide_term(term, inv_matrix_terms, inv_spot_dict, diag_reduction_dict,
                                                          len(VB), divisor_terms_dict)

        #Builds the inv_matrix by dividing the rows of matrix by x.
        for i in range(cuts[0]):
            inv_matrix[i] = divide_row(matrix[i][-len(VB):], matrix_terms[-len(VB):], term_divide_dict, len(inv_matrix_terms))
            spot = matrix_terms[i]
            spot[divisor_var] -= 1
            inv_matrix[i][inv_spot_dict[tuple(spot)]] += 1

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
        #<---------end Chebyshev
    else:
        #--------->Power
        basisDict = makeBasisDict(matrix, matrix_terms, VB)

        #Dictionary of terms in the vector basis their spots in the matrix.
        VBdict = {}
        spot = 0
        for row in VB:
            VBdict[tuple(row)] = spot
            spot+=1

        # Build division matrix
        division_matrix = np.zeros((len(VB), len(VB)))
        for i in range(VB.shape[0]):
            var = np.zeros(dim)
            var[divisor_var] = 1
            term = tuple(VB[i] - var)
            if term in VBdict:
                division_matrix[VBdict[term]][i] += 1
            else:
                division_matrix[:,i] -= basisDict[term]
        #<----------end Power

    vals, vecs = eig(division_matrix.T)

    #Calculates the zeros, the x values from the eigenvalues and the y values from the eigenvectors.
    zeros = list()
    for i in range(len(vals)):
        if abs(vecs[-1][i]) < 1.e-10: #This should be a root at infinity
            continue
        root = np.zeros(dim, dtype=complex)
        root[divisor_var] = 1/vals[i]
        for spot in range(0,divisor_var):
            root[spot] = vecs[-(2+spot)][i]/vecs[-1][i]
        for spot in range(divisor_var+1,dim):
            root[spot] = vecs[-(1+spot)][i]/vecs[-1][i]
        #root = newton_polish(polys,root,tol = tol)
        zeros.append(root)

    return zeros

def get_matrix_terms(poly_coeffs, dim, divisor_var, deg):
    '''Finds the terms in the Macaulay matrix.

    Parameters
    --------
    poly_coeffs: list
        A list of numpy arrays that contain the coefficients of the polynomials to go into the Macaualy Matrix.
    dim : int
        The dimension of the polynomials in the matrix.
    divisor_var : int
        What variable is being divided by. 0 is x, 1 is y, etc.

    Returns
    -----------
    matrix_terms : numpy array
        The matrix_terms. The ith row is the term represented by the ith column of the matrix.
    cuts : tuple
        When the matrix is reduced it is split into 3 parts with restricted pivoting. These numbers indicate
        where those cuts happen.
    '''
    """
    #The following code is just for testing, just two dimensional divide by x.
    mDeg = deg
    array = np.array(mon_combos([0]*dim,mDeg))
    perm = np.arange(mDeg+1)
    perm = np.hstack((perm, np.arange(len(perm)+2,len(array)), np.arange(len(perm),len(perm)+2)[::-1]))
    mons = array[perm]
    cuts = tuple([mDeg+1, len(perm) - 2])
    return mons, cuts
    """
    matrix_term_set_y= set()
    matrix_term_set_other= set()
    for coeffs in poly_coeffs:
        for term in zip(*np.where(coeffs != 0)):
            if term[divisor_var] == 0:
                matrix_term_set_y.add(term)
            else:
                matrix_term_set_other.add(term)

    needed_terms = list()
    base = np.zeros(dim, dtype = 'int')
    base[divisor_var] = 1
    matrix_term_set_other.remove(tuple(base))
    matrix_term_end = base.copy()
    for i in range(dim):
        if i != divisor_var:
            base[i] = 1
            term = tuple(base)
            matrix_term_set_other.remove(term)
            matrix_term_end = np.vstack((term,matrix_term_end))
            base[i] = 0
    for term in needed_terms:
        matrix_term_set_other.remove(term)

    matrix_terms = np.vstack((np.vstack(matrix_term_set_y),np.vstack(matrix_term_set_other),matrix_term_end))

    return matrix_terms, tuple([len(matrix_term_set_y), len(matrix_term_set_y)+len(matrix_term_set_other)])

def makeBasisDict(matrix, matrix_terms, VB):
    '''Calculates and returns the basisDict.

    This is a dictionary of the terms on the diagonal of the reduced TVB matrix to the terms in the Vector Basis.
    It is used to create the multiplication matrix in root_finder.

    Parameters
    --------
    matrix: numpy array
        The reduced TVB matrix.
    matrix_terms : numpy array
        The terms in the matrix. The i'th row is the term represented by the i'th column of the matrix.
    VB : numpy array
        Each row is a term in the vector basis.

    Returns
    -----------
    basisDict : dict
        Maps terms on the diagonal of the reduced TVB matrix (tuples) to numpy arrays of the shape remainder_shape
        that represent the terms reduction into the Vector Basis.
    '''
    basisDict = {}

    VBSet = set()
    for i in VB:
        VBSet.add(tuple(i))

    #We don't actually need most of the rows, so we only get the ones we need.
    neededSpots = set()
    for term, mon in itertools.product(VB,get_var_list(VB.shape[1])):
        if tuple(term-mon) not in VBSet:
            neededSpots.add(tuple(term-mon))

    for i in range(matrix.shape[0]):
        term = tuple(matrix_terms[i])
        if term not in neededSpots:
            continue
        basisDict[term] = matrix[i][matrix.shape[0]:]

    return basisDict

def create_matrix(poly_coeffs, degree, dim, divisor_var):
    ''' Builds a Macaulay matrix for reduction.

    Parameters
    ----------
    poly_coeffs : list.
        Contains numpy arrays that hold the coefficients of the polynomials to be put in the matrix.
    degree : int
        The degree of the TVB Matrix
    dim : int
        The dimension of the polynomials going into the matrix.
    divisor_var : int
        What variable is being divided by. 0 is x, 1 is y, etc. Defaults to x.

    Returns
    -------
    matrix : 2D numpy array
        The Telen Van Barel matrix.
    matrix_terms : numpy array
        The ith row is the term represented by the ith column of the matrix.
    cuts : tuple
        When the matrix is reduced it is split into 3 parts with restricted pivoting. These numbers indicate
        where those cuts happen.
    '''
    bigShape = [degree+1]*dim
    matrix_terms, cuts = get_matrix_terms(poly_coeffs, dim, divisor_var, degree)

    #Get the slices needed to pull the matrix_terms from the coeff matrix.
    matrix_term_indexes = list()
    for row in matrix_terms.T:
        matrix_term_indexes.append(row)

    #Adds the poly_coeffs to flat_polys, using added_zeros to make sure every term is in there.
    added_zeros = np.zeros(bigShape)
    flat_polys = list()
    for coeff in poly_coeffs:
        slices = slice_top(coeff)
        added_zeros[slices] = coeff
        flat_polys.append(added_zeros[matrix_term_indexes])
        added_zeros[slices] = np.zeros_like(coeff)
        coeff = 0
    poly_coeffs = 0

    #Make the matrix. Reshape is faster than stacking.
    matrix = np.reshape(flat_polys, (len(flat_polys),len(matrix_terms)))

    #Sorts the rows of the matrix so it is close to upper triangular.
    matrix = row_swap_matrix(matrix)
    return matrix, matrix_terms, cuts

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

def divide_term(term, inv_matrix_terms, inv_spot_dict, diag_reduction_dict, VB_size, divisor_terms_dict):
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

def get_divisor_terms(term, divisor_var):
    """Finds the terms that will be present when dividing a given term by x.

    Parameters
    ----------
    term: numpy array
        The term to divide.

    Returns
    -------
    terms : numpy array
        Each row is a term that will be in the quotient.
    divisor_var : int
        What variable is being divided by. 0 is x, 1 is y, etc.
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
