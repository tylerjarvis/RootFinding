import numpy as np
import itertools
from groebner.utils import clean_zeros_from_matrix, get_var_list
from groebner.TelenVanBarel import create_matrix, add_polys
from groebner.root_finder import _random_poly
from scipy.linalg import solve_triangular, eig
from scipy.stats import mode
from scipy.linalg import qr

def division_cheb(test_polys):
    dim = test_polys[0].dim
    
    matrix_degree = np.sum(poly.degree for poly in test_polys) - len(test_polys) + 1

    poly_coeff_list = []
    for i in test_polys:
        poly_coeff_list = add_polys(matrix_degree, i, poly_coeff_list)
    matrix, matrix_terms, matrix_shape_stuff = create_matrix(poly_coeff_list, matrix_degree, dim)
    perm, cut = matrix_term_perm(matrix_terms)
    matrix = matrix[:,perm]
    matrix_terms = matrix_terms[perm]

    #Reduce it.
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

    diag = matrix_terms[:matrix.shape[0]]
    zero_spots = np.where(diag == 0)
    maxY = np.max(zero_spots[0][np.where(zero_spots[1] == 0)])
    inverses = np.vstack((-np.ones(maxY+1), np.arange(maxY+1))).T
    inv_matrix_terms = np.vstack((inverses, VB))
    inv_matrix = np.zeros([len(inverses),len(inv_matrix_terms)])

    VB_size = len(VB)

    rowSpotDict = dict()
    spot = 0
    for row in inv_matrix_terms:
        rowSpotDict[tuple(row)] = spot
        spot+=1

    diagDict = dict()
    for i in range(diag.shape[0]):
        term = diag[i]
        diagDict[tuple(term)] = matrix[i][-VB_size:]

    for i in range(maxY+1):
        spot = matrix_terms[i]
        y = spot[1]
        inv_matrix[i] = divide_row_x(matrix[i][-VB_size:], matrix_terms[-VB_size:],\
                                     inv_matrix_terms, rowSpotDict, diagDict, VB_size)
        inv_matrix[i][y] += 1

    Q,R = qr(inv_matrix)
    solution = np.hstack((np.eye(R.shape[0]),solve_triangular(R[:,:R.shape[0]], R[:,R.shape[0]:])))

    VBSpotDict = dict()
    spot = 0
    for row in VB:
        VBSpotDict[tuple(row)] = spot
        spot+=1

    invSpotDict = dict()
    for i in range(len(solution)):
        invSpotDict[tuple(inv_matrix_terms[i])] = solution[i][len(solution):]

    division_matrix = build_division_matrix(VB, VBSpotDict, diagDict, invSpotDict)
    vals, vecs = eig(division_matrix.T)

    for spot in range(len(VB)):
        term = VB[spot]
        if term[1] == 1:
            if tuple(term - [0,1]) in VBSpotDict:
                ys = [spot, VBSpotDict[tuple(term - [0,1])]]
                break
    print(VB[ys[0]],VB[ys[1]])
    zeroY = vecs[ys[0]]/vecs[ys[1]]
    zeroX = 1/vals
    zeros = list()
    for i in range(len(zeroX)):
        zeros.append(np.array([zeroX[i], zeroY[i]]))
    
    return zeros
    
def matrix_term_perm(matrix_terms):
    boundary = np.where(matrix_terms[:,0] == 0)[0]
    bSet = set(boundary)
    other = np.arange(len(matrix_terms))
    mask = [num not in bSet for num in other]
    other = other[mask]
    return np.hstack((boundary,other)), len(boundary)

def divide_row_x(coeffs, terms, inv_matrix_terms, rowSpotDict, diagDict, VB_size):
    """Divides a row of the matrix by x.
    
    Parameters
    ----------
    coeffs : numpy array.
        The numerical values of the terms we want to divide by x.
    terms: numpy array
        The terms corresponding to the numerical values.
    inv_matrix_terms: numpy array
        The terms in the inverse matrix.
    Returns
    -------
    new_row : numpy array
        The row we get in the inverse_matrix by dividing the first row by x.
    """
    new_row = np.zeros(len(inv_matrix_terms))
    for i in range(len(coeffs)):
        new_row+=divide_term_x(coeffs[i], terms[i], inv_matrix_terms, rowSpotDict, diagDict, VB_size)
    return new_row

def divide_term_x(coeff, term, inv_matrix_terms, rowSpotDict, diagDict, VB_size):
    """Divides a term of the matrix by x.
    
    Parameters
    ----------
    coeff : float
        The numerical value of the term we want to divide by x.
    term: numpy array
        The term to divide.
    inv_matrix_terms: numpy array
        The terms in the inverse matrix.
    Returns
    -------
    row : numpy array
        The row we get in the inverse_matrix by dividing the term by x.
    """
    row = np.zeros(len(inv_matrix_terms))
    divisor_terms = get_divisor_terms(term)
    parity = 1
    for spot in divisor_terms[:-1]:
        if tuple(spot) in rowSpotDict:
            row[rowSpotDict[tuple(spot)]] += parity*2
        else:
            row[-VB_size:] -= 2*parity*diagDict[tuple(spot)]
        parity*=-1
    spot = divisor_terms[-1]
    if tuple(spot) in rowSpotDict:
        row[rowSpotDict[tuple(spot)]] += parity
    else:
        row[-VB_size:] -= parity*diagDict[tuple(spot)]
    row*=coeff
    return row

def get_divisor_terms(term):
    initial = term - [1,0]
    height = term[0]//2+1
    terms = np.vstack([initial.copy() for i in range(height)])
    dec = 0
    for i in range(terms.shape[0]):
        terms[i][0]-=2*dec
        dec+=1
    return terms

def build_division_matrix(VB, VBSpotDict, diagDict, invSpotDict):
    div_matrix = np.zeros((len(VB), len(VB)))
    for i in range(len(VB)):
        term = VB[i]
        terms = get_divisor_terms(term)
        parity = 1
        for spot in terms[:-1]:
            if tuple(spot) in VBSpotDict:
                div_matrix[VBSpotDict[tuple(spot)]][i]+=2*parity
            else:
                div_matrix[:,i] -= 2*parity*diagDict[tuple(spot)][-len(VB):]
            parity *= -1
        spot = terms[-1]
        if tuple(spot) in diagDict:
            div_matrix[:,i] -= parity*diagDict[tuple(spot)][-len(VB):]
        else:
            div_matrix[:,i] -= parity*invSpotDict[tuple(spot)]
    return div_matrix
