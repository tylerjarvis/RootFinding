from operator import itemgetter
import itertools
import numpy as np
import math
from scipy.linalg import lu, qr, solve_triangular, inv, solve, svd, qr_multiply
from numpy.linalg import cond
from groebner.polynomial import Polynomial, MultiCheb, MultiPower
from scipy.sparse import csc_matrix, vstack
from groebner.utils import Term, row_swap_matrix, fill_size, clean_zeros_from_matrix, triangular_solve, divides, get_var_list
import matplotlib.pyplot as plt
from collections import defaultdict
import gc
import time

def TelenVanBarel(initial_poly_list, accuracy = 1.e-10):
    '''
    Macaulay will take a list of polynomials and use them to construct a Macaulay matrix.

    Parameters
    --------
    initial_poly_list: A list of polynomials
    accuracy: float
        How small we want a number to be before assuming it is zero.

    Returns
    -----------
    basis_dict : dictionary
        Maps terms on the diagonal of the TvB matrix to their representation
        in the vector space basis
    VB : ?
        Monomials representing the vector space basis

    '''
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

    matrix, matrix_terms, matrix_shape_stuff = create_matrix(poly_coeff_list)

    matrix, matrix_terms = rrqr_reduceTelenVanBarel2(matrix, matrix_terms, matrix_shape_stuff,
                                                        global_accuracy = accuracy)
    matrix = clean_zeros_from_matrix(matrix)

    matrix, matrix_terms = triangular_solve(matrix, matrix_terms, reorder = False)
    matrix = clean_zeros_from_matrix(matrix)

    VB = matrix_terms[matrix.shape[0]:]
    basis_dict = makeBasisDict(matrix, matrix_terms, VB)
    return basis_dict, VB

def makeBasisDict(matrix, matrix_terms, VB):
    '''
    Take a matrix that has been traingular solved and returns a dictionary
    mapping the pivot columns terms behind them, all of which will be in the
    vector basis. All the matrixes that are mapped to will be the same shape.
    '''
    remainder_shape = np.maximum.reduce([mon for mon in VB])
    remainder_shape += np.ones_like(remainder_shape)
    basisDict = {}

    spots = list()
    for dim in range(matrix_terms.shape[1]):
        spots.append(matrix_terms[matrix.shape[0]:].T[dim])

    for i in range(matrix.shape[0]):
        remainder = np.zeros(remainder_shape)
        row = matrix[i]
        pivotSpot = matrix_terms[i]
        row[i] = 0
        remainder[spots] = row[matrix.shape[0]:]
        basisDict[tuple(pivotSpot)] = remainder

    return basisDict

def find_degree(poly_list):
    '''Finds the degree needed for the Macaulay matrix.

    Parameters
    ----------
    poly_list : list, contains polynomial objects
        The polynomials from which to make the Macaulay matrix

    Returns
    -------
    int
        The necessary degree of the Macaulay matrix

    Notes
    -----
    The formula for finding the degree of the Macaulay matrix is
    .. math::
        \sum{d_i} - (n - 1)
    Where d_i is the degree of each polynomial in poly_list, and n is the
    total number of polynomials in poly_list.

    '''

    return sum(poly.degree for poly in poly_list) - len(poly_list) + 1

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

def inVarList(term, varList):
    ''' Checks is a term is in the varList
    Parameters
    ----------
    term : numpy array.
        The term of interest.
    varList : list
        Each object in the list is a numpy array corresponding to a monomial.
    Returns
    -------
    bool : bool
        True if the term is in the varList, False otherwise.
    '''
    for i in varList:
        if (term == i).all():
            return True
    return False

def sort_matrix_terms(matrix_terms):
    '''Sorts the matrix_terms by the term order needed for TelenVanBarel reduction.
    So the highest terms come first,the x,y,z etc monomials last.
    Parameters
    ----------
    matrix_terms : numpy array.
        Each row is one of the terms in the matrix.
    Returns
    -------
    matrix_terms : numpy array
        The sorted matrix_terms.
    matrix_term_stuff : tuple
        The first entry is the number of 'highest' monomial terms. The second entry is the number of 'other' terms,
        those not in the first or third catagory. The third entry is the number of monomials of degree one of a
        single variable, as well as the monomial 1.
    '''
    highest = list()
    dim = len(matrix_terms[0])

    var_list = get_var_list(dim)
    matrix_termSet = set([tuple(term) for term in matrix_terms])
    for i in range(matrix_terms.shape[0]):
        term = tuple(matrix_terms[i])
        mons = term + np.array(var_list)
        if not all(tuple(mon) in matrix_termSet for mon in mons):
            highest.append(i)

    var_list2 = var_list
    var_list2.append(np.zeros(dim, dtype=int))

    others = list()
    for i in range(len(matrix_terms)):
        term = matrix_terms[i]
        if not inVarList(term, var_list2) and i not in highest:
            others.append(i)
    sorted_matrix_terms = np.vstack((matrix_terms[highest], matrix_terms[others], var_list2))
    return sorted_matrix_terms, tuple([len(highest),len(others),len(var_list)])

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
    ''' Builds a Telen Van Barel matrix.

    Parameters
    ----------
    poly_coeffs : list.
        Contains numpy arrays that hold the coefficients of the polynomials to be put in the matrix.
    Returns
    -------
    matrix : 2D numpy array
        The Telen Van Barel matrix.
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

    matrix_terms, matrix_shape_stuff = sort_matrix_terms(matrix_terms)

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
    return matrix, matrix_terms, matrix_shape_stuff

def rrqr_reduceTelenVanBarel(matrix, matrix_terms, matrix_shape_stuff, global_accuracy = 1.e-10):
    ''' Reduces a Telen Van Barel Macaulay matrix.

    The matrix is split into the shape
    A B C
    D E F
    Where A is square and contains all the highest terms, and C contains all the x,y,z etc. terms. The lengths
    are determined by the matrix_shape_stuff tuple. First A and D are reduced using rrqr, and then the rest of
    the matrix is multiplied by Q.T to change it accordingly. Then E is reduced by rrqr, the rows of B are shifted
    accordingly, and F is multipled by Q.T to change it accordingly. This is all done in place to save memory.

    Parameters
    ----------
    matrix : numpy array.
        The Macaulay matrix, sorted in TVB style.
    matrix_terms: numpy array
        Each row of the array contains a term in the matrix. The i'th row corresponds to
        the i'th column in the matrix.
    matrix_shape_stuff : tuple
        Terrible name I know. It has 3 values, the first is how many columnns are in the
        'highest' part of the matrix. The second is how many are in the 'others' part of
        the matrix, and the third is how many are in the 'xs' part.
    Returns
    -------
    matrix : numpy array
        The reduced matrix.
    matrix_terms: numpy array
        The resorted matrix_terms.
    '''
    highest_num = matrix_shape_stuff[0]
    others_num = matrix_shape_stuff[1]
    xs_num = matrix_shape_stuff[2]

    #Try going down to halfway down the matrix. Faster, but if not full rank may cause problems.
    half = min(matrix.shape[0]//2, (highest_num + others_num)//2)
    diff = half - highest_num
    if diff > 0 and diff < others_num:
        highest_num += diff
        others_num -= diff

    #RRQR reduces A and D sticking the result in it's place.
    Q1,matrix[:,:highest_num],P1 = qr(matrix[:,:highest_num], pivoting = True)

    #Multiplying the rest of the matrix by Q.T
    matrix[:,highest_num:] = Q1.T@matrix[:,highest_num:]
    Q1 = 0 #Get rid of Q1 for memory purposes.

    #RRQR reduces E sticking the result in it's place.
    Q,matrix[highest_num:,highest_num:highest_num+others_num],P = qr(matrix[highest_num:,highest_num:highest_num+others_num], pivoting = True)

    #Multiplies F by Q.T.
    matrix[highest_num:,highest_num+others_num:] = Q.T@matrix[highest_num:,highest_num+others_num:]
    Q = 0 #Get rid of Q for memory purposes.

    #Shifts the columns of B
    matrix[:highest_num,highest_num:highest_num+others_num] = matrix[:highest_num,highest_num:highest_num+others_num][:,P]

    #Checks for 0 rows and gets rid of them.
    non_zero_rows = list()
    for i in range(min(highest_num+others_num, matrix.shape[0])):
        if np.abs(matrix[i][i]) > global_accuracy:
            non_zero_rows.append(i)
    matrix = matrix[non_zero_rows,:]

    #Resorts the matrix_terms.
    matrix_terms[:highest_num] = matrix_terms[:highest_num][P1]
    matrix_terms[highest_num:highest_num+others_num] = matrix_terms[highest_num:highest_num+others_num][P]

    return matrix, matrix_terms


def rrqr_reduceTelenVanBarel2(matrix, matrix_terms, matrix_shape_stuff, global_accuracy = 1.e-10):
    ''' Reduces a Telen Van Barel Macaulay matrix.

    This function does the same thing as rrqr_reduceTelenVanBarel but uses qr_multiply instead of qr and a multiplication
    to make the function faster and more memory efficient. It could be less stable, but I haven't seen any stability
    problems while running it.

    Parameters
    ----------
    matrix : numpy array.
        The Macaulay matrix, sorted in TVB style.
    matrix_terms: numpy array
        Each row of the array contains a term in the matrix. The i'th row corresponds to
        the i'th column in the matrix.
    matrix_shape_stuff : tuple
        Terrible name I know. It has 3 values, the first is how many columnns are in the
        'highest' part of the matrix. The second is how many are in the 'others' part of
        the matrix, and the third is how many are in the 'xs' part.
    Returns
    -------
    matrix : numpy array
        The reduced matrix.
    matrix_terms: numpy array
        The resorted matrix_terms.
    '''
    highest_num = matrix_shape_stuff[0]
    others_num = matrix_shape_stuff[1]
    xs_num = matrix_shape_stuff[2]
    ''' #This is breaking right now for this function, not sure why.
    #Try going down to halfway down the matrix.
    half = min(matrix.shape[0]//2, (highest_num + others_num)//2)
    diff = half - highest_num
    if diff > 0 and diff < others_num:
        highest_num += diff
        others_num -= diff


    C1,R1,P1 = qr_multiply(matrix[:,:highest_num], matrix[:,highest_num:].T, mode = 'right', pivoting = True)
    matrix = np.vstack((np.hstack((R1,C1.T)),matrix[highest_num:]))

    A = matrix[highest_num:,:highest_num][:,P1]
    matrix_terms[:highest_num] = matrix_terms[:highest_num][P1]
    P1 = 0
    B = matrix[highest_num:,highest_num:]
    B -= A@solve_triangular(R1,C1.T)
    R1,C1 = 0,0

    '''
    C1,matrix[:highest_num,:highest_num],P1 = qr_multiply(matrix[:,:highest_num], matrix[:,highest_num:].T, mode = 'right', pivoting = True)
    matrix[:highest_num,highest_num:] = C1.T
    C1 = 0

    matrix[:highest_num,highest_num:] = solve_triangular(matrix[:highest_num,:highest_num],matrix[:highest_num,highest_num:])
    matrix[:highest_num,:highest_num] = np.eye(highest_num)
    matrix[highest_num:,highest_num:] -= (matrix[highest_num:,:highest_num][:,P1])@matrix[:highest_num,highest_num:]
    matrix_terms[:highest_num] = matrix_terms[:highest_num][P1]
    P1 = 0

    C,R,P = qr_multiply(matrix[highest_num:,highest_num:highest_num+others_num], matrix[highest_num:,highest_num+others_num:].T, mode = 'right', pivoting = True)
    matrix = np.vstack((matrix[:highest_num],np.hstack((np.zeros_like(matrix[highest_num:R.shape[0]+highest_num,:highest_num]),R,C.T))))
    C,R = 0,0

    #Shifts the columns of B
    matrix[:highest_num,highest_num:highest_num+others_num] = matrix[:highest_num,highest_num:highest_num+others_num][:,P]
    matrix_terms[highest_num:highest_num+others_num] = matrix_terms[highest_num:highest_num+others_num][P]
    P = 0

    #Checks for 0 rows and gets rid of them.
    non_zero_rows = list()
    for i in range(min(highest_num+others_num, matrix.shape[0])):
        if np.abs(matrix[i][i]) > global_accuracy:
            non_zero_rows.append(i)
    matrix = matrix[non_zero_rows,:]

    return matrix, matrix_terms
