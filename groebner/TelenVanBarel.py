from operator import itemgetter
import itertools
import numpy as np
import math
from scipy.linalg import lu, qr, solve_triangular, inv, solve, svd
from numpy.linalg import cond
from groebner.polynomial import Polynomial, MultiCheb, MultiPower
from scipy.sparse import csc_matrix, vstack
from groebner.utils import Term, row_swap_matrix, fill_size, clean_zeros_from_matrix, triangular_solve, divides, get_var_list
import matplotlib.pyplot as plt
from collections import defaultdict
import gc
import time

def TelenVanBarel(initial_poly_list, global_accuracy = 1.e-10):
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
    matrix, matrix_terms, matrix_shape_stuff = create_matrix2(poly_coeff_list)
        
    matrix, matrix_terms = rrqr_reduceTelenVanBarel(matrix, matrix_terms, matrix_shape_stuff, 
                                                        global_accuracy = global_accuracy)
    matrix = clean_zeros_from_matrix(matrix)
    non_zero_rows = np.sum(np.abs(matrix),axis=1) != 0
    matrix = matrix[non_zero_rows,:] #Only keeps the non_zero_polymonials

    matrix, matrix_terms = triangular_solve(matrix, matrix_terms, reorder = False)
    matrix = clean_zeros_from_matrix(matrix)
    
    VB = list()
    for i in matrix_terms[matrix.shape[0]:]:
        VB.append(tuple(i))
    
    VB = matrix_terms[matrix.shape[0]:]
    basisDict = makeBasisDict(matrix, matrix_terms, VB)
    return basisDict, VB

def makeBasisDict(matrix, matrix_terms, VB):
    '''
    Take a matrix that has been traingular solved and returns a dictionary mapping the pivot columns terms
    behind them, all of which will be in the vector basis. All the matrixes that are mapped to will be the same shape.
    '''
    remainder_shape = np.maximum.reduce([mon for mon in VB])
    remainder_shape += np.ones_like(remainder_shape)
    basisDict = {}
    for i in range(matrix.shape[0]):
        remainder = np.zeros(remainder_shape)
        row = matrix[i]
        pivotSpot = matrix_terms[i]
        row[i] = 0
        spots = list()
        for dim in range(matrix_terms.shape[1]):
            spots.append(matrix_terms[matrix.shape[0]:].T[dim])
        remainder[spots] = row[matrix.shape[0]:]
        basisDict[tuple(pivotSpot)] = remainder
    
    return basisDict

def find_degree(poly_list):
    '''
    Takes a list of polynomials and finds the degree needed for a Macaulay matrix.
    Adds the degree of each polynomial and then subtracts the total number of polynomials and adds one.

    Example:
        For polynomials [P1,P2,P3] with degree [d1,d2,d3] the function returns d1+d2+d3-3+1
    '''
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

def sort_matrix(matrix, matrix_terms):
    '''
    Takes a matrix and matrix_terms (holding the terms in each column of the matrix), and sorts them both
    by the term order needed for TelenVanBarel reduction. So the highest terms come first, the x,y,z etc monomials last.
    Returns the sorted matrix and matrix_terms.
    '''
    highest = set()
    dim = len(matrix_terms[0])

    var_list = get_var_list(dim)
    matrix_termSet = set(matrix_terms)
    for term in matrix_terms:
        mons = term + np.array(var_list)
        if not all(tuple(mon) in matrix_termSet for mon in mons):
            highest.add(term)

    var_list = get_var_list(dim)
    var_list.append(tuple(np.zeros(dim, dtype=int)))
    for mon in var_list:
        if mon not in matrix_termSet:
            matrix_terms = np.append(matrix_terms, 0)
            matrix_terms[::-1][0] = mon
            matrix = np.hstack((matrix,np.zeros((matrix.shape[0],1))))

    others = set()
    for term in matrix_terms:
        if term not in var_list and term not in highest:
            others.add(term)
    sorted_matrix_terms = list(highest) + list(others) + list(var_list)

    order = np.zeros(len(matrix_terms), dtype = int)
    matrix_termsList = list(matrix_terms)
    for i in range(len(matrix_terms)):
        order[i] = matrix_termsList.index(sorted_matrix_terms[i])
    return matrix[:,order], sorted_matrix_terms, tuple([len(highest),len(others),len(var_list)])

def isNonZeroColumn(col):
    return np.any(col)

def clean_matrix(matrix, matrix_terms):
    '''
    Gets rid of columns in the matrix that are all zero and returns it and the updated matrix_terms.
    '''
    if len(matrix_terms[0]) == 2: #The matrix is more dense in this case so this is faster.
        keepers = [np.any(matrix[:,i:i+1]) for i in range(matrix.shape[1])]
    else: #In more than 2D the matrix is much less dense, so this is faster.
        non_zero_columns = set(np.where(matrix != 0)[1])
        keepers = [i in non_zero_columns for i in range(matrix.shape[1])]    
    matrix = matrix[:,keepers]
    matrix_terms = matrix_terms[keepers]
    return matrix, matrix_terms

def inVarList(term, varList):
    for i in varList:
        if (term == i).all():
            return True
    return False

def sort_matrix_terms(matrix_terms):
    '''
    Sorts the matrix_terms by the term order needed for TelenVanBarel reduction. So the highest terms come first,
    the x,y,z etc monomials last.
    Returns the matrix_terms and a tuple containing the number of elements in each part of the matrix (highest, others, xs).
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

def create_matrix2(poly_coeffs):
    bigShape = np.maximum.reduce([p.shape for p in poly_coeffs])

    non_zeroSet = set()
    for coeff in poly_coeffs:
        for term in zip(*np.where(coeff != 0)):
            non_zeroSet.add(term)
    matrix_terms = np.zeros_like(bigShape)
    for term in non_zeroSet:
        matrix_terms = np.vstack((matrix_terms,term))
    matrix_terms = matrix_terms[1:]
        
    matrix_terms, matrix_shape_stuff = sort_matrix_terms(matrix_terms)
        
    flat_polys = list()
    for coeff in poly_coeffs:
        slices = list()
        for i in range(len(bigShape)):
            slices.append(matrix_terms.T[i])
        flat_polys.append(fill_size(bigShape,coeff)[slices])
    #Make the matrix
    matrix = np.vstack(flat_polys[::-1])
        
    #Sorts the rows of the matrix so it is close to upper triangular.
    matrix = row_swap_matrix(matrix)
    return matrix, matrix_terms, matrix_shape_stuff

def create_matrix(poly_coeffs):
    '''
    Takes a list of polynomial objects (polys) and uses them to create a matrix. That is ordered by the monomial
    ordering. Returns the matrix and the matrix_terms, a list of the monomials corresponding to the rows of the matrix.
    '''
    #Gets an empty polynomial whose lm all other polynomial divide into.
    bigShape = np.maximum.reduce([p.shape for p in poly_coeffs])

    #Gets a list of all the flattened polynomials.
    flat_polys = list()
    for coeff in poly_coeffs:
        #Gets a matrix that is padded so it is the same size as biggest, and flattens it. This is so
        #all flattened polynomials look the same.
        newMatrix = fill_size(bigShape, coeff)
        flat_polys.append(newMatrix.ravel())

    #Make the matrix
    matrix = np.vstack(flat_polys[::-1])

    #Makes matrix_terms, a list of all the terms in the matrix.
    terms = np.zeros(bigShape, dtype = tuple)
    for i,j in np.ndenumerate(terms):
        terms[i] = i
    matrix_terms = terms.ravel()

    #Gets rid of any columns that are all 0.
    matrix, matrix_terms = clean_matrix(matrix, matrix_terms)

    #Sorts the matrix and matrix_terms by term order.
    matrix, matrix_terms, matrix_shape_stuff = sort_matrix(matrix, matrix_terms)

    #Sorts the rows of the matrix so it is close to upper triangular.
    matrix = row_swap_matrix(matrix)
    return matrix, matrix_terms, matrix_shape_stuff

def rrqr_reduceTelenVanBarel(matrix, matrix_terms, matrix_shape_stuff, global_accuracy = 1.e-10):
    ''' Reduces a Telen Van Barel Macaulay matrix.
    parameters
    ----------
    matrix : numpy array.
        The Macaulay matrix, sorted in TVB style.
    matrix_terms: numpy array
        Each row of the array contains a term in the matrix. The i'th row corresponds to the i'th column in the matrix.
    matrix_shape_stuff : tuple
        Terrible name I know. It has 3 values, the first is how many columnns are in the 'highest' part of the matrix.
        The second is how many are in the 'others' part of the matrix, and the third is how many are in the 'xs' part.
    returns
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
    half = matrix.shape[0]//2
    diff = half - highest_num
    highest_num += diff
    others_num -= diff
        
    Highs = matrix[:,:highest_num]
    Others = matrix[:,highest_num:]
    Q1,R1,P1 = qr(Highs, pivoting = True, check_finite = False)
    
    matrix[:,:highest_num] = R1
    matrix[:,highest_num:] = Q1.T@Others

    C = matrix[:highest_num,highest_num:highest_num+others_num]
    E = matrix[highest_num:,highest_num:highest_num+others_num]
    Mlow = matrix[highest_num:,highest_num+others_num:]

    Q,R,P = qr(E, pivoting = True, check_finite = False)
    matrix[:highest_num,highest_num:highest_num+others_num] = C[:,P]
    matrix[highest_num:,highest_num:highest_num+others_num] = R
    matrix[highest_num:,highest_num+others_num:] = Q.T@Mlow
        
    non_zero_rows = np.sum(np.abs(matrix[:,:highest_num+others_num]),axis=1) > global_accuracy
    matrix = matrix[non_zero_rows,:] #Only keeps the non_zero_polymonials
    non_zero_rows = list()
    for i in range(matrix.shape[0]):
        if np.abs(matrix[i][i]) > global_accuracy:
            non_zero_rows.append(i)
    matrix = matrix[non_zero_rows,:] #Only keeps the non_zero_polymonials
    
    #Resort the Matrix terms 
    highest = matrix_terms[:highest_num]
    others = matrix_terms[highest_num:highest_num+others_num]
    xs = matrix_terms[highest_num+others_num:]
    highest = highest[P1]
    others = others[P]
    
    matrix_terms = np.vstack((highest,others,xs))
    return matrix, matrix_terms
