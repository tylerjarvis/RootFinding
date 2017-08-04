from operator import itemgetter
import itertools
import numpy as np
import math
from scipy.linalg import lu, qr, solve_triangular, inv, solve, svd
from numpy.linalg import cond
from groebner.polynomial import Polynomial, MultiCheb, MultiPower
from scipy.sparse import csc_matrix, vstack
from groebner.utils import Term, row_swap_matrix, fill_size, clean_zeros_from_matrix, triangular_solve, divides
import matplotlib.pyplot as plt
import time
from collections import defaultdict

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
    times = {}
    startTime = time.time()
    MultiCheb.clearTime()
    MultiPower.clearTime()
    Polynomial.clearTime()

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

    startAdding = time.time()
    for i in initial_poly_list:
        poly_coeff_list = add_polys(degree, i, poly_coeff_list)
    endAdding = time.time()
    times["adding polys"] = (endAdding - startAdding)

    startCreate = time.time()
    matrix, matrix_terms, matrix_shape_stuff = create_matrix(poly_coeff_list, initial_poly_list)
    endCreate = time.time()
    times["create matrix"] = (endCreate - startCreate)
    
    print(matrix.shape)
        
    startReduce = time.time()
    matrix, matrix_terms = rrqr_reduceTelenVanBarel(matrix, matrix_terms, matrix_shape_stuff, 
                                                        global_accuracy = global_accuracy)
    matrix = clean_zeros_from_matrix(matrix)
    non_zero_rows = np.sum(abs(matrix),axis=1) != 0
    matrix = matrix[non_zero_rows,:] #Only keeps the non_zero_polymonials
    endReduce = time.time()
    times["reduce matrix"] = (endReduce - startReduce)

    #plt.matshow([i==0 for i in matrix])

    startTri = time.time()
    matrix = triangular_solve(matrix)
    matrix = clean_zeros_from_matrix(matrix)
    endTri = time.time()
    times["triangular solve"] = (endTri - startTri)
    
    startBasisDict = time.time()
    VB = matrix_terms[matrix.shape[0]:]
    basisDict = makeBasisDict(matrix, matrix_terms, VB)
    endBasisDict = time.time()
    times["basisDict"] = (endBasisDict - startBasisDict)
    
    endTime = time.time()
    print("TelenVanBarel run time is {} seconds".format(endTime-startTime))
    print(times)
    Polynomial.printTime()
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
        for spot in np.where(row != 0)[0]:
            remainder[matrix_terms[spot]] = row[spot]
        basisDict[pivotSpot] = remainder
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

def mon_combosFull(mon, numLeft, spot = 0):
    '''
    Same as mon_combos but returns only monomials OF the degree, not those less than it.
    '''
    answers = list()
    if len(mon) == spot+1: #We are at the end of mon, no more recursion.
        mon[spot] = numLeft
        answers.append(mon.copy())
        return answers
    if numLeft == 0: #Nothing else can be added.
        answers.append(mon.copy())
        return answers
    temp = mon.copy() #Quicker than copying every time inside the loop.
    for i in range(numLeft+1): #Recursively add to mon further down.
        temp[spot] = i
        answers += mon_combosFull(temp, numLeft-i, spot+1)
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

def in_basis(highest, term):
    '''
    Returns True if term divides one of highest. False otherwise.
    '''
    for mon in highest:
        if divides(term,mon):
            return True
    return False

def sort_matrix(matrix, matrix_terms, initial_polys):
    '''
    Takes a matrix and matrix_terms (holding the terms in each column of the matrix), and sorts them both
    by the term order needed for TelenVanBarel reduction. So the highest terms come first, the x,y,z etc monomials last.
    Returns the sorted matrix and matrix_terms.
    '''
    degree = find_degree(initial_polys)
    num_initial_polys = len(initial_polys)
    highest = set()
    #Get a better way to determine highest stuff. Those that when multiplied by x,y,z etc don't fit a mon we have.
    dim = None
    '''
    for poly in initial_polys:
        degree_needed = poly.degree - degree
        dim = poly.dim
        mons = mon_combosFull(np.zeros(dim, dtype = int),degree_needed)
        for term in zip(*np.where(poly.coeff != 0)):
            for mon in mons:
                highest.add(mon+term)
    '''
    for term in matrix_terms:
        if sum(term) == degree:
            highest.add(term)
    
    
    xs = set()
    for i in range(num_initial_polys+1):
        for term in matrix_terms:
            for spot in range(len(term)):
                mon = np.zeros_like(term)
                mon[spot] = i
                mon = tuple(mon)
                if term == mon:
                    xs.add(term)
    others = set()
    for term in matrix_terms:
        if term not in xs and term not in highest:
            others.add(term)
    sorted_matrix_terms = list(highest) + list(others) + list(xs)

    order = np.zeros(len(matrix_terms), dtype = int)
    matrix_termsList = list(matrix_terms)
    for i in range(len(matrix_terms)):
        order[i] = matrix_termsList.index(sorted_matrix_terms[i])
    return matrix[:,order], sorted_matrix_terms, tuple([len(highest),len(others),len(xs)])

def clean_matrix(matrix, matrix_terms):
    '''
    Gets rid of columns in the matrix that are all zero and returns it and the updated matrix_terms.
    '''
    non_zero_monomial = np.sum(abs(matrix), axis=0) != 0
    matrix = matrix[:,non_zero_monomial] #Only keeps the non_zero_monomials
    matrix_terms = matrix_terms[non_zero_monomial] #Only keeps the non_zero_monomials
    return matrix, matrix_terms

def create_matrix(polys_coeffs, initial_polys):
    '''
    Takes a list of polynomial objects (polys) and uses them to create a matrix. That is ordered by the monomial
    ordering. Returns the matrix and the matrix_terms, a list of the monomials corresponding to the rows of the matrix.
    '''
    #Gets an empty polynomial whose lm all other polynomial divide into.
    bigShape = np.maximum.reduce([p.shape for p in polys_coeffs])
    #Gets a list of all the flattened polynomials.
    flat_polys = list()
    for coeff in polys_coeffs:
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
    matrix, matrix_terms, matrix_shape_stuff = sort_matrix(matrix, matrix_terms, initial_polys)

    #Sorts the rows of the matrix so it is close to upper triangular.
    matrix = row_swap_matrix(matrix)
    return matrix, matrix_terms, matrix_shape_stuff

def rrqr_reduceTelenVanBarel(matrix, matrix_terms, matrix_shape_stuff, clean = False, global_accuracy = 1.e-10):
    '''
    Reduces a Telen Van Barel Macaulay matrix.
    '''
    highest_num = matrix_shape_stuff[0]
    others_num = matrix_shape_stuff[1]
    xs_num = matrix_shape_stuff[2]
        
    highest = matrix_terms[:highest_num]
    others = matrix_terms[highest_num:highest_num+others_num]
    xs = matrix_terms[highest_num+others_num:]
    
    A = matrix[:,:highest_num+others_num]
    
    Mhigh = matrix[:highest_num,highest_num+others_num:]
    Mlow = matrix[highest_num:,highest_num+others_num:]
    
    B = A[:highest_num,:highest_num]
    C = A[:highest_num,highest_num:]
    D = A[highest_num:,:highest_num]
    E = A[highest_num:,highest_num:]
    
    Q,R,P = qr(E, pivoting = True)
    Q1,R1,P1 = qr(B, pivoting = True)
    
    matrix = np.vstack((np.hstack((R1, Q1.T@(C[:,P]), Q1.T@Mhigh)), np.hstack((np.zeros_like(D), R, Q.T@Mlow))))
    
    non_zero_rows = np.sum(abs(matrix),axis=1) > global_accuracy
    matrix = matrix[non_zero_rows,:] #Only keeps the non_zero_polymonials
    
    highest = list(np.array(highest)[P1])
    others = list(np.array(others)[P])
    matrix_termsTemp = highest+others+xs
    #I need matrix_terms to have tuples but the above code makes it nd arrays. Annoying, but this flips it back.
    matrix_terms = list()
    for i in matrix_termsTemp:
        matrix_terms.append(tuple(i))
    
    return matrix, matrix_terms