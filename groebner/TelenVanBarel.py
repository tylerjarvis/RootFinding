import numpy as np
import itertools
from scipy.linalg import qr, solve_triangular, qr_multiply
from groebner.polynomial import Polynomial, MultiCheb, MultiPower, is_power
from groebner.utils import row_swap_matrix, clean_zeros_from_matrix, TVBError, slice_top, get_var_list, mon_combos, mon_combosHighest, inverse_P, sort_polys_by_degree, deg_d_polys, all_permutations, num_mons_full, memoized_all_permutations, mons_ordered, all_permutations_cheb, num_mons
import time
import random
from matplotlib import pyplot as plt
from scipy.misc import comb
from math import factorial

def TelenVanBarel(initial_poly_list, accuracy = 1.e-10):
    """Uses Telen and VanBarels matrix reduction method to find a vector basis for the system of polynomials.

    Parameters
    --------
    initial_poly_list: list
        The polynomials in the system we are solving.
    run_checks : bool
        If True, checks will be run to make sure TVB works and if it doesn't an S-polynomial will be searched
        for to fix it.
    accuracy: float
        How small we want a number to be before assuming it is zero.

    Returns
    -----------
    basisDict : dict
        A dictionary of terms not in the vector basis a matrixes of things in the vector basis that the term
        can be reduced to.
    VB : numpy array
        The terms in the vector basis, each row being a term.
    degree : int
        The degree of the Macaualy matrix that was constructed.
    """
    power = is_power(initial_poly_list)
    dim = initial_poly_list[0].dim
    poly_coeff_list = []
    degree = find_degree(initial_poly_list)

    #This sorting is required for fast matrix construction. Ascending should be False.
    initial_poly_list = sort_polys_by_degree(initial_poly_list, ascending = False)

    """This is the first construction option, simple monomial multiplication."""
    #for i in initial_poly_list:
    #    poly_coeff_list = add_polys(degree, i, poly_coeff_list)
    """This is the second construction option, it uses the fancy triangle method that is faster but less stable."""
    #for deg in reversed(range(min([poly.degree for poly in initial_poly_list]), degree+1)):
    #    poly_coeff_list += deg_d_polys(initial_poly_list, deg, dim)

    #Creates the matrix for either of the above two methods. Comment out if using the third method.

    """This is the thrid matrix construction option, it uses the permutation arrays."""
    if power:
        matrix, matrix_terms, matrix_shape_stuff = createMatrixFast(initial_poly_list, degree, dim)
    else:
        matrix, matrix_terms, matrix_shape_stuff = construction(initial_poly_list, degree, dim)

    matrix, matrix_terms = rrqr_reduceTelenVanBarel2(matrix, matrix_terms, matrix_shape_stuff, accuracy = accuracy)

    height = matrix.shape[0]
    matrix[:,height:] = solve_triangular(matrix[:,:height],matrix[:,height:])
    matrix[:,:height] = np.eye(height)

    VB = matrix_terms[height:]

    basisDict = makeBasisDict(matrix, matrix_terms, VB, power, [degree]*dim)

    return basisDict, VB, degree

def makeBasisDict(matrix, matrix_terms, VB, power, remainder_shape):
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
    power : bool
        If True, the initial polynomials were MultiPower. If False, they were MultiCheb.
    remainder_shape: list
        The shape of the numpy arrays that will be mapped to in the basisDict.

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

    if power: #We don't actually need most of the rows, so we only get the ones we need.
        neededSpots = set()
        for term, mon in itertools.product(VB,get_var_list(VB.shape[1])):
            if tuple(term+mon) not in VBSet:
                neededSpots.add(tuple(term+mon))

    spots = list()
    for dim in range(VB.shape[1]):
        spots.append(VB.T[dim])

    for i in range(matrix.shape[0]):
        term = tuple(matrix_terms[i])
        if power and term not in neededSpots:
            continue
        remainder = np.zeros(remainder_shape)
        row = matrix[i]
        remainder[spots] = row[matrix.shape[0]:]
        basisDict[term] = remainder

    return basisDict

def find_degree(poly_list):
    '''Finds the degree of a Macaulay Matrix.

    Parameters
    --------
    poly_list: list
        The polynomials used to construct the matrix.

    Returns
    -----------
    find_degree : int
        The degree of the Macaulay Matrix.

    '''
    return sum(poly.degree for poly in poly_list) - len(poly_list) + 1

def add_polys(degree, poly, poly_coeff_list):
    """Adds polynomials to a Macaulay Matrix.

    This function is called on one polynomial and adds all monomial multiples of it to the matrix.

    Parameters
    ----------
    degree : int
        The degree of the TVB Matrix
    poly : Polynomial
        One of the polynomials used to make the matrix.
    poly_coeff_list : list
        A list of all the current polynomials in the matrix.
    Returns
    -------
    poly_coeff_list : list
        The original list of polynomials in the matrix with the new monomial multiplications of poly added.
    """
    poly_coeff_list.append(poly.coeff)
    deg = degree - poly.degree
    dim = poly.dim

    mons = mon_combos([0]*dim,deg)
    for i in mons[1:]: #skips the first all 0 mon
        poly_coeff_list.append(poly.mon_mult(i, returnType = 'Matrix'))
    return poly_coeff_list

def sorted_matrix_terms(degree, dim):
    '''Finds the matrix_terms sorted in the term order needed for TelenVanBarel reduction.
    So the highest terms come first,the x,y,z etc monomials last.
    Parameters
    ----------
    degree : int
        The degree of the TVB Matrix
    dim : int
        The dimension of the polynomials going into the matrix.
    Returns
    -------
    matrix_terms : numpy array
        The sorted matrix_terms.
    matrix_term_stuff : tuple
        The first entry is the number of 'highest' monomial terms. The second entry is the number of 'other' terms,
        those not in the first or third catagory. The third entry is the number of monomials of degree one of a
        single variable, as well as the monomial 1.
    '''
    highest_mons = mon_combosHighest([0]*dim,degree)[::-1]

    other_mons = list()
    d = degree - 1
    while d > 1:
        other_mons += mon_combosHighest([0]*dim,d)[::-1]
        d -= 1

    xs_mons = mon_combos([0]*dim,1)[::-1]
    sorted_matrix_terms = np.reshape(highest_mons+other_mons+xs_mons, (len(highest_mons+other_mons+xs_mons),dim))
    return sorted_matrix_terms, tuple([len(highest_mons),len(other_mons),len(xs_mons)])

def create_matrix(poly_coeffs, degree, dim):
    ''' Builds a Telen Van Barel matrix.

    Parameters
    ----------
    poly_coeffs : list.
        Contains numpy arrays that hold the coefficients of the polynomials to be put in the matrix.
    degree : int
        The degree of the TVB Matrix
    dim : int
        The dimension of the polynomials going into the matrix.
    Returns
    -------
    matrix : 2D numpy array
        The Telen Van Barel matrix.
    '''
    bigShape = [degree+1]*dim

    matrix_terms, matrix_shape_stuff = sorted_matrix_terms(degree, dim)

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

    if matrix_shape_stuff[0] > matrix.shape[0]: #The matrix isn't tall enough, these can't all be pivot columns.
        raise TVBError("HIGHEST NOT FULL RANK. TRY HIGHER DEGREE")

    #Sorts the rows of the matrix so it is close to upper triangular.
    matrix = row_swap_matrix(matrix)
    return matrix, matrix_terms, matrix_shape_stuff

def checkEqual(lst):
    '''Helper function for createMatrixFast. Checks if each element in a list is the same.
        
    Parameters
    ----------
    lst : list
        The list of interest.
    Returns
    -------
    checkEqual : bool
        True if each element in the list is the same. False otherwise.
    '''
    return lst.count(lst[0]) == len(lst)

def get_ranges(nums):
    '''Helper function for createMatrixFast. Finds where to slice the different parts of the matrix into.
    
    This is in an effort to avoid row_swap_matrix which can be slow. Instead, as we are buiding the part of the
    matrix corresponding to each polynomial seperately, this tells us where each part should go in the whole matrix.
    
    Parameters
    ----------
    nums : list
        The Macualay matrix degree minus the polynomial degrees for for each polynomial.
    Returns
    -------
    ranges : list
        The rows in the Macaulay Matrix that the given polynomail will be sliced into.
    '''
    ranges = []
    for i in nums:
        ranges.append(np.array([],dtype=int))
    start = 0
    count = 0
    n = len(nums)
    for num in nums:
        spot = count
        for r in ranges[count:]:
            r = np.hstack((r,np.arange(start,start+(n-count)*(num-len(r)),n-count)))
            ranges[spot] = r
            start+=1
            spot += 1
        start = ranges[-1][-1]+1
        count+=1
    return ranges

def createMatrixFast(polys, degree, dim):
    ''' Builds a Telen Van Barel matrix using fast construction.

    Parameters
    ----------
    poly_coeffs : list.
        Contains numpy arrays that hold the coefficients of the polynomials to be put in the matrix.
    degree : int
        The degree of the TVB Matrix
    dim : int
        The dimension of the polynomials going into the matrix.
    Returns
    -------
    matrix : 2D numpy array
        The Telen Van Barel matrix.
    '''
    bigShape = [degree+1]*dim

    matrix_terms, matrix_shape_stuff = sorted_matrix_terms(degree, dim)
    columns = len(matrix_terms)

    range_split = [num_mons_full(degree-poly.degree,dim) for poly in polys]
    rows = np.sum(range_split)
    ranges = get_ranges(range_split)    #How to slice the poly into the matrix rows.
    matrix = np.zeros((rows,columns))
    curr = 0
    
    #Get the slices needed to pull the matrix_terms from the coeff matrix.
    matrix_term_indexes = list()
    for row in matrix_terms.T:
        matrix_term_indexes.append(row)

    permutations = None
    currentDegree = 2
    #Adds the poly_coeffs to flat_polys, using added_zeros to make sure every term is in there.
    added_zeros = np.zeros(bigShape)

    for poly,matrix_range in zip(polys,ranges):
        slices = slice_top(poly.coeff)
        added_zeros[slices] = poly.coeff
        array = added_zeros[matrix_term_indexes]
        added_zeros[slices] = np.zeros_like(poly.coeff)

        permutations = memoized_all_permutations(degree - poly.degree, dim, degree, permutations, currentDegree)
        currentDegree = degree - poly.degree
        permList = list(permutations.values())
        
        temp = array[np.reshape(permList, (len(permList), columns))[::-1]]
        matrix[matrix_range] = temp

    if matrix_shape_stuff[0] > matrix.shape[0]: #The matrix isn't tall enough, these can't all be pivot columns.
        raise TVBError("HIGHEST NOT FULL RANK. TRY HIGHER DEGREE")
    #Sorts the rows of the matrix so it is close to upper triangular.
    if not checkEqual([poly.degree for poly in polys]): #Will need some switching possibly if some degrees are different.
        matrix = row_swap_matrix(matrix)
    return matrix, matrix_terms, matrix_shape_stuff

def construction(polys, degree, dim):
    ''' Builds a Telen Van Barel matrix using fast construction in the Chebyshev basis.

    Parameters
    ----------
    polys : list.
        Contains numpy arrays that hold the coefficients of the polynomials to be put in the matrix.
    degree : int
        The degree of the TVB Matrix
    dim : int
        The dimension of the polynomials going into the matrix.
    Returns
    -------
    matrix : 2D numpy array
        The Telen Van Barel matrix.
    '''
    bigShape = [degree+1]*dim
    matrix_terms, matrix_shape_stuff = sorted_matrix_terms(degree, dim)
    #print(matrix_shape_stuff)
    matrix_term_indexes = list()
    for row in matrix_terms.T:
        matrix_term_indexes.append(row)

    permutations = all_permutations_cheb(degree - np.min([poly.degree for poly in polys]), dim, degree)
    #print(permutations)
    added_zeros = np.zeros(bigShape)
    flat_polys = list()
    i = 0;
    for poly in polys:
        slices = slice_top(poly.coeff)
        added_zeros[slices] = poly.coeff
        array = added_zeros[matrix_term_indexes]
        added_zeros[slices] = np.zeros_like(poly.coeff)
        #print(array)

        #flat_polys.append(array[np.vstack(permutations.values())])
        degreeNeeded = degree - poly.degree
        mons = mons_ordered(dim,degreeNeeded)
        mons = np.pad(mons, (0,1), 'constant', constant_values = i)
        i += 1
        flat_polys.append(array)
        for mon in mons[1:-1]:
            result = np.copy(array)
            for i in range(dim):
                if mon[i] != 0:

                    mult = [0]*dim
                    mult[i] = mon[i]
                    result = np.sum(result[permutations[tuple(mult)]], axis = 0)
            flat_polys.append(result)
    #print(flat_polys)
    matrix = np.vstack(flat_polys)
    if matrix_shape_stuff[0] > matrix.shape[0]: #The matrix isn't tall enough, these can't all be pivot columns.
        raise TVBError("HIGHEST NOT FULL RANK. TRY HIGHER DEGREE")
    matrix = row_swap_matrix(matrix)
    #print(matrix)
    return matrix, matrix_terms, matrix_shape_stuff

def rrqr_reduceTelenVanBarel(matrix, matrix_terms, matrix_shape_stuff, accuracy = 1.e-10):
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

    #RRQR reduces A and D sticking the result in it's place.
    Q1,matrix[:,:highest_num],P1 = qr(matrix[:,:highest_num], pivoting = True)

    if abs(matrix[:,:highest_num].diagonal()[-1]) < accuracy:
        raise TVBError("HIGHEST NOT FULL RANK")

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
    rank = np.sum(np.abs(matrix.diagonal())>accuracy)
    matrix = matrix[:rank]

    #Resorts the matrix_terms.
    matrix_terms[:highest_num] = matrix_terms[:highest_num][P1]
    #permutations[:highest_num] = permutations[:highest_num][P1]
    matrix_terms[highest_num:highest_num+others_num] = matrix_terms[highest_num:highest_num+others_num][P]
    #permutations[highest_num:highest_num+others_num] = permutations[highest_num:highest_num+others_num][P]

    return matrix, matrix_terms

def rrqr_reduceTelenVanBarel2(matrix, matrix_terms, matrix_shape_stuff, accuracy = 1.e-10):
    ''' Reduces a Telen Van Barel Macaulay matrix.

    This function does the same thing as rrqr_reduceTelenVanBarel but uses qr_multiply instead of qr and a multiplication
    to make the function faster and more memory efficient.

    Parameters
    ----------
    matrix : numpy array.
        The Macaulay matrix, sorted in TVB style.
    matrix_terms: numpy array
        Each row of the array contains a term in the matrix. The i'th row corresponds to
        the i'th column in the matrix.
    matrix_shape_stuff : tuple
        Terrible name I know. It has 3 values, the first is how many columns are in the
        'highest' part of the matrix. The second is how many are in the 'others' part of
        the matrix, and the third is how many are in the 'xs' part.
    accuracy : float
        What is determined to be 0.
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

    C1,matrix[:highest_num,:highest_num],P1 = qr_multiply(matrix[:,:highest_num], matrix[:,highest_num:].T, mode = 'right', pivoting = True)
    matrix[:highest_num,highest_num:] = C1.T
    C1 = 0

    if abs(matrix[:,:highest_num].diagonal()[-1]) < accuracy:
        raise TVBError("HIGHEST NOT FULL RANK")

    matrix[:highest_num,highest_num:] = solve_triangular(matrix[:highest_num,:highest_num],matrix[:highest_num,highest_num:])
    matrix[:highest_num,:highest_num] = np.eye(highest_num)
    matrix[highest_num:,highest_num:] -= (matrix[highest_num:,:highest_num][:,P1])@matrix[:highest_num,highest_num:]
    matrix_terms[:highest_num] = matrix_terms[:highest_num][P1]
    P1 = 0

    C,R,P = qr_multiply(matrix[highest_num:,highest_num:highest_num+others_num], matrix[highest_num:,highest_num+others_num:].T, mode = 'right', pivoting = True)

    matrix = matrix[:R.shape[0]+highest_num]
    matrix[highest_num:,:highest_num] = np.zeros_like(matrix[highest_num:,:highest_num])
    matrix[highest_num:,highest_num:highest_num+R.shape[1]] = R
    matrix[highest_num:,highest_num+R.shape[1]:] = C.T
    C,R = 0,0

    #Shifts the columns of B.
    matrix[:highest_num,highest_num:highest_num+others_num] = matrix[:highest_num,highest_num:highest_num+others_num][:,P]
    matrix_terms[highest_num:highest_num+others_num] = matrix_terms[highest_num:highest_num+others_num][P]
    P = 0

    # Check if there are no solutions
    rank = np.sum(np.abs(matrix.diagonal())>accuracy)
    # extra_block = matrix[rank:, -matrix_shape_stuff[2]:]
    # Q,R = qr(extra_block)
    # if np.sum(np.abs(R.diagonal())>accuracy) == matrix_shape_stuff[2]:
    #     raise ValueError("The system given has no roots.")

    #Get rid of 0 rows at the bottom.
    matrix = matrix[:rank]
    return matrix, matrix_terms

def rrqr_reduceTelenVanBarelFullRank(matrix, matrix_terms, matrix_shape_stuff, accuracy = 1.e-10):
    ''' Reduces a Telen Van Barel Macaulay matrix.

    This function does the same thing as rrqr_reduceTelenVanBarel2 but only works if the matrix is full rank.
    In this case it is faster.

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
    accuracy : float
        What is determined to be 0.
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

    C1,matrix[:highest_num,:highest_num],P1 = qr_multiply(matrix[:highest_num,:highest_num], matrix[:highest_num,highest_num:].T, mode = 'right', pivoting = True)
    matrix[:highest_num,highest_num:] = C1.T
    C1 = 0

    if abs(matrix[highest_num][highest_num]) < accuracy:
        raise TVBError("HIGHEST NOT FULL RANK")

    matrix_terms[:highest_num] = matrix_terms[:highest_num][P1]
    P1 = 0

    C,matrix[highest_num:,highest_num:highest_num+others_num],P = qr_multiply(matrix[highest_num:,highest_num:highest_num+others_num], matrix[highest_num:,highest_num+others_num:].T, mode = 'right', pivoting = True)

    matrix[highest_num:,highest_num+others_num:] = C.T
    C,R = 0,0

    #Shifts the columns of B.
    matrix[:highest_num,highest_num:highest_num+others_num] = matrix[:highest_num,highest_num:highest_num+others_num][:,P]
    matrix_terms[highest_num:highest_num+others_num] = matrix_terms[highest_num:highest_num+others_num][P]
    P = 0    
    return matrix, matrix_terms
