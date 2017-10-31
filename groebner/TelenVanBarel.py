import numpy as np
import itertools
from scipy.linalg import qr, solve_triangular, qr_multiply
<<<<<<< HEAD
from groebner.polynomial import Polynomial, MultiCheb, MultiPower, is_power
from groebner.utils import row_swap_matrix, clean_zeros_from_matrix, TVBError, slice_top, get_var_list, mon_combos, mon_combosHighest, inverse_P
=======
from groebner.polynomial import Polynomial, MultiCheb, MultiPower
from groebner.utils import row_swap_matrix, clean_zeros_from_matrix, TVBError, slice_top, get_var_list, mon_combos, mon_combosHighest, inverse_P, sort_polys_by_degree, deg_d_polys, all_permutations
>>>>>>> a45fac974b1981ec69bf2aeecef1c761d0ecf920
import time
import random
from matplotlib import pyplot as plt
from scipy.misc import comb
from math import factorial

<<<<<<< HEAD
def mons_ordered(dim, deg):
    mons_ordered = []
    for i in range(deg):
        for j in mon_combosHighest([0]*dim,i):
            mons_ordered.append(j)
    return np.array(mons_ordered)

def numMons(deg, dim):
    return comb(deg+dim-1,deg,exact=True)

def sort_polys_by_degree(polys):
    degs = [poly.degree for poly in polys]
    argsort_list = np.argsort(degs)
    sorted_polys = list()
    for i in argsort_list:
        sorted_polys.append(polys[i])
    return sorted_polys

def deg_d_polys(polys, d, dim, keep = False):
    ignoreVar = 0
    poly_coeff_list = list()
    for poly in polys:
        mons = mon_combosHighest([0]*dim,d - poly.degree)
        for mon in mons:
            if np.all([mon[i] <= (polys[i].degree - 1) for i in range(ignoreVar)]):
                poly_coeff_list.append(poly.mon_mult(mon, returnType = 'Matrix'))
            #elif random.random() > 1:
            #    poly_coeff_list.append(poly.mon_mult(mon, returnType = 'Matrix'))
            #elif keep and random.random() > 0:
            #    poly_coeff_list.append(poly.mon_mult(mon, returnType = 'Matrix'))
        ignoreVar += 1
    return poly_coeff_list

def arrays(deg,dim,mon):
    '''mon = 0 is lowest mon, mon == dim-1 is highest'''
    if dim-1==mon:
        total = numMons(deg, dim)
        end = numMons(deg, dim-1)
        return [True]*(total-end)+[False]*end
    elif deg==1:
        temp = [False]*(dim)
        temp[dim-mon-1] = True
        return temp
    else:
        return arrays(deg-1,dim,mon)+arrays(deg,dim-1,mon)

def permutationArray(deg,dim,mon):
    if mon == dim -1:
        array = [False]
        for d in range(1,deg+1):
            array = arrays(d,dim,mon) + array
    else:
        array = [False]
        first = [False]*(dim)
        first[dim-mon-1] = True
        array = first + array
        for d in range(2,deg+1):
            first = first + arrays(d,dim-1,mon)
            array = first+array
    #print(array)
    return np.array(inverse_P(np.hstack((np.where(~np.array(array))[0],np.where(array)[0]))))

def all_permutations_cheb(deg,dim,matrixDegree):
    permutations = {}
    for i in range(dim):
        mon = [0]*dim
        mon[i] = 1
        mon = tuple(mon)
        permutations[mon] = np.array([permutationArray(matrixDegree,dim,dim-1-i), inverse_P(permutationArray(matrixDegree,dim,dim-1-i))])
    #print("Cheb permutations made")
    #print(permutations)
    return permutations

def all_permutations(deg,dim,matrixDegree):
    permutations = {}
    for i in range(dim):
        mon = [0]*dim
        mon[i] = 1
        mon = tuple(mon)
        permutations[mon] = permutationArray(matrixDegree,dim,dim-1-i)

    varList = permutations.keys()

    for d in range(2,deg+1):
        mons = mon_combosHighest([0]*dim,d)
        for mon in mons:
            for var in varList:
                diff = tuple(np.subtract(mon,var))
                if diff in permutations:
                    permutations[tuple(mon)] = permutations[var][permutations[diff]]
                    break
    #print(permutations)
    return permutations

def createMatrix2(polys, degree, dim):
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
    if is_power(polys):
        permutations = all_permutations(degree - np.min([poly.degree for poly in polys]), dim, degree)
    else:
        permutations = all_permutations_cheb(degree - np.min([poly.degree for poly in polys]), dim, degree)
    #print(permutations[(0,1)],permutations[(1,0)])
    #Adds the poly_coeffs to flat_polys, using added_zeros to make sure every term is in there.
    added_zeros = np.zeros(bigShape)
    flat_polys = list()
    for poly in polys:
        slices = slice_top(poly.coeff)
        added_zeros[slices] = poly.coeff
        array = added_zeros[matrix_term_indexes]
        added_zeros[slices] = np.zeros_like(poly.coeff)
        #print(array)
        flat_polys.append(array)
        #flat_polys.append(array[np.vstack(permutations.values())])
        degreeNeeded = degree - poly.degree
        mons = mon_combos([0]*dim,degreeNeeded)
        for mon in mons[1:]:
            flat_polys.append(array[permutations[tuple(mon)]])
        added_zeros[slices] = np.zeros_like(poly.coeff)

    #Make the matrix. Reshape is faster than stacking.
    #matrix = np.vstack(flat_polys)
    matrix = np.reshape(flat_polys, (len(flat_polys),len(matrix_terms)))

    if matrix_shape_stuff[0] > matrix.shape[0]: #The matrix isn't tall enough, these can't all be pivot columns.
        raise TVBError("HIGHEST NOT FULL RANK. TRY HIGHER DEGREE")

    #Sorts the rows of the matrix so it is close to upper triangular.
    matrix = row_swap_matrix(matrix)
    return matrix, matrix_terms, matrix_shape_stuff

def construction(polys, degree, dim):
    bigShape = [degree+1]*dim
    matrix_terms, matrix_shape_stuff = sorted_matrix_terms(degree, dim)

    matrix_term_indexes = list()
    for row in matrix_terms.T:
        matrix_term_indexes.append(row)

    permutations = all_permutations_cheb(degree - np.min([poly.degree for poly in polys]), dim, degree)
    #print(permutations)
    added_zeros = np.zeros(bigShape)
    flat_polys = dict()
    i = 0;
    for poly in polys:
        slices = slice_top(poly.coeff)
        added_zeros[slices] = poly.coeff
        array = added_zeros[matrix_term_indexes]
        added_zeros[slices] = np.zeros_like(poly.coeff)
        #print(array)

        #flat_polys.append(array[np.vstack(permutations.values())])
        degreeNeeded = degree - poly.degree
        mons = mons_ordered(dim,degree)
        mons = np.pad(mons, (0,1), 'constant', constant_values = i)
        i += 1
        flat_polys[tuple(mons[0])] = array
        for mon in mons[1:-1]:
            not_zero = np.nonzero(mon)[0]
            mult = mon.copy()
            mult[not_zero[0]] -= 1
            var_to_mult = [0]*dim
            var_to_mult[not_zero[0]] = 1
            flat_polys[tuple(mon)] = flat_polys[tuple(mult)][permutations[tuple(var_to_mult)][0]] + flat_polys[tuple(mult)][permutations[tuple(var_to_mult)][1]]
    #matrix = []
    #for poly in mons[:-1]:
    #    matrix.append(flat_polys[tuple(poly)])
    #matrix = np.reshape(flat_polys.values(), (len(flat_polys.values()),len(matrix_terms)))
    matrix = np.vstack(flat_polys.values())
    print(matrix)
    if matrix_shape_stuff[0] > matrix.shape[0]: #The matrix isn't tall enough, these can't all be pivot columns.
        raise TVBError("HIGHEST NOT FULL RANK. TRY HIGHER DEGREE")
    matrix = row_swap_matrix(matrix)
    return matrix, matrix_terms, matrix_shape_stuff

=======
>>>>>>> a45fac974b1981ec69bf2aeecef1c761d0ecf920
def TelenVanBarel(initial_poly_list, run_checks = True, accuracy = 1.e-10):
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
<<<<<<< HEAD
    print(degree)

    initial_poly_list = sort_polys_by_degree(initial_poly_list)
=======
    
    #This sorting is required for fast matrix construction. Ascending should be False.
    initial_poly_list = sort_polys_by_degree(initial_poly_list, ascending = False)
>>>>>>> a45fac974b1981ec69bf2aeecef1c761d0ecf920

    if run_checks:
        #Checks to make sure TVB will work.
        if not has_top_xs(initial_poly_list):
            raise TVBError("Doesn't have all x^n's on diagonal. Do linear transformation")
        S = get_S_Poly(initial_poly_list)
        if isinstance(S,Polynomial):
            initial_poly_list.append(S)
            degree = find_degree(initial_poly_list)
<<<<<<< HEAD
    #start = time.time()
    #for i in initial_poly_list:
        #poly_coeff_list = add_polys(degree, i, poly_coeff_list)

    #for d in reversed(range(min([poly.degree for poly in initial_poly_list]), degree+1)):
    #    poly_coeff_list += deg_d_polys(initial_poly_list, d, dim, d==(degree))

    #matrix, matrix_terms, matrix_shape_stuff = create_matrix(poly_coeff_list, degree, dim)
    #print(time.time() - start)
    #start = time.time()
    if power:
        matrix, matrix_terms, matrix_shape_stuff = createMatrix2(initial_poly_list, degree, dim)
    else:
        matrix, matrix_terms, matrix_shape_stuff = construction(initial_poly_list, degree, dim)
    #print(time.time() - start)

    print(matrix.shape)

    matrix, matrix_terms = rrqr_reduceTelenVanBarel(matrix, matrix_terms, matrix_shape_stuff, accuracy = accuracy)

=======

    """This is the first construction option, simple monomial multiplication."""
    for i in initial_poly_list:
        poly_coeff_list = add_polys(degree, i, poly_coeff_list)
    
    """This is the second construction option, it uses the fancy triangle method that is faster but less stable."""
    #for deg in reversed(range(min([poly.degree for poly in initial_poly_list]), degree+1)):
    #    poly_coeff_list += deg_d_polys(initial_poly_list, deg, dim)
    
    #Creates the matrix for either of the above two methods. Comment out if using the third method.
    matrix, matrix_terms, matrix_shape_stuff = create_matrix(poly_coeff_list, degree, dim)
    
    """This is the thrid matrix construction option, it uses the permutation arrays, only works for power."""
    #if Power:
    #    matrix, matrix_terms, matrix_shape_stuff = createMatrix2(initial_poly_list, degree, dim)
    #else:
    #    for i in initial_poly_list:
    #        poly_coeff_list = add_polys(degree, i, poly_coeff_list)
    #    matrix, matrix_terms, matrix_shape_stuff = create_matrix(poly_coeff_list, degree, dim)
    
    matrix, matrix_terms = rrqr_reduceTelenVanBarel2(matrix, matrix_terms, matrix_shape_stuff, accuracy = accuracy)
    
>>>>>>> a45fac974b1981ec69bf2aeecef1c761d0ecf920
    height = matrix.shape[0]
    matrix[:,height:] = solve_triangular(matrix[:,:height],matrix[:,height:])
    matrix[:,:height] = np.eye(height)

    VB = matrix_terms[height:]
<<<<<<< HEAD

    #print(VB)

    #plt.plot(VB[:,0],VB[:,1], 'k.')
    #plt.show()
    basisDict = makeBasisDict(matrix, matrix_terms, VB, power, [degree]*dim)
=======
        
    basisDict = makeBasisDict(matrix, matrix_terms, VB, Power, [degree]*dim)
>>>>>>> a45fac974b1981ec69bf2aeecef1c761d0ecf920
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
<<<<<<< HEAD
    #neededList = np.vstack(neededSpots)
    #plt.plot(neededList[:,0],neededList[:,1], 'k.')
    #plt.show()

=======
    
>>>>>>> a45fac974b1981ec69bf2aeecef1c761d0ecf920
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

    Example:
        For polynomials [P1,P2,P3] with degree [d1,d2,d3] the function returns d1+d2+d3-3+1
    '''
    degree_needed = 0
    for poly in poly_list:
        degree_needed += poly.degree
    return ((degree_needed - len(poly_list)) + 1)

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
<<<<<<< HEAD

    #plt.matshow([i == 0 for i in matrix])

=======
        
    #Sorts the rows of the matrix so it is close to upper triangular.
    matrix = row_swap_matrix(matrix)
    return matrix, matrix_terms, matrix_shape_stuff

def createMatrix2(polys, degree, dim):
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

    #Get the slices needed to pull the matrix_terms from the coeff matrix.
    matrix_term_indexes = list()
    for row in matrix_terms.T:
        matrix_term_indexes.append(row)

    permutations = None
    currentDegree = 2
    #Adds the poly_coeffs to flat_polys, using added_zeros to make sure every term is in there.
    added_zeros = np.zeros(bigShape)
    parts = list()
    for poly in polys:
        print(poly.degree)
        slices = slice_top(poly.coeff)
        added_zeros[slices] = poly.coeff
        array = added_zeros[matrix_term_indexes]
        added_zeros[slices] = np.zeros_like(poly.coeff)
        
        permutations = all_permutations(degree - poly.degree, dim, degree, permutations, currentDegree)
        currentDegree = degree - poly.degree
        permList = list(permutations.values())
        print(len(permList))
        parts.append(array[np.reshape(permList, (len(permList), columns))])
    
    matrix = np.concatenate(parts)

    if matrix_shape_stuff[0] > matrix.shape[0]: #The matrix isn't tall enough, these can't all be pivot columns.
        raise TVBError("HIGHEST NOT FULL RANK. TRY HIGHER DEGREE")
    
>>>>>>> a45fac974b1981ec69bf2aeecef1c761d0ecf920
    #Sorts the rows of the matrix so it is close to upper triangular.
    matrix = row_swap_matrix(matrix)
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
<<<<<<< HEAD
    '''
    #Q,R,P = qr(matrix, pivoting = True)
    #rank = len(np.where(np.abs(R.diagonal()) > 1.e-10)[0])

    #original = matrix
    #original_terms = matrix_terms
    #Q,R,P = qr(matrix[:,:rank], pivoting = True)
    #print(R[0][0]/R[rank-1][rank-1])
    #permutations = np.arange(matrix.shape[1])

=======
    '''    
>>>>>>> a45fac974b1981ec69bf2aeecef1c761d0ecf920
    highest_num = matrix_shape_stuff[0]
    others_num = matrix_shape_stuff[1]
    xs_num = matrix_shape_stuff[2]
    #plt.matshow([i==0 for i in matrix])
    #print(highest_num)
    #RRQR reduces A and D sticking the result in it's place.
    Q1,matrix[:,:highest_num],P1 = qr(matrix[:,:highest_num], pivoting = True)
<<<<<<< HEAD

    #if abs(matrix[:,:highest_num].diagonal()[-1]) < accuracy:
    #    raise TVBError("HIGHEST NOT FULL RANK")

=======
    
    if abs(matrix[:,:highest_num].diagonal()[-1]) < accuracy:
        raise TVBError("HIGHEST NOT FULL RANK")
            
>>>>>>> a45fac974b1981ec69bf2aeecef1c761d0ecf920
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
<<<<<<< HEAD
    #original = matrix
    #Q,R,P = qr(matrix[:,:matrix.shape[0]], pivoting = True)
    #print(R[0][0]/R[-1][-1])
    #permutations = np.arange(matrix.shape[1])

    highest_num = matrix_shape_stuff[0]
    others_num = matrix_shape_stuff[1]
    xs_num = matrix_shape_stuff[2]

    C1,matrix[:highest_num,:highest_num],P1 = qr_multiply(matrix[:highest_num,:highest_num], matrix[:highest_num,highest_num:].T, mode = 'right', pivoting = True)
    matrix[:highest_num,highest_num:] = C1.T
    C1 = 0

    if abs(matrix[highest_num][highest_num]) < accuracy:
        raise TVBError("HIGHEST NOT FULL RANK")

=======
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
>>>>>>> a45fac974b1981ec69bf2aeecef1c761d0ecf920
    matrix_terms[:highest_num] = matrix_terms[:highest_num][P1]
    P1 = 0

<<<<<<< HEAD
    C,matrix[highest_num:,highest_num:highest_num+others_num],P = qr_multiply(matrix[highest_num:,highest_num:highest_num+others_num], matrix[highest_num:,highest_num+others_num:].T, mode = 'right', pivoting = True)

    matrix[highest_num:,highest_num+others_num:] = C.T
=======
    C,R,P = qr_multiply(matrix[highest_num:,highest_num:highest_num+others_num], matrix[highest_num:,highest_num+others_num:].T, mode = 'right', pivoting = True)
    matrix = matrix[:R.shape[0]+highest_num]
    matrix[highest_num:,:highest_num] = np.zeros_like(matrix[highest_num:,:highest_num])
    matrix[highest_num:,highest_num:highest_num+R.shape[1]] = R
    matrix[highest_num:,highest_num+R.shape[1]:] = C.T
>>>>>>> a45fac974b1981ec69bf2aeecef1c761d0ecf920
    C,R = 0,0

    #Shifts the columns of B.
    matrix[:highest_num,highest_num:highest_num+others_num] = matrix[:highest_num,highest_num:highest_num+others_num][:,P]
    matrix_terms[highest_num:highest_num+others_num] = matrix_terms[highest_num:highest_num+others_num][P]
    P = 0

<<<<<<< HEAD
    diag = matrix.diagonal()
    plt.matshow(matrix)

    #new = original[:,permutations]
    #Q,R,P = qr(new[:,:matrix.shape[0]], pivoting = True)
    #print(R[0][0]/R[-1][-1])

    return matrix, matrix_terms

def rrqr_reduceTelenVanBarel3(matrix, matrix_terms, matrix_shape_stuff, accuracy = 1.e-10):
    ''' Reduces a Telen Van Barel Macaulay matrix.'''
    permutations = np.arange(matrix.shape[1])

    highest_num = matrix_shape_stuff[0]
    others_num = matrix_shape_stuff[1]
    xs_num = matrix_shape_stuff[2]

    Q,R,P = qr(matrix[highest_num:highest_num+others_num,highest_num:highest_num+others_num], pivoting = True)

    permutations[highest_num:highest_num+others_num] = permutations[highest_num:highest_num+others_num][P]
    matrix_terms[highest_num:highest_num+others_num] = matrix_terms[highest_num:highest_num+others_num][P]

    matrix = matrix[:,permutations]

    R2 = np.linalg.solve(matrix[:,:matrix.shape[0]],matrix[:,matrix.shape[0]:])
    matrix = np.hstack((np.eye(matrix.shape[0]),R2))
=======
    #Get rid of 0 rows at the bottom.
    rank = np.sum(np.abs(matrix.diagonal())>accuracy)
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
    
>>>>>>> a45fac974b1981ec69bf2aeecef1c761d0ecf920
    return matrix, matrix_terms

def has_top_xs(polys):
    '''Finds out if the Macaulay Matrix will have an x^d in each dimension.

    TVB redction will work if an only if this is true. So in 2 dimensions a Macaulay matrix of degree d
    needs to have a x^d and y^d in it, in 3 dimensions it needs an x^d, y^d and z^d etc.

    Parameters
    ----------
    polys : list
        The polynomials with which the Macaulay Matrix is created.
    Returns
    -------
    has_top_xs : bool
        True if it has all the x^d's; False otherwise.
    '''
    dim = polys[0].dim

    hasXs = np.zeros(dim)
    #Make everything behind the diagonal 0,
    for poly in polys:
        deg = poly.degree

        possibleXs = set()
        for row in deg*get_var_list(dim):
            possibleXs.add(tuple(deg*np.array(row)))

        for mon in zip(*np.where(poly.coeff!=0)):
            #Checks to see if it's an x^n.
            if mon in possibleXs:
                hasXs += mon
    return np.all(hasXs)

def getDiagPoly(poly):
    '''Gets the diagonal polynomial of a polynomial.

    This is defined as only the monomials in a polynomial that are of the highest degree. Everything else is 0.

    Parameters
    ----------
    poly : Polynomial
        The polynomial of interest.
    Returns
    -------
    poly : Polynomial
        The diagonal polynomial.
    '''
    diagCoeff = poly.coeff.copy()
    deg = poly.degree
    for mon in zip(*np.where(diagCoeff!=0)):
        if np.sum(mon) != deg:
            diagCoeff[mon] = 0
    if isinstance(poly,MultiPower):
        return MultiPower(diagCoeff)
    else:
        return MultiCheb(diagCoeff)

def topDegreeMatrix(polys, degree):
    '''Gets the upper left corner of a Macaulay Matrix, the top degree part.

    Only includes the columns that are monomials of highest degree and the rows that have non-zero elements in those columns

    Parameters
    ----------
    polys : list
        The polynomials used to make the matrix.
    degree : int
        The degree of the Macaulay Matrix to be made.
    Returns
    -------
    matrix : numpy array
        The matrix.
    matrixMons : list
        A list of the monomials that were used to create the matrix. The i'th element is the monomial used to create the
        i'th row of the matrix.
    full : numpy array
        An array of zeros with the shape of the degree of the matrix in each dimension.
    '''
    dim = polys[0].dim
    power = isinstance(polys[0],MultiPower)

    diagPolys = list()
    for poly in polys:
        diagPolys.append(getDiagPoly(poly))

    diagSpots = np.vstack(mon_combosHighest([0]*dim,degree))
    diagPlaces = list()
    for i in range(dim):
        diagPlaces.append(diagSpots.T[i])

    full = np.zeros([degree+1]*dim)

    matrixRows = list()
    matrixMons = list()
    for diagPoly in diagPolys:
        mons = mon_combosHighest([0]*dim,degree - diagPoly.degree)
        matrixMons.append(mons)
        for mon in mons:
            coeff = diagPoly.mon_mult(mon, returnType = 'Matrix')
            full[slice_top(coeff)] = coeff
            matrixRows.append(full[diagPlaces])
            full[slice_top(coeff)] = np.zeros_like(coeff)
    matrix = np.vstack(matrixRows)
    return matrix, matrixMons, full

def getFPolys(fcoeffs, matrixMons, full, power):
    '''Finds the f-polynomials needed to make an S polynomial.

    Given a set of polynomials p1,p2 ... pn S = p1f1 + p2f2 + ... +pnfn. matrixMons is the monomials in the fPolys
    and fcoeffs is all the coefficients for them.

    Parameters
    ----------
    fcoeffs : numpy array
        Each entry is a coefficient in one of the fpolys. It is in the same order as the monomials in matrixMons.
    matrixMons : list
        Each entry is a list of the monomials in one f polynomial. They are in the same order as the fcoeffs.
    full : numpy array
        A matrix of zeros that is the maximum size of the coefficient matrix for an f polynomial. So each f polynomial
        starts as a copy of it and then the fcoeffs are put in the matrixMons spots.
    power : bool
        True if the fPolys should be MultiPower objects. False if they should be MultiCheb.
    Returns
    -------
    fPolys : list
        The f polynomials.
    '''
    fPolys = list()
    for mons in matrixMons:
        fCoeff = full.copy()
        for mon in mons:
            fCoeff[tuple(mon)] = fcoeffs[0]
            fcoeffs = fcoeffs[1:]
        if power:
            fPolys.append(MultiPower(fCoeff))
        else:
            fPolys.append(MultiCheb(fCoeff))
    return fPolys

def finalizeS(polys, S):
    '''Makes sure an S polynomial will make TVB work, if not, finds a new one.

    Parameters
    ----------
    polys : list
        The original polys used to make a TVB matrix.
    S : Polyomial
        A potential S polynomial to make TVB work.
    Returns
    -------
    finalizeS : Polynomail
        A polynomial that will make TVB work. If S works, S is returned, otherwise a new potential S2 is calculated
        and finalizeS is called again on S2.
    '''
    #print(S.coeff)

    if S.degree <= 0:
        raise TVBError('Polys are non-zero dimensional')

    dim = polys[0].dim
    power = isinstance(polys[0],MultiPower)
    degree = find_degree(polys)
    #print(degree)

    matrix, matrixMons, full = topDegreeMatrix(polys+list([S]), degree)
    Q,R,P = qr(matrix, pivoting = True)
    #print(R.diagonal())
    if abs(R.diagonal()[-1]) > 1.e-10:
        return S

    fPolys = getFPolys(clean_zeros_from_matrix(Q.T[-1]), matrixMons, full, power)
    if power:
        S2 = MultiPower(np.array([0]))
    else:
        S2 = MultiCheb(np.array([0]))
    for i in range(len(polys)):
        poly = polys[i]
        f = fPolys[i]
        S2 += poly*f
    S2 += S*fPolys[-1]
    S2.__init__(clean_zeros_from_matrix(S2.coeff))
    return finalizeS(polys, S2)

def get_S_Poly(polys):
    '''Gets an S polynomial if needed to make sure TVB will work.

    The code checks if an S is needed first. If so, it calculates one potential S, and then calls finalizeS on it
    to make sure that S is actually valid, and if not to find a better one.

    Parameters
    ----------
    polys : list
        The original polys used to make a TVB matrix.
    Returns
    -------
    get_S_Poly : int or Polynomial
        Returns -1 if no S-Poly is needed, menaing TVB will work fine as is. Otherwise, returns a Polynomial Object
        S that when added to the basis will make TVB work.
    '''
    dim = polys[0].dim
    power = isinstance(polys[0],MultiPower)
    degree = find_degree(polys)

    matrix, matrixMons, full = topDegreeMatrix(polys, degree)

    #print(matrix)
    Q,R,P = qr(matrix, pivoting = True)
    #print(R.diagonal())
    if abs(R.diagonal()[-1]) > 1.e-10:
        return -1 #It works fine.
    #return -1 #For testing, this breaks it.
    fPolys = getFPolys(clean_zeros_from_matrix(Q.T[-1]), matrixMons, full, power)
    if power:
        S = MultiPower(np.array([0]))
    else:
        S = MultiCheb(np.array([0]))
    for i in range(len(polys)):
        poly = polys[i]
        f = fPolys[i]
        S += poly*f
    S.__init__(clean_zeros_from_matrix(S.coeff))

    #Now make a new function to check if it's done and if not keep going.
    return finalizeS(polys, S)
