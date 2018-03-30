import numpy as np
from scipy.linalg import qr, solve_triangular, qr_multiply
from CHEBYSHEV.TVB_Method.cheb_class import Polynomial, MultiCheb, TVBError, slice_top, get_var_list, mon_combos, mon_combos_highest, sort_polys_by_degree


"""
    This module contains methods for constructing the TvB Matrix associated 
    to a collection of Chebyshev polynomials

 Methods in this module:

    telen_van_barel(initial_poly_list): Use the TvB matrix reduction method 
        to find a vector basis for C[x_1, ..., x_n]/I, where I is the ideal defined by 
        'initial_poly_list'.

    make_basis_dict(matrix, matrix_terms, vector_basis, remainder_shape): Calculate and 
        returns the basis_dict, which is a mapping of the terms on the diagonal of the 
        reduced TVB matrix to the terms in the vector basis. It is used to create the 
        multiplication matrix in root_finder.

    clean_zeros_from_matrix(array, accuracy): 

    find_degree(poly_list): Find the degree needed for a Macaulay/TvB Matrix.

    add_polys(degree, poly, poly_coeff_list): Adds polynomials to the Macaulay Matrix.

    sorted_matrix_terms(degree, dim): Find the matrix_terms sorted in the term order 
        needed for telen_van_barel reduction. The highest terms come first, the x,y,z etc 
        monomials last, the rest in the middle.

    create_matrix(poly_coeffs, degree, dim):
        Build a Telen Van Barel matrix with specified degree, in specified dimension.

    rrqr_reduce_telen_van_barel(matrix, matrix_terms, matrix_shape_stuff):  Reduce a 
        Macaulay matrix in the TvB way--not pivoting the highest and lowest-degree columns.

    clean_zeros_from_matrix(array, accuracy): Set all values in the array less than 
        'accuracy' to 0.

    row_swap_matrix(matrix): Rearrange the rows of a matrix so it is closer to upper traingular.

"""

def telen_van_barel(initial_poly_list, accuracy = 1.e-10):
    """Use the Telen-VanBarel matrix reduction method to find a vector basis 
       for C[x_1, ..., x_n]/I, where I is the ideal defined by initial_poly_list.

    Parameters
    --------
    initial_poly_list: list of Chebyshev polynomials
        The polynomials in the system we are solving.  These should all be the 
        same dimension (same number of variables).

    accuracy: float
        How small a number should be before assuming it is zero.

    Returns
    -----------
    basis_dict : dict
        Maps terms on the diagonal of the reduced TVB matrix to the
        terms in the vector basis.

    vector_basis : numpy array
        The terms in the vector basis, each row being a term.

    degree : int
        The degree of the Macaualy/TvB matrix that was constructed.

    """
    
    dim = initial_poly_list[0].dim #assumes all polys are the same dimension
    poly_coeff_list = []
    degree = find_degree(initial_poly_list) #find the required degree of the Macaulay matrix

    # This sorting is required for fast matrix construction. Ascending should be False.
    initial_poly_list = sort_polys_by_degree(initial_poly_list, ascending = False)

    # Construct the Macaulay matrix
    for i in initial_poly_list:
        poly_coeff_list = add_polys(degree, i, poly_coeff_list)
    matrix, matrix_terms, matrix_shape_stuff = create_matrix(poly_coeff_list, degree, dim)

    # Reduce the matrix to RREF, but leaving the top-degree and lowest-degree terms unpivoted
    matrix, matrix_terms = rrqr_reduce_telen_van_barel(matrix, matrix_terms, matrix_shape_stuff, accuracy = accuracy)
    height = matrix.shape[0]               # Number of rows
    matrix[:,height:] = solve_triangular(matrix[:,:height],matrix[:,height:])
    matrix[:,:height] = np.eye(height)

    vector_basis = matrix_terms[height:]

    basis_dict = make_basis_dict(matrix, matrix_terms, vector_basis, [degree]*dim)
    return basis_dict, vector_basis, degree

def make_basis_dict(matrix, matrix_terms, vector_basis, remainder_shape):
    '''Calculates and returns the basis_dict.

    This is a dictionary of the terms on the diagonal of the reduced TVB matrix to the terms in the Vector Basis.
    It is used to create the multiplication matrix in root_finder.

    Parameters
    --------
    matrix: numpy array
        The reduced TVB matrix.
    matrix_terms : numpy array
        The terms in the matrix. The i'th row is the term represented by the i'th column of the matrix.
    vector_basis : numpy array
        Each row is a term in the vector basis.
    remainder_shape: list
        The shape of the numpy arrays that will be mapped to in the basis_dict.

    Returns
    -----------
    basis_dict : dict
        Maps terms on the diagonal of the reduced TVB matrix (tuples) to numpy arrays of the shape remainder_shape
        that represent the terms reduction into the Vector Basis.
    '''
    basis_dict = {}

    VBSet = set()
    for i in vector_basis:
        VBSet.add(tuple(i))

    spots = list()
    for dim in range(vector_basis.shape[1]):
        spots.append(vector_basis.T[dim])

    for i in range(matrix.shape[0]):
        term = tuple(matrix_terms[i])
        remainder = np.zeros(remainder_shape)
        row = matrix[i]
        remainder[spots] = row[matrix.shape[0]:]
        basis_dict[term] = remainder

    return basis_dict

def find_degree(poly_list):
    '''Find the degree needed for a Macaulay Matrix.

    Parameters
    --------
    poly_list: list
        The polynomials used to construct the matrix.

    Returns
    -----------
    find_degree : int
        The degree of the Macaulay Matrix.

    Example:
        For polynomials [P1,P2,P3] with degree [d1,d2,d3] the function returns d1+d2+d3-(number of Polynomaials)+1
    '''

    #print('len(poly_list) = {}'.format(len(poly_list)))
    degree_needed = 0
    #print('initializing degree at {}'.format(degree_needed))
    for poly in poly_list:
        degree_needed += poly.degree
        #print('poly.degree = {}'.format(poly.degree))
        #print('degree adjusted to {}'.format(degree_needed))

    return ((degree_needed - len(poly_list)) + 1)


def add_polys(degree, poly, poly_coeff_list):
    """Adds polynomials to a Macaulay Matrix.

    This function is called on one polynomial and adds all monomial multiples of it to the matrix.

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
        The original list of polynomials in the matrix with the new 
        monomial multiplications of poly appended.
    """
    poly_coeff_list.append(poly.coeff)
    deg = degree - poly.degree
    dim = poly.dim

    mons = mon_combos([0]*dim,deg)
    for i in mons[1:]: #skips the first, all-zero (constant) monomial
        poly_coeff_list.append(poly.mon_mult(i, return_type = 'Matrix'))
        
    return poly_coeff_list

def sorted_matrix_terms(degree, dim):
    '''Find the matrix_terms sorted in the term order needed for telen_van_barel reduction.
    The highest terms come first, the x,y,z etc monomials last, the rest in the middle.
    Parameters
    ----------
    degree : int
        The degree of the TVB Matrix (degree of matrix is highest degreefound in find_degree)
    dim : int
        The dimension of the polynomials going into the matrix. (dimension = how many variables a polynomial has)
    Returns
    -------
    matrix_terms : numpy array
        The sorted matrix_terms.
    matrix_term_stuff : tuple
        The first entry is the number of 'highest' monomial terms. The second entry 
        is the number of 'other' terms, those not in the first or third catagory. 
        The third entry is the number of monomials of degree one of a single variable, 
        as well as the monomial 1.
    '''
    
    highest_mons = mon_combos_highest([0]*dim,degree)[::-1]

    other_mons = list()
    d = degree - 1
    while d > 1:
        other_mons += mon_combos_highest([0]*dim,d)[::-1]
        d -= 1

    xs_mons = mon_combos([0]*dim,1)[::-1]

    sorted_matrix_terms = np.reshape(highest_mons+other_mons+xs_mons, (len(highest_mons+other_mons+xs_mons),dim))

    return sorted_matrix_terms, tuple([len(highest_mons),len(other_mons),len(xs_mons)])

def create_matrix(poly_coeffs, degree, dim):
    ''' Build a Telen Van Barel matrix with specified degree, in specified dimension.

    Parameters
    ----------
    poly_coeffs : list of ndarrays
        The coefficients of the Chebyshev polynomials from which to build the TvB matrix.

    degree : int
        The top degree of the polynomials appearing in the TVB Matrix

    dim : int
        The dimension (number of variables) of all the polynomials appearing in the matrix.
 
    Returns
    -------
    matrix : 2D numpy array
        The Telen Van Barel matrix.
    '''

    bigShape = [degree+1]*dim
    #print('degree = {}, dim = {}, bigShape = {}'.format(degree,dim,bigShape))
    
    matrix_terms, matrix_shape_stuff = sorted_matrix_terms(degree, dim)
    
    #Get the slices needed to pull the matrix_terms from the coeff matrix.
    matrix_term_indexes = [row for row in matrix_terms.T]
    
    #Adds the poly_coeffs to flat_polys, using added_zeros to make sure every term is in there.
    added_zeros = np.zeros(bigShape)
    #print('added_zeros.shape = {}'.format(added_zeros.shape))
    flat_polys = list()
    for coeff in poly_coeffs:
        #print('coeff of poly_coeffs = {}'.format(coeff)) 
        slices = slice_top(coeff)
        #print('slices = {}'.format(slices))
        added_zeros[slices] = coeff
        flat_polys.append(added_zeros[matrix_term_indexes])
        added_zeros[slices] = np.zeros_like(coeff)
        coeff = 0
    poly_coeffs = 0

    #Make the matrix. Reshape is faster than stacking.
    matrix = np.reshape(flat_polys, (len(flat_polys),len(matrix_terms)))

    if matrix_shape_stuff[0] > matrix.shape[0]: #The matrix isn't tall enough, these can't all be pivot columns.
        raise TVBError("HIGHEST NOT FULL RANK. TRY HIGHER DEGREE")

    #Sort the rows of the matrix so it is close to upper triangular.
    matrix = row_swap_matrix(matrix)
    
    return matrix, matrix_terms, matrix_shape_stuff



def rrqr_reduce_telen_van_barel(matrix, matrix_terms, matrix_shape_stuff, accuracy = 1.e-10):
    ''' Reduces a Macaulay matrix in the TvB way--not pivoting the highest 
    and lowest-degree columns.

    This function does the same thing as rrqr_reduce_telen_van_barel but 
    uses qr_multiply instead of qr and a multiplication
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

    #Get rid of 0 rows at the bottom.
    rank = np.sum(np.abs(matrix.diagonal())>accuracy)
    matrix = matrix[:rank]
    return matrix, matrix_terms


########## Utils  ##################


def row_swap_matrix(matrix):
    '''Rearrange the rows of a matrix so it is close to upper traingular.

    Parameters
    ----------
    matrix : 2D numpy array
        The matrix whose rows need to be switched

    Returns
    -------
    2D numpy array
        The same matrix but with the rows changed so it is closer to upper
        triangular

    Examples
    --------
    >>> row_swap_matrix(np.array([[0,2,0,2],[0,1,3,0],[1,2,3,4]]))
    array([[1, 2, 3, 4],
           [0, 2, 0, 2],
           [0, 1, 3, 0]])
    '''
    leading_mon_columns = list()
    for row in matrix:
        leading_mon_columns.append(np.where(row!=0)[0][0])

    return matrix[np.argsort(leading_mon_columns)]


def clean_zeros_from_matrix(array, accuracy=1.e-10):
    '''Sets all values in the array less than the given accuracy to 0.

    Parameters
    ----------
    array : numpy array
    accuracy : float, optional
        Values in the matrix less than this will be set to 0.

    Returns
    -------
    array : numpy array
        Same array, but with values less than the given accuracy set to 0.
    '''
    array[(array < accuracy) & (array > -accuracy)] = 0
    return array


