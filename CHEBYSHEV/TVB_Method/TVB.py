import numpy as np
import itertools
from scipy.linalg import qr, solve_triangular, qr_multiply
from CHEBYSHEV.TVB_Method.cheb_class import Polynomial, MultiCheb
from CHEBYSHEV.TVB_Method.cheb_utils import row_swap_matrix, clean_zeros_from_matrix, TVBError, \
                            slice_top, get_var_list, mon_combos, sort_polys_by_degree, mon_combos_highest
import time
import random
from matplotlib import pyplot as plt
from scipy.misc import comb
from math import factorial

def telen_van_barel(initial_poly_list, accuracy = 1.e-10):
    """Uses Telen and VanBarels matrix reduction method to find a vector basis for the system of polynomials.

    Parameters
    --------
    initial_poly_list: list (polynomial objects)
        The polynomials in the system we are solving.
    accuracy: float
        How small we want a number to be before assuming it is zero.

    Returns
    -----------
    basis_dict : dict
        This is a dictionary of the terms on the diagonal of the reduced TVB matrix to the
        terms in the Vector Basis.
    vector_basis : numpy array
        The terms in the vector basis, each row being a term.
    degree : int
        The degree of the Macaualy matrix that was constructed.
    """
    dim = initial_poly_list[0].dim
    poly_coeff_list = []
    degree = find_degree(initial_poly_list)

    #This sorting is required for fast matrix construction. Ascending should be False.
    initial_poly_list = sort_polys_by_degree(initial_poly_list, ascending = False)

    #matrix, matrix_terms, matrix_shape_stuff = construction(initial_poly_list, degree, dim)
    for i in initial_poly_list:
        poly_coeff_list = add_polys(degree, i, poly_coeff_list)
    matrix, matrix_terms, matrix_shape_stuff = create_matrix(poly_coeff_list, degree, dim)

    matrix, matrix_terms = rrqr_reduce_telen_van_barel(matrix, matrix_terms, matrix_shape_stuff, accuracy = accuracy)

    height = matrix.shape[0]
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
        For polynomials [P1,P2,P3] with degree [d1,d2,d3] the function returns d1+d2+d3-(number of Polynomaials)+1
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
        poly_coeff_list.append(poly.mon_mult(i, return_type = 'Matrix'))
    return poly_coeff_list

def sorted_matrix_terms(degree, dim):
    '''Finds the matrix_terms sorted in the term order needed for telen_van_barel reduction.
    So the highest terms come first,the x,y,z etc monomials last.
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
        The first entry is the number of 'highest' monomial terms. The second entry is the number of 'other' terms,
        those not in the first or third catagory. The third entry is the number of monomials of degree one of a
        single variable, as well as the monomial 1.
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

def rrqr_reduce_telen_van_barel(matrix, matrix_terms, matrix_shape_stuff, accuracy = 1.e-10):
    ''' Reduces a Telen Van Barel Macaulay matrix.

    This function does the same thing as rrqr_reduce_telen_van_barel but uses qr_multiply instead of qr and a multiplication
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
