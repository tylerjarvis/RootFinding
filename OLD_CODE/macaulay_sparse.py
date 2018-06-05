import numpy as np
import math
from scipy.linalg import lu, qr, solve_triangular, inv, solve, svd
from numpy.linalg import cond
from numalgsolve.polynomial import Polynomial, MultiCheb, MultiPower, is_power
from scipy.sparse import csr_matrix, vstack
from numalgsolve.utils import Term, row_swap_matrix, clean_zeros_from_matrix, inverse_P, triangular_solve, divides, slice_top, mon_combos
import matplotlib.pyplot as plt
from collections import defaultdict
import numalgsolve.utils as utils

def Macaulay(initial_poly_list, global_accuracy = 1.e-10):
    """
    Accepts a list of polynomials and use them to construct a Macaulay matrix.

    parameters
    --------
    initial_poly_list: list
        Polynomials for Macaulay construction.
    global_accuracy : float
        Round-off parameter: values within global_accuracy of zero are rounded to zero. Defaults to 1e-10.

    Returns
    -------
    final_polys : list
        Reduced Macaulay matrix that can be passed into the root finder.
    """
    Power = is_power(initial_poly_list)

    poly_coeff_list = []
    degree = find_degree(initial_poly_list)

    for i in initial_poly_list:
        poly_coeff_list = add_polys(degree, i, poly_coeff_list)

    matrix, matrix_terms = create_matrix(poly_coeff_list)

    plt.matshow(matrix)
    plt.show()

    #rrqr_reduce2 and rrqr_reduce same pretty matched on stability, though I feel like 2 should be better.
    matrix = utils.rrqr_reduce2(matrix, global_accuracy = global_accuracy) # here
    matrix = clean_zeros_from_matrix(matrix)
    non_zero_rows = np.sum(np.abs(matrix),axis=1) != 0
    matrix = matrix[non_zero_rows,:] #Only keeps the non_zero_polymonials

    matrix = triangular_solve(matrix)
    matrix = clean_zeros_from_matrix(matrix)

    #The other reduction option. I thought it would be really stable but seems to be the worst of the three.
    #matrix = matrixReduce(matrix, triangular_solve = True, global_accuracy = global_accuracy)

    rows = get_good_rows(matrix, matrix_terms)
    final_polys = get_polys_from_matrix(matrix, matrix_terms, rows, Power)

    return final_polys
