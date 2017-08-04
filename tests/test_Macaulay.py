import numpy as np
from groebner.Macaulay import Macaulay, add_polys, create_matrix
from groebner.polynomial import MultiCheb, MultiPower
from groebner.root_finder import roots
import pytest
import random

def test_Macaulay():
    #test 1 - compare Groebner results and Macaulay results with Chebyshev
    for i in range(5):
        ACoeff = np.random.rand(3,3)
        BCoeff = np.random.rand(3,3)
        for i,j in np.ndenumerate(ACoeff):
            if np.sum(i) > 3:
                ACoeff[i] = 0
                BCoeff[i] = 0
        A = MultiCheb(ACoeff)
        B = MultiCheb(BCoeff)

        A1 = MultiPower(ACoeff)
        B1 = MultiPower(BCoeff)

        zeros_from_Macaulay = roots([A,B], 'Macaulay')
        zeros_from_Groebner = roots([A,B], 'Groebner')

        zeros_from_Macaulay1 = roots([A1,B1], 'Macaulay')
        zeros_from_Groebner1 = roots([A1,B1], 'Groebner')

        sorted_from_Macaulay = np.sort(zeros_from_Macaulay, axis = 0)
        sorted_from_Groebner = np.sort(zeros_from_Groebner, axis = 0)

        sorted_from_Macaulay1 = np.sort(zeros_from_Macaulay1, axis = 0)
        sorted_from_Groebner1 = np.sort(zeros_from_Groebner1, axis = 0)

        assert np.allclose(sorted_from_Macaulay, sorted_from_Groebner)
        assert np.allclose(sorted_from_Macaulay1, sorted_from_Groebner1)

    #test 2 - same as test 1 but in the Power basis
    #for i in range(5):





"""
def test_fullRank():

def test_triangular_solve():

def test_get_poly_from_matrix():

def test_divides():

def test_get_good_rows():

def test_find_degree():

def test_mon_combos():

def test_add_polys():

def test_row_swap_matrix():

def test_fill_size():

def test_sort_matrix():

def test_in_basis():

def test_sort_matrixTelenVanBarel():

def test_clean_matrix():

def test_create_matrix():

def test_create_matrix2():

def test_rrqr_reduce():

def test_inverse_P():

def test_clean_zeros_from_matrix():

def test_rrqr_reduce2():

def test_rrqr_reduceTelenVanBarel():
"""
