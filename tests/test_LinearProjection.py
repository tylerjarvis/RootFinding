import numpy as np
from yroots.LinearProjection import remove_linear, project_down, bounding_parallelepiped, proj_approximate_nd
from yroots.polynomial import Polynomial, MultiCheb, MultiPower, getPoly
from yroots.MacaulayReduce import find_degree, mon_combos
from yroots import polyroots as pr
from yroots.utils import InstabilityWarning, arrays
from yroots.Multiplication import create_matrix
from itertools import product
import unittest
import warnings
import yroots.subdivision as sbd

def correctZeros(original_polys, new_polys, transform, MSmatrix):
    '''
    A helper function for polyroots tests. Takes in polynomials, find their common zeros using polyroots, and calculates
    how many of the zeros are correct.
    In this function it asserts that the number of zeros is equal to the product of the degrees, which is only valid if
    the polynomials are random and upper triangular, and that at least 95% of the zeros are correct (so it will pass even
    on bad random runs)
    '''
    zeros = transform(pr.solve(new_polys, MSmatrix = MSmatrix))
    correct = 0
    outOfRange = 0
    for zero in zeros:
        good = True
        for poly in original_polys:
            if not np.isclose(0, poly(zero), atol = 1.e-3):
                good = False
                if (np.abs(zero) > 1).any():
                    outOfRange += 1
                break
        if good:
            correct += 1
    assert(100*correct/(len(zeros)-outOfRange) > 95)

def test_bounding_parallelepiped():
    num_test_cases = 10

    np.random.seed(31)
    A = getPoly(1, 2, True)
    p0,edges = bounding_parallelepiped(A.coeff)
    rand = np.random.rand(edges.shape[1], num_test_cases)
    pts = np.dot(edges, rand).T + p0
    assert np.allclose(A(pts), 0)

    A = getPoly(1, 3, True)
    p0,edges = bounding_parallelepiped(A.coeff)
    rand = np.random.rand(edges.shape[1], num_test_cases)
    pts = np.dot(edges, rand).T + p0
    assert np.allclose(A(pts), 0)

    A = getPoly(1, 6, True)
    p0,edges = bounding_parallelepiped(A.coeff)
    rand = np.random.rand(edges.shape[1], num_test_cases)
    pts = np.dot(edges, rand).T + p0
    assert np.allclose(A(pts), 0)

# def test_project_down():
#     num_test_cases = 10

#     np.random.seed(821)
#     linear = getPoly(1, 2, True)
#     A = getPoly(3, 2, True)
#     (A_prj,), T = project_down([A],linear.coeff, 1e-4, 1e-8)
#     A_prj = MultiCheb(A_prj)
#     pts = np.random.rand(num_test_cases, linear.dim-1)
#     assert np.allclose(A(T(pts)), A_prj(pts))

#     linear = getPoly(1, 3, False)
#     A = getPoly(10, 3, False)
#     B = getPoly(10, 3, False)
#     (A_prj, B_prj), T = project_down([A,B],linear.coeff, 1e-4, 1e-8)
#     A_prj, B_prj = map(MultiCheb, (A_prj, B_prj))
#     pts = np.random.rand(num_test_cases, linear.dim-1)
#     assert np.allclose(A(T(pts)), A_prj(pts))
#     assert np.allclose(B(T(pts)), B_prj(pts))

#     linear = getPoly(1, 5, True)
#     A = getPoly(3, 5, True)
#     B = getPoly(3, 5, True)
#     C = getPoly(3, 5, True)
#     D = getPoly(3, 5, True)
#     (A_prj, B_prj, C_prj, D_prj), T = project_down([A,B,C,D],linear.coeff, 1e-4, 1e-8)
#     A_prj, B_prj, C_prj, D_prj = map(MultiCheb, (A_prj, B_prj, C_prj, D_prj))
#     pts = np.random.rand(num_test_cases, linear.dim-1)
#     assert np.allclose(A(T(pts)), A_prj(pts))
#     assert np.allclose(B(T(pts)), B_prj(pts))
#     assert np.allclose(C(T(pts)), C_prj(pts))
#     assert np.allclose(D(T(pts)), D_prj(pts))

# def test_remove_linear():
#     linear = getPoly(1, 3, False)
#     A = getPoly(4, 3, False)
#     B = getPoly(4, 3, False)
#     (A_prj, B_prj), T, is_projected = remove_linear([A, B, linear], 1e-4, 1e-8)
#     assert is_projected == True
#     correctZeros([A, B], [A_prj, B_prj], T, 0)
#     correctZeros([A, B], [A_prj, B_prj], T, -1)


if __name__ == "__main__":
    # test_bounding_parallelepiped()
    # test_project_down()
    test_remove_linear()
