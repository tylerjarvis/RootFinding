import pytest
import numpy as np
import sympy as sy
from groebner import utils as ut
from groebner.utils import *
from groebner.polynomial import MultiCheb, MultiPower
from scipy.linalg import qr, solve_triangular

def test_inverse_P():
    # Test Case 1:
    # Create matrix
    M = np.array([[0, 1, 2, 3, 4, 5],
       [0, 1, 2, 3, 4, 5],
       [0, 1, 2, 3, 4, 5],
       [0, 1, 2, 3, 4, 5]])

    # Order of Column flips.
    p = [1,4,3,2,0,5]

    # N is the matrix with the columns flipped.
    N = M[:,p]

    # Find the order of columns to flip back to.
    pt = inverse_P(p)
    print(pt)
    # Result done by hand.
    pt_inv = [4,0,3,2,1,5]
    assert(np.allclose(M,N[:,pt])), "Matrix are not the same."
    assert(np.all(pt == pt_inv)), "Wrong matrix order."


    # Test Case 2:
    A = np.random.random((5,10))

    Q,R,P = qr(A,pivoting=True)

    pt = inverse_P(P)

    # We know that A[:,p] = QR, want to see if pt flips QR back to A.
    assert(np.allclose(A,(Q@R)[:,pt]))

def test_triangular_solve():
    """This tests the triangular_solve() method.
    A visual graph of zeroes on the diagonal was also used to test this function.
    """
    # Simple Test Case.
    A = np.array([[1, 2, 3, 4, 5],
                  [0, 1, 2, 3, 4],
                  [0, 0, 0, 1, 2]])

    matrix = triangular_solve(A)
    answer = np.array([[ 1.,  0., -1.,  0.,  1.],
                       [ 0.,  1.,  2.,  0., -2.],
                       [ 0.,  0.,  0.,  1.,  2.]])
    assert(np.allclose(matrix,answer))

    # Randomize test case: square matrix.
    A = np.random.random((50,50))
    Q,R,p = qr(A,pivoting=True)
    diagonal = np.diag(R)
    r = np.sum(np.abs(diagonal)>1e-10)
    matrix = R[:r]
    new_matrix = triangular_solve(matrix)
    true=sy.Matrix(new_matrix).rref()
    x = sy.symbols('x')
    f = sy.lambdify(x,true[0])
    assert(np.allclose(new_matrix,f(1)))

    # Randomize test case: shorter rows than column.
    A = np.random.random((10,50))
    Q,R,p = qr(A,pivoting=True)
    diagonal = np.diag(R)
    r = np.sum(np.abs(diagonal)>1e-10)
    matrix = R[:r]
    new_matrix = triangular_solve(matrix)
    true=sy.Matrix(new_matrix).rref()
    x = sy.symbols('x')
    f = sy.lambdify(x,true[0])
    print(f(1))
    print(new_matrix)
    assert(np.allclose(new_matrix,f(1)))

    # Randomize test case: longer rows than columns.
    A = np.random.random((50,10))
    Q,R,p = qr(A,pivoting=True)
    diagonal = np.diag(R)
    r = np.sum(np.abs(diagonal)>1e-10)
    matrix = R[:r]
    new_matrix = triangular_solve(matrix)
    true=sy.Matrix(new_matrix).rref()
    x = sy.symbols('x')
    f = sy.lambdify(x,true[0])
    print(f(1))
    print(new_matrix)
    assert(np.allclose(new_matrix,f(1)))

def test_sorted_polys_monomial():
    A = MultiPower(np.array([[3,0,-5,0,0,4,2,-6,3,-6],
                         [-2,0,-1,1,-1,4,2,-6,-5,-2],
                         [-1,3,2,-2,0,4,-1,-2,-4,6],
                         [4,2,5,9,0,3,2,-1,-3,-3],
                         [3,-3,-5,-2,0,4,-2,2,1,-6]]))
    x = sorted_polys_monomial([A])
    assert(A == x[0])

    B = MultiPower(np.array([[2,0,-3,0,0],
                         [0,1,0,0,0],
                         [-2,0,0,0,0],
                         [0,0,4,0,0],
                         [0,0,0,0,-2]]))
    assert(list((B,A)) == sorted_polys_monomial([B,A]))

    C = MultiPower(np.array([[0,0,-3,0,0],
                         [0,0,0,0,0],
                         [0,0,0,0,0],
                         [0,0,4,0,0],
                         [0,0,0,0,-2]]))
    assert(list((C,B,A)) == sorted_polys_monomial([A,B,C]))


    D = MultiPower(np.array([[2,0,-3,0,0],
                         [0,1,0,0,0],
                         [-2,0,2,0,0],
                         [0,0,4,0,0],
                         [0,0,0,0,-2]]))
    assert(list((C,B,D,A)) == sorted_polys_monomial([B,D,C,A]))

    E = MultiPower(np.array([[3,0,-5,0,0,4,2,-6,3,-6],
                         [-2,0,-1,1,-1,4,2,-6,-5,-2],
                         [-1,3,2,-2,0,4,-1,-2,-4,6],
                         [4,2,5,9,0,3,2,-1,-3,-3],
                         [3,-3,-5,-2,2,4,-2,2,1,-6]]))
    assert(list((C,B,D,A,E)) == sorted_polys_monomial([B,D,E,C,A]))

    F = MultiPower(np.array([[3,0,-5,0,0,0,2,-6,3,-6],
                         [-2,0,-1,1,-1,4,2,-6,-5,-2],
                         [-1,3,2,-2,0,4,-1,-2,-4,6],
                         [4,2,5,9,0,3,2,-1,-3,-3],
                         [3,-3,-5,-2,0,4,-2,2,1,-6]]))
    assert(list((C,B,D,F,A,E)) == sorted_polys_monomial([F,B,D,E,C,A]))

def test_sorted_polys_coeff():
    A = MultiPower(np.array([[2,0,-3,0,0],
                         [0,1,0,0,0],
                         [-2,0,0,0,0],
                         [0,0,4,0,0],
                         [0,0,0,0,-2]]))
    B = MultiPower(np.array([[3,0,-5,0,0,4,2,-6,3,-6],
                         [-2,0,-1,1,-1,4,2,-6,-5,-2],
                         [-1,3,2,-2,0,4,-1,-2,-4,6],
                         [4,2,5,9,0,3,2,-1,-3,-3],
                         [3,-3,-5,-2,0,4,-2,2,1,-6]]))

    assert([A,B] == ut.sorted_polys_coeff([A,B]))

    C = MultiPower(np.array([[1]]))
    assert([C,A,B] == ut.sorted_polys_coeff([A,B,C]))

def test_makePolyCoeffMatrix():
    A = MultiPower('1')
    B = MultiPower(np.array([1]))
    assert (A.coeff==B.coeff).all()
    
    A = MultiPower('2x0+x1+x0*x1')
    B = MultiPower(np.array([[0,2],[1,1]]))
    assert (A.coeff==B.coeff).all()

    A = MultiPower('-4.7x0*x1+2x1+5x0+-3')
    B = MultiPower(np.array([[-3,5],[2,-4.7]]))
    assert (A.coeff==B.coeff).all()

    A = MultiPower('x0^2+-x1^2')
    B = MultiPower(np.array([[0,0,1],[0,0,0],[-1,0,0]]))
    assert (A.coeff==B.coeff).all()

    A = MultiPower('x0+2x1+3x2+4x3')
    B = MultiPower(np.array(
        [[[[0,1],[2,0]],
          [[3,0],[0,0]]],
         [[[4,0],[0,0]],
          [[0,0],[0,0]]]]))
    assert (A.coeff==B.coeff).all()
    
    A = MultiPower('1+x0')
    B = MultiPower('1+x1')
    A1 = MultiPower(np.array([1,1]))
    B1 = MultiPower(np.array([[1],[1]]))
    
    assert (A.coeff==A1.coeff).all() and (B.coeff==B1.coeff).all()
