import pytest
import numpy as np
import sympy as sy
from groebner import utils as ut
from groebner.utils import triangular_solve, inverse_P, MaxHeap, Term
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
    # Result done by hand. 
    pt_inv = [4,0,3,2,1,5]
    assert(np.allclose(M,N[:,pt])), "Matrix are not the same."
    assert(all(pt == pt_inv)), "Wrong matrix order."
    
    
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

def test_push_pop():
    a0 = Term((0,0,1,0,0))
    a1 = Term((0,1,1,3,1))
    a2 = Term((0,1,1,3,0,0,0,1))
    a3 = Term((2,2,2,3,4,1,4,3))
    a4 = Term((0,1,1,2,2))
    maxh = MaxHeap()
    maxh.heappush(a1)
    maxh.heappush(a3)
    maxh.heappush(a0)
    maxh.heappush(a2)
    maxh.heappush(a4)
    assert maxh.heappop() == a3
    assert maxh.heappop() == a2

    maxh.heappush(a3)
    maxh.heappush(a3)

    assert maxh.heappop() == a3
    assert maxh.heappop() == a1
    assert maxh.heappop() == a4
    assert maxh.heappop() == a0

