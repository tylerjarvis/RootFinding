import numpy as np
from itertools import permutations
from groebner import maxheap
from groebner.multi_power import MultiPower
from groebner.multi_cheb import MultiCheb
from groebner.groebner_class import Groebner
import pytest
from scipy.linalg import qr
import sympy as sy
#write more tests

def test_reduce_matrix():
    poly1 = MultiPower(np.array([[1., 0.],[0., 1.]]))
    poly2 = MultiPower(np.array([[0., 0.],[1., 0.]]))
    poly3 = MultiPower(np.array([[1., 0.],[12., -5.]]))
    grob = Groebner([])
    grob.new_polys = list((poly1, poly2, poly3))
    grob.matrix_terms = []
    grob.np_matrix = np.array([])
    grob.term_set = set()
    grob.lead_term_set = set()
    grob._add_polys(grob.new_polys)
    grob.create_matrix()
    
    assert(grob.reduce_matrix())
    #assert(len(grob.old_polys) == 2)
    assert(len(grob.new_polys) == 1)

    poly1 = MultiPower(np.array([[1., 0.],[0., 0.]]))
    poly2 = MultiPower(np.array([[0., 0.],[1., 0.]]))
    poly3 = MultiPower(np.array([[1., 0.],[12., 0.]]))
    grob = Groebner([])
    grob.new_polys = list((poly1, poly2, poly3))
    grob.matrix_terms = []
    grob.np_matrix = np.array([])
    grob.term_set = set()
    grob.lead_term_set = set()
    grob._add_polys(grob.new_polys)
    grob.create_matrix()

    assert(not grob.reduce_matrix())
    #assert(len(grob.old_polys) == 3)
    assert(len(grob.new_polys) == 0)

    poly1 = MultiPower(np.array([[1., -14.],[0., 2.]]))
    poly2 = MultiPower(np.array([[0., 3.],[1., 6.]]))
    poly3 = MultiPower(np.array([[1., 0.],[12., -5.]]))
    grob = Groebner([])
    grob.new_polys = list((poly1, poly2, poly3))
    grob.matrix_terms = []
    grob.np_matrix = np.array([])
    grob.term_set = set()
    grob.lead_term_set = set()
    
    grob._add_polys(grob.new_polys)
    grob.create_matrix()
    assert(grob.reduce_matrix())
    #assert(len(grob.old_polys) == 3)
    assert(len(grob.new_polys) == 2)

def test_solve():
    #First Test
    A = MultiPower(np.array([[-10,0],[0,1],[1,0]]))
    B = MultiPower(np.array([[-26,0,0],[0,0,1],[0,0,0],[1,0,0]]))
    C = MultiPower(np.array([[-70,0,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,0]]))
    grob = Groebner([A,B,C])
    X = MultiPower(np.array([[-2.],[ 1.]]))
    Y = MultiPower(np.array([[-3.,1.]]))
    x1, y1 = grob.solve()
    assert(np.any([X==i and Y==j for i,j in permutations((x1,y1),2)]))

    #Second Test
    A = MultiPower(np.array([
                         [[[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]],
                         [[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]
                        ]
                         ))
    B = MultiPower(np.array([
                         [[[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]],
                         [[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]
                        ]
                         ))
    C = MultiPower(np.array([
                         [[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]],
                         [[[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]
                        ]
                         ))
    grob = Groebner([A,B,C])
    w1, x1, y1, z1 = grob.solve()

    W = MultiPower(np.array([[[[ 0.],[ 0.]],[[ 0.],[ 0.]],[[ 0.],[ 0.]],[[ 1.],[ 0.]]],
                             [[[ 0.],[-1.]],[[ 0.],[ 0.]],[[ 0.],[ 0.]],[[ 0.],[ 0.]]]]))
    X = MultiPower(np.array([[[[ 0.,0.,0.,0.,0.,1.],[-1.,0.,0.,0.,0.,0.]]]]))
    Y = MultiPower(np.array([[[[ 0.],[ 0.],[ 1.]],[[-1.],[ 0.],[ 0.]]]]))
    Z = MultiPower(np.array([[[[ 0.],[ 0.]],[[ 0.],[ 0.]],[[ 0.],[ 1.]]],
                             [[[-1.],[ 0.]],[[ 0.],[ 0.]],[[ 0.],[ 0.]]]]))

    assert(np.any([W==i and X==j and Y==k and Z==l for i,j,k,l in permutations((w1,x1,y1,z1),4)]))

    #Third Test
    A = MultiPower(np.array([[-1,0,1],[0,0,0]]))
    B = MultiPower(np.array([[-1,0,0],[0,1,0],[1,0,0]]))
    grob = Groebner([A,B])
    x1, y1 = grob.solve()
    assert(np.any([A==i and B==j for i,j in permutations((x1,y1),2)]))

    #Fourth Test
    A = MultiPower(np.array([[-10,0],[0,1],[1,0]]))
    B = MultiPower(np.array([[-25,0,0],[0,0,1],[0,0,0],[1,0,0]]))
    C = MultiPower(np.array([[-70,0,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,0]]))
    grob = Groebner([A,B,C])
    X = MultiPower(np.array([[1.]]))
    x1 = grob.solve()
    assert(X == x1[0])

    #Fifth Test
    A = MultiPower(np.array([[1,1],[0,0]]))
    B = MultiPower(np.array([[1,0],[1,0]]))
    C = MultiPower(np.array([[1,0],[1,0],[0,1]]))
    grob = Groebner([A,B,C])
    X = MultiPower(np.array([[1.]]))
    x1 = grob.solve()
    assert(X == x1[0])
    
    
def test_phi_criterion():    
    # Same as grob.solve(), but added true/false to test the phi's. 
    # *WARNING* MAKE SURE TO CHANGE solve_phi() test to match .solve method always! 
    def solve_phi(grob,phi=True):
        polys = True
        i = 1
        while polys:
            print("Starting Loop #"+str(i))
            grob.initialize_np_matrix()
            grob.add_phi_to_matrix(phi)
            grob.add_r_to_matrix()
            grob.create_matrix()
            print(grob2.np_matrix.shape)
            polys = grob.reduce_matrix()
            i+=1
        print("WE WIN")
        grob.get_groebner()
        print("Basis - ")
        
        grob.reduce_groebner_basis()
        
        for poly in grob.groebner_basis:
            print(poly.coeff)
        
        return grob.groebner_basis
    
    
    # Simple Test Case (Nothing gets added )
    A = MultiPower(np.array([[-1,0,1],[0,0,0]]))
    B = MultiPower(np.array([[-1,0,0],[0,1,0],[1,0,0]]))
    grob1 = Groebner([A,B])
    grob2 = Groebner([A,B])
    
    x1,y1 = solve_phi(grob1,True)
    x2,y2 = solve_phi(grob2,False)
            
    assert(np.any([x2==i and y2==j for i,j in permutations((x1,y1),2)])), "Not the same basis!"
    
            
    #Second Test
    A = MultiPower(np.array([
                         [[[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]],
                         [[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]
                        ]
                         ))
    B = MultiPower(np.array([
                         [[[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]],
                         [[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]
                        ]
                         ))
    C = MultiPower(np.array([
                         [[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]],
                         [[[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                         [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]
                        ]
                         ))
    grob1 = Groebner([A,B,C])
    grob2 = Groebner([A,B,C])    


    w1, x1, y1, z1 = solve_phi(grob1,True)
    w2, x2, y2, z2 = solve_phi(grob2,False)
  
    assert(np.any([w2==i and x2==j and y2==k and z2==l for i,j,k,l in permutations((w1,x1,y1,z1),4)]))

     #Third Test
    A = MultiPower(np.array([[-1,0,1],[0,0,0]]))
    B = MultiPower(np.array([[-1,0,0],[0,1,0],[1,0,0]]))
    grob1 = Groebner([A,B])
    grob2 = Groebner([A,B])
    x1, y1 = solve_phi(grob1,True)
    x2, y2 = solve_phi(grob2,False)
   
    assert(np.any([A==i and B==j for i,j in permutations((x1,y1),2)]))

    #Fourth Test
    A = MultiPower(np.array([[-10,0],[0,1],[1,0]]))
    B = MultiPower(np.array([[-25,0,0],[0,0,1],[0,0,0],[1,0,0]]))
    C = MultiPower(np.array([[-70,0,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,0]]))
    grob1 = Groebner([A,B,C])
    grob2 = Groebner([A,B,C])
    x1 = solve_phi(grob1,True)
    x2 = solve_phi(grob2,False)
    
    assert(x2[0] == x1[0])
    

    #Fifth Test
    A = MultiPower(np.array([[1,1],[0,0]]))
    B = MultiPower(np.array([[1,0],[1,0]]))
    C = MultiPower(np.array([[1,0],[1,0],[0,1]]))
    grob1 = Groebner([A,B,C])
    grob2 = Groebner([A,B,C])
    x1 = solve_phi(grob1,True)
    x2 = solve_phi(grob2,False)
    assert(x2[0]== x1[0])
        
def test_inverse_P():
    
    # Simple Test Case. 
    C = MultiPower(np.array([[-1,0,1],[0,0,0]]))
    D = MultiPower(np.array([[-1,0,0],[0,1,0],[1,0,0]]))
    # Creating a random object to run tests. 
    grob = Groebner([C,D])
    
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
    pt = grob.inverse_P(p)
    # Result done by hand. 
    pt_inv = [4,0,3,2,1,5]
    assert(np.allclose(M,N[:,pt])), "Matrix are not the same."
    assert(all(pt == pt_inv)), "Wrong matrix order."
    
    
    # Test Case 2: 
    A = np.random.random((5,10))
    
    Q,R,p = qr(A,pivoting=True)
    
    pt = grob.inverse_P(p)
    
    # We know that A[:,p] = QR, want to see if pt flips QR back to A. 
    assert(np.allclose(A,np.dot(Q,R)[:,pt]))

def test_triangular_solve():
    """This tests the triangular_solve() method. 
    A visual graph of zeroes on the diagonal was also used to test this function.
    """
    
    # Simple Test Case. 
    M = MultiPower(np.array([[-1,0,1],[0,0,0]]))
    N = MultiPower(np.array([[-1,0,0],[0,1,0],[1,0,0]]))
    # Creating a random object to run tests. 
    grob = Groebner([M,N])
    
    A = np.array([[1, 2, 3, 4, 5],
                  [0, 1, 2, 3, 4],
                  [0, 0, 0, 1, 2]])

    matrix = grob.triangular_solve(A)
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
    new_matrix = grob.triangular_solve(matrix)
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
    new_matrix = grob.triangular_solve(matrix)
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
    new_matrix = grob.triangular_solve(matrix)
    true=sy.Matrix(new_matrix).rref()
    x = sy.symbols('x')
    f = sy.lambdify(x,true[0])
    print(f(1))
    print(new_matrix)
    assert(np.allclose(new_matrix,f(1)))



def test_init_():
    C = MultiPower(np.array([[-1,0,1],[0,0,0]]))
    D = MultiCheb(np.array([[-1,0,0],[0,1,0],[1,0,0]]))
    with pytest.raises(ValueError):
        grob = Groebner([C,D])
    pass

def test_sorted_polys_monomial():
    A = MultiPower(np.array([[3,0,-5,0,0,4,2,-6,3,-6],
                         [-2,0,-1,1,-1,4,2,-6,-5,-2],
                         [-1,3,2,-2,0,4,-1,-2,-4,6],
                         [4,2,5,9,0,3,2,-1,-3,-3],
                         [3,-3,-5,-2,0,4,-2,2,1,-6]]))
    grob = Groebner([A])
    x = grob.sorted_polys_monomial([A])
    assert(A == x[0])
    
    B = MultiPower(np.array([[2,0,-3,0,0],
                         [0,1,0,0,0],
                         [-2,0,0,0,0],
                         [0,0,4,0,0],
                         [0,0,0,0,-2]]))
    assert(list((B,A)) == grob.sorted_polys_monomial([B,A]))
    
    C = MultiPower(np.array([[0,0,-3,0,0],
                         [0,0,0,0,0],
                         [0,0,0,0,0],
                         [0,0,4,0,0],
                         [0,0,0,0,-2]]))
    assert(list((C,B,A)) == grob.sorted_polys_monomial([A,B,C]))
    
    
    D = MultiPower(np.array([[2,0,-3,0,0],
                         [0,1,0,0,0],
                         [-2,0,2,0,0],
                         [0,0,4,0,0],
                         [0,0,0,0,-2]]))
    assert(list((C,B,D,A)) == grob.sorted_polys_monomial([B,D,C,A]))
    
    E = MultiPower(np.array([[3,0,-5,0,0,4,2,-6,3,-6],
                         [-2,0,-1,1,-1,4,2,-6,-5,-2],
                         [-1,3,2,-2,0,4,-1,-2,-4,6],
                         [4,2,5,9,0,3,2,-1,-3,-3],
                         [3,-3,-5,-2,2,4,-2,2,1,-6]]))
    assert(list((C,B,D,A,E)) == grob.sorted_polys_monomial([B,D,E,C,A]))
    
    F = MultiPower(np.array([[3,0,-5,0,0,0,2,-6,3,-6],
                         [-2,0,-1,1,-1,4,2,-6,-5,-2],
                         [-1,3,2,-2,0,4,-1,-2,-4,6],
                         [4,2,5,9,0,3,2,-1,-3,-3],
                         [3,-3,-5,-2,0,4,-2,2,1,-6]]))
    assert(list((C,B,D,F,A,E)) == grob.sorted_polys_monomial([F,B,D,E,C,A]))
    pass

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
    grob = Groebner([A,B])
    assert(list((A,B)) == grob.sorted_polys_coeff())
    
    C = MultiPower(np.array([[1]]))
    grob = Groebner([A,B,C])
    assert(list((A,B,C)) == grob.sorted_polys_coeff())

    
    
    
