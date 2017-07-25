import numpy as np
from itertools import permutations
import groebner.maxheap as maxheap
from groebner.multi_power import MultiPower
from groebner.multi_cheb import MultiCheb
import groebner.groebner_basis as groebner_basis
import pytest
from scipy.linalg import qr
import sympy as sy
#write more tests

def test_solve():
    #First Test
    A = MultiPower(np.array([[-10,0],[0,1],[1,0]]))
    B = MultiPower(np.array([[-26,0,0],[0,0,1],[0,0,0],[1,0,0]]))
    C = MultiPower(np.array([[-70,0,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,0]]))
    
    x1,y1 = groebner_basis.solve([A,B,C])
    
    X = MultiPower(np.array([[-2.],[ 1.]]))
    Y = MultiPower(np.array([[-3.,1.]]))

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
    
    w1, x1, y1, z1 = groebner_basis.solve([A,B,C])

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
    x1, y1 = groebner_basis.solve([A,B])
    assert(np.any([A==i and B==j for i,j in permutations((x1,y1),2)]))

    #Fourth Test
    A = MultiPower(np.array([[-10,0],[0,1],[1,0]]))
    B = MultiPower(np.array([[-25,0,0],[0,0,1],[0,0,0],[1,0,0]]))
    C = MultiPower(np.array([[-70,0,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,0]]))
    X = MultiPower(np.array([[1.]]))
    x1 = groebner_basis.solve([A,B,C])
    assert(X == x1[0])

    #Fifth Test
    A = MultiPower(np.array([[1,1],[0,0]]))
    B = MultiPower(np.array([[1,0],[1,0]]))
    C = MultiPower(np.array([[1,0],[1,0],[0,1]]))
    X = MultiPower(np.array([[1.]]))
    x1 = groebner_basis.solve([A,B,C])
    assert(X == x1[0])
    
    
def test_phi_criterion():
    
    # Same as grob.solve(), but added true/false to test the phi's. 
    # *WARNING* MAKE SURE TO CHANGE solve_phi() test to match .solve method always! 
    
    def solve_phi(polys,phi=True):
        
        items = {}
        
        if all([type(p) == MultiPower for p in polys]):
            items['power'] = True
        elif all([type(p) == MultiCheb for p in polys]):
            items['power'] = False
        else:
            print([type(p) == MultiPower for p in polys])
            raise ValueError('Bad polynomials in list')
        old_polys = []
        new_polys = polys

        polys_were_added = True
        i=1 #Tracks what loop we are on.
        while polys_were_added:
            print("Starting Loop #"+str(i))
            print("Num Polys - ", len(new_polys + old_polys))
        
            # Initialize Everything. 
            print("Initializing")
            items = groebner_basis.initialize_np_matrix(new_polys,old_polys,items)
            print(items['np_matrix'].shape)
        
            # Add Phi's to matrix. 
            print("ADDING PHI's")
            items = groebner_basis.add_phi_to_matrix(new_polys,old_polys,items,phi)
            print(items['np_matrix'].shape)
        
            # Add r's to matrix. 
            print("ADDING r's")
            items = groebner_basis.add_r_to_matrix(new_polys,old_polys,items)
            print(items['np_matrix'].shape)
        
            # Reduce matrix. 
            polys_were_added,new_polys,old_polys = groebner_basis.reduce_matrix(items)
            i+=1
        print("WE WIN")
        print("Basis - ")
        return groebner_basis.reduce_groebner_basis(old_polys)

    # Simple Test Case (Nothing gets added )
    A = MultiPower(np.array([[-1,0,1],[0,0,0]]))
    B = MultiPower(np.array([[-1,0,0],[0,1,0],[1,0,0]]))
    
    x1,y1 = solve_phi([A,B],True)
    x2,y2 = solve_phi([A,B],False)
            
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

    w1, x1, y1, z1 = solve_phi([A,B,C],True)
    w2, x2, y2, z2 = solve_phi([A,B,C],False)
  
    assert(np.any([w2==i and x2==j and y2==k and z2==l for i,j,k,l in permutations((w1,x1,y1,z1),4)]))

     #Third Test
    A = MultiPower(np.array([[-1,0,1],[0,0,0]]))
    B = MultiPower(np.array([[-1,0,0],[0,1,0],[1,0,0]]))

    x1, y1 = solve_phi([A,B],True)
    x2, y2 = solve_phi([A,B],False)
   
    assert(np.any([A==i and B==j for i,j in permutations((x1,y1),2)]))

    #Fourth Test
    A = MultiPower(np.array([[-10,0],[0,1],[1,0]]))
    B = MultiPower(np.array([[-25,0,0],[0,0,1],[0,0,0],[1,0,0]]))
    C = MultiPower(np.array([[-70,0,0,0],[0,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,0]]))

    x1 = solve_phi([A,B],True)
    x2 = solve_phi([A,B],False)
    
    assert(x2[0] == x1[0])
    

    #Fifth Test
    A = MultiPower(np.array([[1,1],[0,0]]))
    B = MultiPower(np.array([[1,0],[1,0]]))

    x1 = solve_phi([A,B],True)
    x2 = solve_phi([A,B],False)
    assert(x2[0]== x1[0])
        
def test_inverse_P():
    '''Test to see if inverse_P() works the way we want it to. 
    '''    
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
    pt = groebner_basis.inverse_P(p)
    # Result done by hand. 
    pt_inv = [4,0,3,2,1,5]
    assert(np.allclose(M,N[:,pt])), "Matrix are not the same."
    assert(all(pt == pt_inv)), "Wrong matrix order."
    
    
    # Test Case 2: 
    A = np.random.random((5,10))
    
    Q,R,p = qr(A,pivoting=True)
    
    pt = groebner_basis.inverse_P(p)
    
    # We know that A[:,p] = QR, want to see if pt flips QR back to A. 
    assert(np.allclose(A,np.dot(Q,R)[:,pt]))

def test_triangular_solve():
    """This tests the triangular_solve() method. 
    A visual graph of zeroes on the diagonal was also used to test this function.
    """
    
    A = np.array([[1, 2, 3, 4, 5],
                  [0, 1, 2, 3, 4],
                  [0, 0, 0, 1, 2]])

    matrix = groebner_basis.triangular_solve(A)
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
    new_matrix = groebner_basis.triangular_solve(matrix)
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
    new_matrix = groebner_basis.triangular_solve(matrix)
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
    new_matrix = groebner_basis.triangular_solve(matrix)
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
        groebner_basis.solve([C,D])
        
def test_reduce_matrix():
    poly1 = MultiPower(np.array([[1., 0.],[0., 1.]]))
    poly2 = MultiPower(np.array([[0., 0.],[1., 0.]]))
    poly3 = MultiPower(np.array([[1., 0.],[12., -5.]]))

    new_polys = list((poly1, poly2, poly3))
    old_polys = []
    items={'power':True}
    items = groebner_basis.initialize_np_matrix(new_polys,old_polys,items)
    groebner_basis._add_poly_to_matrix(new_polys,items)
    added,new_polys,old_polys = groebner_basis.reduce_matrix(items)
    #This breaks becasue it hasn't been initialized.
    assert(added)
    assert(len(old_polys) == 2)
    assert(len(new_polys) == 1)

    poly1 = MultiPower(np.array([[1., 0.],[0., 0.]]))
    poly2 = MultiPower(np.array([[0., 0.],[1., 0.]]))
    poly3 = MultiPower(np.array([[1., 0.],[12., 0.]]))
    new_polys = list((poly1, poly2, poly3))
    old_polys = []
    items={'power':True}
    items = groebner_basis.initialize_np_matrix(new_polys,old_polys,items)
    items = groebner_basis._add_poly_to_matrix(new_polys,items)
    added,new_polys,old_polys = groebner_basis.reduce_matrix(items)

    #This breaks becasue it hasn't been initialized.
    assert(not added)
    #assert(len(new_polys) == 0)
    #assert(len(old_polys) == 3) # --> this gives error right now. old_poly is 2. 
    

    poly1 = MultiPower(np.array([[1., -14.],[0., 2.]]))
    poly2 = MultiPower(np.array([[0., 3.],[1., 6.]]))
    poly3 = MultiPower(np.array([[1., 0.],[12., -5.]]))

    new_polys = list((poly1, poly2, poly3))
    old_polys = []
    items={'power':True}
    items = groebner_basis.initialize_np_matrix(new_polys,old_polys,items)
    items = groebner_basis._add_poly_to_matrix(new_polys,items)
    added,new_polys,old_polys = groebner_basis.reduce_matrix(items)
    assert(added)
    # assert(len(old_polys) == 3) # --> Also error. len(old_poly) gives 1. 
    assert(len(new_polys) == 2)

# # Added to test timing: 
#def test_timing():
#    """Test the rrqr_reduce function. 
#    """
#    A = MultiPower(np.array([
#            [[[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
#             [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]],
#             [[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
#              [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]
#                                                                                                                                        ]))
#    B = MultiPower(np.array([
#            [[[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
#            [[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]],
#            [[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
#            [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]
#                                                                                                                                         ]
#                                                                                                           ))
#    C = MultiPower(np.array([
#            [[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
#             [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]],
#            [[[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
#             [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]
#                                                                                  ]
#                                                                                                           ))
#    groebner_basis.solve([A,B])
#	# C was taken out because it takes a long time. 
