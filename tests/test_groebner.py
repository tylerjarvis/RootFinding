import numpy as np
from itertools import permutations
from groebner.polynomial import MultiCheb, MultiPower
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

def test_init_():
    C = MultiPower(np.array([[-1,0,1],[0,0,0]]))
    D = MultiCheb(np.array([[-1,0,0],[0,1,0],[1,0,0]]))
    with pytest.raises(ValueError):
        grob = Groebner([C,D])
