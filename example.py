# from the groebner library
from groebner.multi_cheb import MultiCheb
from groebner.multi_power import MultiPower
from groebner import maxheap
from groebner.groebner_class import Groebner

# other libraries
import numpy as np
import pandas as pd
import scipy.linalg as la


# Example 1: 3-dimensional system

A = MultiPower(np.array([
    [[1,1,3],[0,0,2],[6,4,3]],
    [[1,2,3],[1,3,2],[5,1,4]],
    [[2,4,3],[4,1,2],[4,2,3]]
    ]))

B = MultiPower(np.array([
    [[1,3,3],[0,3,2],[6,4,3]],
    [[3,2,3],[1,13,1],[5,4,5]],
    [[2,1,3],[4,1,2],[2,1,2]]
    ]))

C = MultiPower(np.array([
    [[2,3,3],[0,3,-2],[-6,4,3]],
    [[-3,2,3],[1,1,1],[5,4,5]],
    [[2,1,-3],[-4,1,2],[-2,1,2]]
    ]))


grob = Groebner([A,B,C])
grob.solve()


# Example 2: 2-dimensional system

A = MultiPower(np.array([[1,0,-2,1],[2,0,5,1],[1,0,4,1],[2,0,3,1]]))
B = MultiPower(np.array([[1,0,8,7],[1,0,1,2],[0,4,1,2],[0,1,5,4]]))

grob = Groebner([A,B])
grob.solve()


# Example 3: Step-by-step

# Step 1: Define the system.
A = MultiPower(np.array([[1,1],[2,3]]))
B = MultiPower(np.array([[1,1],[3,4]]))
C = MultiPower(np.array([[5,2],[2,4]]))
D = MultiPower(np.array([[1,1,1],[2,2,2],[3,3,3]]))

grob = Groebner([A,B,C,D])
grob.initialize_np_matrix()
input("The system as a matrix:\n" + str(grob.np_matrix))

# Step 2: Add phis.
grob.add_phi_to_matrix()
input("The matrix with Phis (?) appended:\n" + str(grob.np_matrix))
