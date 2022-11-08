#imports
import time
import numpy as np
#import random
#import yroots as yr
#import ast
#from ChebyshevSubdivisionSolver import solveChebyshevSubdivision
from ChebyshevSubdivisionSolverWCTC import solveChebyshevSubdivision #as solveChebyshevSubdivisionWCTC
np.random.seed(17)
def getMs(dim=2, deg=5, divisor = 3, rig_const_term=True):
    #This returns functions that look hopefully kind of lke Chebyshev approximations
    Ms = []
    for i in range(dim):
        M = np.random.rand(*[deg]*dim)*2-1
        M[tuple([0]*dim)] = (np.random.rand()*2-1)/10
        for spot, num in np.ndenumerate(M):
            scaler = divisor**np.sum(spot)
            M[spot] /= scaler
        m = np.max(M)
        M = M / m
        if rig_const_term:
            non_const_sum = np.sum(np.abs(M)) - np.abs(M[tuple([0]*dim)])
            M[tuple([0]*dim)] = np.random.normal(non_const_sum/2, non_const_sum/4)
        Ms.append(M)
    return Ms
##Test one degree and dimension at a time
#dim = 3
#deg = 4
#print("dim: ", end = '')
#print(dim,end='')
#print(" deg: ", end = '')
#print(deg)
#intervals = np.vstack([-np.ones(dim),np.ones(dim)]).T
#errs = np.zeros(dim)
#ms = getMs(dim, deg, rig_const_term=True)
## ms = [x*1000 for x in ms]
#start = time.time()
#roots = solveChebyshevSubdivision(ms,intervals,errs, constant_term_check="inside")
#end = time.time()
#print(ms)
#print(roots)
#print(end - start)
 # Test many different degrees at a time in any dimension
dim = 5
deg_range = [3, 12]
num_iters = 500
time_array = np.array([0.0, 0.0, 0.0])
try:
    for j in range(num_iters):
        if (j != 0) and (j % 10 == 0):
            print("{} iterations done".format(j))
        deg = j % (deg_range[1] - deg_range[0] + 1) + deg_range[0]
        print("dim: ", end = '')
        print(dim,end='')
        print(" deg: ", end = '')
        print(deg)
        intervals = np.vstack([-np.ones(dim),np.ones(dim)]).T
        errs = np.zeros(dim)
        ms = getMs(dim, deg)
        # ms = [x*1000 for x in ms]
        test_roots = solveChebyshevSubdivision(ms, intervals, errs)
        temp_list = []
        for i in range(3):
            start = time.time()
            roots = solveChebyshevSubdivision(ms,intervals,errs,constant_term_check=i+1)
            end = time.time()
            if len(test_roots) != len(roots):
                raise Exception("Roots from method {0} do not match! Method {0} got {1} while the base case got {2}".format(i + 1, roots, test_roots))
            temp_list.append(end - start)
        time_array += np.array(temp_list)
except KeyboardInterrupt:
    print("Terminating early.")
finally:
    for i in range(3):
        print("Method {} took {} seconds".format(i+1, time_array[i]))