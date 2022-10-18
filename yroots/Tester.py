#imports
import time
import numpy as np
import yroots as yr
from ChebyshevSubdivisionSolver import solveChebyshevSubdivision

np.random.seed(17)
def getMs(dim=2, deg=5, divisor = 3):
    #This returns functions that look hopefully kind of lke Chebyshev approximations
    Ms = []
    for i in range(dim):
        M = np.random.rand(*[deg]*dim)*2-1
        M[tuple([0]*dim)] = (np.random.rand()*2-1)/10
        for spot, num in np.ndenumerate(M):
            scaler = divisor**np.sum(spot)
            M[spot] /= scaler
        Ms.append(M)
    return Ms


#Test one degree and dimension at a time
dim = 2
deg = 5
print("dim: ", end = '')
print(dim,end='')
print(" deg: ", end = '')
print(deg)

intervals = np.vstack([-np.ones(dim),np.ones(dim)]).T
errs = np.zeros(dim)

ms = getMs(dim, deg)
# ms = [x*1000 for x in ms]

start = time.time()
roots = solveChebyshevSubdivision(ms,errs)
end = time.time() 
print(roots)
print(end - start)


# # Test many different degrees at a time in any dimension
# dim = 3

# for j in range(3,30):
#     deg = j
#     print("dim: ", end = '')
#     print(dim,end='')
#     print(" deg: ", end = '')
#     print(deg)

#     intervals = np.vstack([-np.ones(dim),np.ones(dim)]).T
#     errs = np.zeros(dim)

#     ms = getMs(dim, deg)
#     # ms = [x*1000 for x in ms]

#     start = time.time()
#     roots = solveChebyshevSubdivision(ms, errs, exact = False)
#     end = time.time()


#     print(roots)
#     print(end - start)