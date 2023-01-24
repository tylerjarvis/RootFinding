#imports
import time
import numpy as np
import yroots as yr
from ChebyshevSubdivisionSolver import solveChebyshevSubdivision
from yroots.polynomial import MultiCheb

#This chunk of the code produces a funciton that looks kind of like a Chebyshev approximation
np.random.seed(15)
def getMs(dim=2, deg=5, divisor = 3):
    Ms = []
    for i in range(dim):
        M = np.random.rand(*[deg]*dim)*2-1
        M[tuple([0]*dim)] = (np.random.rand()*2-1)/10
        for spot, num in np.ndenumerate(M):
            scaler = divisor**np.sum(spot)
            M[spot] /= scaler
        Ms.append(M)
    return Ms





# #THIS BLOCK RUNS ONE DIMENSION AND DEGREE AT A TIME
# dim = 5
# deg = 4
# print("dim: ", end = '')
# print(dim,end='')
# print(" deg: ", end = '')
# print(deg)

# intervals = np.vstack([-np.ones(dim),np.ones(dim)]).T
# errs = np.zeros(dim)

# ms = getMs(dim, deg)

# start = time.time()
# roots = solveChebyshevSubdivision(ms,errs, exact = False)
# end = time.time() 
# print(roots)
# print(end - start)







#THIS CODE RUNS A SPECIFIC SEED OF A SPECIFIC DIMENSION AND DEGREE
#If the degree you want to run is 8, then set the if function to test for one less than that, i.e.: if i == 7 (rather than if i == 8)
# dim = 5
# for i in range(2,8):
#     getMs(dim,i)

#     if i == 7:
#         this_seed = np.random.get_state()
# np.random.set_state(this_seed)

# deg = 8
# print("dim: ", end = '')
# print(dim,end='')
# print(" deg: ", end = '')
# print(deg)

# intervals = np.vstack([-np.ones(dim),np.ones(dim)]).T
# errs = np.zeros(dim)

# ms = getMs(dim, deg)
# start2 = time.time()
# roots = solveChebyshevSubdivision(ms, errs, exact = False)
# end2 = time.time()

# print(f" Roots = {roots}")
# # print(f" residuals = {residuals}")
# print(" Time =", end2 - start2)
# print()







#THIS CODE PRINTS THE RESIDUALS OF ANY ROOTS THAT WERE FOUND
# residuals = np.abs(np.hstack([[MultiCheb(M)(point) for M in ms] for point in roots]))
# print(residuals)





#THIS CODE RUNS A RANGE OF DEGREES FOR A GIVEN DIMENSION
dim = 9
start = time.time()
for j in range(2,30):
       
    deg = j
    print("dim: ", end = '')
    print(dim,end='')
    print(" deg: ", end = '')
    print(deg)

    intervals = np.vstack([-np.ones(dim),np.ones(dim)]).T
    errs = np.zeros(dim)

    ms = getMs(dim, deg)
    start2 = time.time()
    roots = solveChebyshevSubdivision(ms, errs, exact = False)
    end2 = time.time()

    print(f" Roots = {roots}")
    # print(f" residuals = {residuals}")
    print(" Time =", end2 - start2)
    print()
end = time.time()
print(f"Final Time: {end-start}")