import pytest
import yroots.ChebyshevSubdivisionSolver as chebsolver
import numpy as np
from mpmath import mp
from numba import njit, float64
from numba.types import UniTuple
from scipy.spatial import HalfspaceIntersection
from scipy.optimize import linprog
from itertools import permutations, product
from time import time
import yroots.M_maker as M_maker

n = 5
interval = np.array([np.random.random(n)*-1,np.random.random(n)]).T
tracked = chebsolver.TrackedInterval(interval)

@pytest.fixture
def set_up_Ms_errs():
    """
    Makes pytest fixtures of the test functions so we have linespace
    """
    f = lambda x,y: 144*(x**4+y**4)-225*(x**2+y**2) + 350*x**2*y**2+81     #TODO: make f and g pytest fixtures
    g = lambda x,y: y-x**6
    f_deg,g_deg = 4,6
    a,b = np.array([-1.,-1.]),np.array([1.,1.])
    f_approx = M_maker.M_maker(f,a,b,f_deg) 
    g_approx = M_maker.M_maker(g,a,b,g_deg)
    return f_approx.M,g_approx.M,f_approx.err,g_approx.err

def test_size_tracked():
    assert tracked.size() == np.product(interval[:,1] - interval[:,0])

def test_copy():
    tracked_copy = tracked.copy()
    assert np.allclose(tracked.interval,tracked_copy.interval) == True
    for i in range(len(tracked.transforms)):
        assert np.allclose(tracked.transforms[i],tracked_copy.transforms[i])
    assert np.allclose(tracked.topInterval, tracked_copy.topInterval) == True
    assert tracked.empty == tracked_copy.empty
    assert tracked.ndim == tracked_copy.ndim

def test_contains():
    point_bad = 5*np.random.random(n)
    in_bool_bad = np.all(point_bad >= tracked.interval[:,0]) and np.all(point_bad <= tracked.interval[:,1])
    assert in_bool_bad == tracked.__contains__(point_bad)
    point_good = np.zeros(n)
    in_bool_good = np.all(point_good >= tracked.interval[:,0]) and np.all(point_good <= tracked.interval[:,1])
    assert in_bool_good == tracked.__contains__(point_good)

def test_overlaps():
    test_interval = np.array([np.array([-1]*n),np.array([1]*n)])
    test_object = chebsolver.TrackedInterval(test_interval)
    bad_interval = np.array([np.array([2]*n),np.array([3]*n)]).T
    good_interval = np.array([np.array([0.2]*n),np.array([0.3]*n)]).T
    assert test_object.overlapsWith(bad_interval) == False
    assert test_object.overlapsWith(good_interval) == True

def test_get_Linear_Terms():
    """
    Test1: test that we calculated the terms correctly
    Test2: needed or covered by test_linearCheck1()? 
    think what would happen if we just did a linear check on just quad terms...cubic terms...
    """
    Mf= set_up_Ms_errs[0]
    assert np.allclose(chebsolver.getLinearTerms(Mf),np.array([Mf[1,0],Mf[0,1]]))

def test_find_vertices():
    #get shape of A_ub
    #vertically stack positive, negative identities
    #slap a negative ones vector of double column length on bottom
    #get number of rows of this stacked thing
    #make halfspaces, a matrix of space m+o by n+1
    #assign input A_ub for the first m by n entries of halfspaces
    #assign next row to be negative -b_ub
    #assign remaining rows to be the vertical stacking we made

    #take some norm
    #stack norm_vector onto A, subtract columns from before away from b ??
    #c vector is a vector of zeros with a negative one at the end
    #run linprog on c A and b over (-1,1)

    #gets a feasible point or fails to do so, returning the whole interval or a set of inetersections
    pass #not forever

def test_linearCheck1():
    """
    Test 1: uses randomly generated A,consts,totalErrs and validates correctness of calculations
    Test 2: uses a test case from chebfun2_suite and validates the location of roots in tighter interval
    """
    A = np.array([[0.12122536, 0.58538202, 0.28835862, 0.90334211, 0.17009259],
       [0.8381725 , 0.49698512, 0.1761786 , 0.40609808, 0.30879373],
       [0.73555404, 0.27864861, 0.54397928, 0.90567404, 0.49692915],
       [0.80468046, 0.41315412, 0.99003273, 0.98359542, 0.25886889],
       [0.43675962, 0.13999244, 0.9270024 , 0.17952587, 0.79644536]])
    totalErrs = np.array([0.674356  , 0.36258152, 0.47688965, 0.58847478, 0.37490811])
    consts = np.array([0.8383123 , 0.44548865, 0.55498803, 0.91740163, 0.0369741 ])
    result = chebsolver.linearCheck1(totalErrs,A,consts)
    assert np.allclose(result[0],np.array([0.5674142 , 0.27043788, 0.59556943, 0.47344229, 0.52927329])) == True
    assert np.allclose(result[1],np.array([-9.26781277, -4.01661874, -4.47577129, -2.30115309, -6.89248849])) == True

    #get your approximations with errors
    Mf, Mg, err_f, err_g = set_up_Ms_errs
    Ms,errs = [Mf,Mg],[err_f,err_g]

    #get their linear terms,consts,totalErrs
    A = np.array([chebsolver.getLinearTerms(M) for M in Ms]) 
    consts = np.array([M.ravel()[0] for M in Ms]) 
    totalErrs = np.array([np.sum(np.abs(M)) + e for M,e in zip(Ms, errs)])

    #built the interval using TrackedInterval class
    a,b = chebsolver.linearCheck1(totalErrs, A, consts)
    interval_fg = np.array([a,b]).T 
    tracked_fg = chebsolver.TrackedInterval(interval_fg) 
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_1.1.csv', delimiter=',') #load the roots in

    #assert that each root still belongs in this tighter interval
    for root in chebfun_roots:
        assert tracked_fg.__contains__(root) 

def test_BoundIntervalLinearSystem():
    #Mf, Mg, err_f, err_g = set_up_Ms_errs
    pass #not forever   

def sortRoots(roots, seed = 12398):
    if len(roots) == 0:
        return roots
    np.random.seed(seed)
    dim = roots.shape[1]
    r = np.array(np.random.rand(dim))
    order = np.argsort(roots@r)
    return roots[order]

def runSystem(degList):
    #Each row of degList is the degrees of 1 polynomial
    degList = np.array(degList)
    dim = len(degList)
    #Get the actual roots
    mp.dps = 50
    actualRoots = []
    for i in permutations(range(dim)):
        currDegs = np.array([degList[i[j],j] for j in range(dim)])
        currRootList = []
        for deg in currDegs:
            d = int(deg)
            theseRoots = [float(mp.cos(mp.pi*(mp.mpf(num)+0.5)/mp.mpf(d))) for num in mp.arange(d)]
            currRootList.append(np.array(theseRoots))
        for root in product(*currRootList):
            actualRoots.append(np.array(root))
    actualRoots = sortRoots(np.array(actualRoots))
    #Construct the problem
    Ms = []
    for degs in degList:
        M = np.zeros(degs+1)
        M[tuple(degs)] = 1.0
        Ms.append(M)
    errors = np.zeros(dim)
    #Solve
    foundRoots = chebsolver.solveChebyshevSubdivision(Ms, errors)
    return sortRoots(np.array(foundRoots)), actualRoots

def isGoodSystem(degList):
    zeros = [[float(mp.cos(mp.pi*(num+0.5)/d)) for num in mp.arange(d)] for d in degList]
    zeros = np.sort(np.hstack(zeros).ravel())
    return len(zeros) <= 1 or np.min(np.diff(zeros)) > 1e-12

def getTestSystems(dim, maxDeg):
    systems = []
    for degrees in product(range(1, maxDeg+1), repeat=dim):
        if isGoodSystem(degrees):
            systems.append(degrees)
    return systems

def test_runChebMonomialsTests(dims, maxDegs, verboseLevel = 0, returnErrors = False):
    allErrors = []
    for dim, maxDeg in zip(dims, maxDegs):
        currErrors = []
        if verboseLevel > 0:
            print(f"Running Cheb Monomial Test Dimension: {dim}, Max Degree: {maxDeg}")
        testSytems = getTestSystems(dim, maxDeg)
        numTests = len(testSytems)**dim
        count = 0
        for degrees in product(testSytems, repeat = dim):
            count += 1
            polyDegs = np.array(degrees).T
            if verboseLevel > 1:
                print(f"Running Cheb Monomial Test {count}/{numTests} Degrees: {polyDegs}")
            errorString = "Test on degrees: " + str(polyDegs)
            foundRoots, actualRoots = runSystem(polyDegs)
            assert(len(foundRoots) == len(actualRoots)), "Wrong Number of Roots! " + errorString
            maxError = np.max(np.abs(foundRoots - actualRoots))
            if returnErrors:
                currErrors.append(np.linalg.norm(foundRoots - actualRoots, axis=1))
            assert(maxError < 1e-15), "Error Too Large! " + errorString
        if returnErrors:
            allErrors.append(np.hstack(currErrors))
    if returnErrors:
        return allErrors    

def test_makeMatrix():
    n = 2
    a = -0.25
    b = 0.125
    alpha,beta = 0.5*(b-a),0.5*(b+a)
    C = np.array([[1,beta],[0,alpha]])
    madematrix = chebsolver.makeMatrix(n,a,b)
    assert np.allclose(C,madematrix)

def test_getTransformPoints():
    n = np.random.randint(1,11)
    interval = np.array([-1*np.random.random(n), np.random.random(n)]).T
    interval = chebsolver.TrackedInterval(interval)
    a,b = interval
    alpha_hat,beta_hat = 0.5*(b-a), 0.5*(b+a)
    alpha,beta = chebsolver.getTransformPoints(interval)
    xhat,x = np.array([alpha_hat,beta_hat]),np.array([alpha,beta])
    assert np.allclose(x,xhat)

def test_isValidSpot():
    #functionality testing
    assert chebsolver.isValidSpot(4,4)
    assert chebsolver.isValidSpot(3,4)
    assert chebsolver.isValidSpot(4,3) == False
    #intent testing

def test_makeMatrix():
    mat = chebsolver.makeMatrix(5,43,1)
    assert mat.shape == (5,5)
    assert mat[0,1] == 1
    assert mat[1,1] == 43

def test_BoundingIntervalLinearSystem():
    """
    Makes sure that the roots the solver finds are actually contained in the 
    shrinked interval.
    """
    f = lambda x,y: 144*(x**4+y**4)-225*(x**2+y**2) + 350*x**2*y**2+81   
    #TODO: make f and g pytest fixtures
    g = lambda x,y: y-x**6
    f_deg,g_deg = 4,6
    a,b = np.array([-1.,-1.]),np.array([1.,1.])
    f_approx = M_maker.M_maker(f,a,b,f_deg) 
    g_approx = M_maker.M_maker(g,a,b,g_deg)
    Ms = [f_approx.M,g_approx.M]
    errs = [f_approx.err,g_approx.err]
    newInterval = chebsolver.BoundingIntervalLinearSystem(Ms,errs)[0] #don't include the bool return value
    newInterval = chebsolver.TrackedInterval(newInterval)
    for root in chebsolver.solveChebyshevSubdivision(Ms,errs):
        assert newInterval.__contains__(root)

def test_getTransformationError():
    """sets the transformation Error"""
    M = np.ones((2,2))
    dim = 2
    machEps = 2**-52
    assert chebsolver.getTransformationError(M,dim) == (machEps * 4 * 2)
    #this doesn't make sense: M.shape[dim]

def test_transformChebInPlaceND():
    #why would dim ever be zero
    pass

def test_transformCheb():
    pass

def test_transformChebToInterval():
    pass

def test_find_vertices():
    pass

def test_zoomInOnIntervalIter():
    #why is dims=len(Ms) ASK Erik:
    #not sure how we test the np.any(...) part
    pass