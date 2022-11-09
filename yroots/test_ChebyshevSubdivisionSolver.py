import pytest
import ChebyshevSubdivisionSolver as chebsolver
import numpy as np
from time import time
import M_maker

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
    start = time()
    f_approx = M_maker(f,a,b,f_deg) 
    g_approx = M_maker(g,a,b,g_deg)
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
    return True #not forever

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
    """
    Tests...
    """
    Mf, Mg, err_f, err_g = set_up_Ms_errs
    

    
