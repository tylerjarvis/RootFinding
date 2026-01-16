import numpy as np
import yroots as yr
import yroots.ChebyshevSubdivisionSolver as ChebyshevSubdivisionSolver
import pytest

# These are tests from Combined

tol1 = 1e-12
tol2 = 1e-14

def assert_same_points(A, B, tol=1e-10):
    """
    Use to verify roots in different order are the same
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    assert A.shape == B.shape

    A = A.copy()
    B = B.copy()

    used = np.zeros(len(B), dtype=bool)
    for i in range(len(A)):
        d = np.linalg.norm(B - A[i], axis=1)
        j = np.argmin(d)
        assert d[j] < tol, f"Point {A[i]} not found within tol; best dist={d[j]}"
        assert not used[j], "Matched the same point twice (duplicate roots?)"
        used[j] = True

# Test Univariate Examples
def test_univariate():
    f = lambda x : np.sin(np.exp(3*x))
    roots = yr.solve(f, -1, 2)
    assert len(roots) == 128
    assert np.max(np.abs(f(roots))) < tol1

def test_univariate_power():
    coeff = np.zeros(5)
    coeff[0], coeff[1], coeff[2], coeff[3], coeff[4] = -2, 2, 3, 4, 5
    f = yr.MultiPower(coeff)

    roots = yr.solve(f, -2, 1)
    assert len(roots) == 2
    assert np.max(np.abs(f(roots))) < tol1

def test_univariate_cheb():
    coeff = np.zeros(4)
    coeff[0], coeff[1], coeff[2], coeff[3] = 0, 1, 2, 3
    f = yr.MultiCheb(coeff)

    roots = yr.solve(f, -0.5, 1)
    assert len(roots) == 2
    assert np.max(np.abs(f(roots))) < tol1

# Test Multidimensional Examples
def test_bivariate():
    f = lambda x,y : np.sin(x*y) + x*np.log(y+3) - x**2 + 1/(y-4)
    g = lambda x,y : np.cos(3*x*y) + np.exp(3*y/(x-2)) - x - 6

    a = [-1,-2] #lower bounds on x and y
    b = [0,1] #upper bounds on x and y

    roots = yr.solve([f,g], a, b)

    assert len(roots) == 2
    assert np.max(np.abs(f(roots[:,0],roots[:,1]))) < tol2
    assert np.max(np.abs(g(roots[:,0],roots[:,1]))) < tol2

def test_high_dim():
    f1 = lambda x1, x2, x3, x4, x5: np.cos(x1) + x5 - 1
    f2 = lambda x1, x2, x3, x4, x5: np.cos(x2) + x4 - 2
    f3 = lambda x1, x2, x3, x4, x5: np.cos(x3) + x3 - 3
    f4 = lambda x1, x2, x3, x4, x5: np.cos(x4) + x2 - 4
    f5 = lambda x1, x2, x3, x4, x5: np.cos(x5) + x1 - 5

    a = [0]*5
    b = [2*np.pi]*5

    roots = yr.solve([f1,f2,f3,f4,f5],a,b,verbose=True)

    assert len(roots) == 1
    assert np.max([np.abs(f(*[roots[:,i] for i in range(5)])) for f in [f1,f2,f3,f4,f5]]) < tol2

# Test MultiCheb and MultiPower
def test_multiCheb_multiPower():
    """
    f(x,y) = 5x^3 + 4 xy^2 + 3x^2 + 2y^2 + 1
    g(x,y) = 5 T_2(x) + 3T_1(x)T_2(y) + 2

    """

    coeff = np.zeros((4,4))
    coeff[3,0], coeff[1,2], coeff[2,0], coeff[0,2], coeff[0,0] = 5, 4, 3, 2, 1
    f = yr.MultiPower(coeff)

    coeff = np.zeros((3,3))
    coeff[2,0], coeff[1,2], coeff[0,0] = 5, 3, 2
    g = yr.MultiCheb(coeff)

    roots = yr.solve([f,g],[-1,-1],[1,1])

    assert len(roots) == 2
    assert np.max(np.abs(f(roots))) < tol2
    assert np.max(np.abs(g(roots))) < tol2

def test_multiPower():
    """
    f(x,y) = 5x^3 + 4 xy^2 + 3x^2 + 2y^2 - 5
    g(x,y) = 3x^2 y - 4x y^2 + 3x^2 + 2y^2 -1

    """
    coeff = np.zeros((4,4))
    coeff[3,0], coeff[1,2], coeff[2,0], coeff[0,2], coeff[0,0] = 5, 4, 3, 2, -5
    f = yr.MultiPower(coeff)

    coeff = np.zeros((4,4))
    coeff[2,1], coeff[1,2], coeff[2,0], coeff[0,2], coeff[0,0] = 3, -4, 3, 2, -1
    g = yr.MultiPower(coeff)

    roots = yr.solve([f,g],[-2,-2],[2,2])

    assert len(roots) == 2
    assert np.max(np.abs(f(roots))) < tol2
    assert np.max(np.abs(g(roots))) < tol2

def test_multiCheb():
    """
    f(x,y) = 5x^3 + 4 xy^2 + 3x^2 + 2y^2 - 5
    g(x,y) = 5 T_2(x) + 3T_1(x)T_2(y) + 2

    """
    coeff = np.zeros((3, 3, 3))
    coeff[1, 0, 0], coeff[0, 1, 2], coeff[2, 1] = -1, 2, 4
    f = yr.MultiCheb(coeff)

    coeff = np.zeros((3, 3, 3))
    coeff[0, 2,0], coeff[1,2, 0], coeff[1, 1, 1] = 5, 3, 2
    g = yr.MultiCheb(coeff)

    coeff = np.zeros((3, 3, 3))
    coeff[0, 0, 1], coeff[1,0, 0], coeff[2, 1, 0] = 2, -1, 3
    h = yr.MultiCheb(coeff)

    roots = yr.solve([f, g, h],[-1, -1, -1],[1, 1, 1])

    assert len(roots) > 0
    assert np.max(np.abs(f(roots))) < tol2
    assert np.max(np.abs(g(roots))) < tol2
    assert np.max(np.abs(h(roots))) < tol2

def test_no_roots():
    # Test 1
    coeff = np.zeros((4,4))
    coeff[3,0], coeff[1,2], coeff[2,0], coeff[0,2], coeff[0,0] = 5, 4, 3, 2, -5
    f = yr.MultiPower(coeff)

    coeff = np.zeros((4,4))
    coeff[2,1], coeff[1,2], coeff[2,0], coeff[0,2], coeff[0,0] = 3, -4, 3, 2, 1
    g = yr.MultiPower(coeff)

    roots = yr.solve([f,g],[-2,-2],[2,2])
    assert len(roots) == 0

    # Test 2
    coeff = np.zeros((4,4))
    coeff[3,0], coeff[1,2], coeff[2,0], coeff[0,2], coeff[0,0] = 5, 4, 3, 2, -5
    f = yr.MultiPower(coeff)

    coeff = np.zeros((4,4))
    coeff[2,1], coeff[1,2], coeff[2,0], coeff[0,2], coeff[0,0] = 3, -4, 3, 2, 0.062
    g = yr.MultiPower(coeff)

    roots = yr.solve([f,g],[-1,-1],[1,1])

def test_bad_intervals():
    """
    tests to make sure bad intervals get rejected by solve:
    (a) a lower bound is greater than an upper bound
    (b) the bounding arrays are unequal in length
    """

    f = lambda x,y: (x-1)*(np.cos(x*y**2)+2)
    g = lambda x,y: np.sin(8*np.pi*y)*(np.cos(x*y)+2)

    a,b = np.array([1.2, -1]), np.array([1, 1])

    with pytest.raises(ValueError) as excinfo:
        yr.solve([f,g], a, b)
    assert excinfo.value.args[0] == "Invalid input: at least one lower bound is greater than the corresponding upper bound."

    a = [a[0]]
    with pytest.raises(ValueError) as excinfo:
        yr.solve([f,g], a, b)
    assert excinfo.value.args[0] == f"Invalid input: {len(a)} lower bounds were given but {len(b)} upper bounds were given"

def test_exact_option():
    """
    Solve has an "exact" option. 
    This tests that option on test case 2.3 from chebfun2_suite.
    We find the roots using the exact method and non-exact method.
    Then we make sure we got the same roots between the two, and that those roots are correct.
    """
    f = lambda x,y: np.sin(4*(x + y/10 + np.pi/10))
    g = lambda x,y: np.cos(2*(x-2*y+ np.pi/7))
    a,b = np.array([-1,-1]),np.array([1,1])

    funcs = [f,g]
    yroots_non_exact = yr.solve(funcs,a,b,exact=False)
    yroots_exact = yr.solve(funcs,a,b,exact=True)

    actual_roots = np.load('../Polished_results/polished_2.3.npy')
    chebfun_roots = np.loadtxt('../Chebfun_results/test_roots_2.3.csv', delimiter=',')

    assert len(yroots_non_exact) == len(actual_roots)
    assert len(yroots_exact) == len(actual_roots)
    assert len(yroots_exact) == len(chebfun_roots)

    actual_roots = np.sort(actual_roots)
    yroots_non_exact = np.sort(yroots_non_exact)
    yroots_exact = np.sort(yroots_exact) 
    chebfun_roots = np.sort(chebfun_roots) #sort the Roots

    assert_same_points(yroots_exact, actual_roots)
    assert_same_points(yroots_exact, chebfun_roots)
    assert_same_points(yroots_non_exact, actual_roots)
    assert_same_points(yroots_non_exact, chebfun_roots)

def testreturnBoundingBoxes():
    """
    Solve has an option to return the bounding boxes on the roots. 
    This test makes sure each root lies within their respective box.
    """
    f = lambda x,y: np.sin(4*(x + y/10 + np.pi/10))
    g = lambda x,y: np.cos(2*(x-2*y+ np.pi/7))
    a,b = np.array([-1,-1]),np.array([1,1])

    yroots, boxes = yr.solve([f, g], a, b, returnBoundingBoxes=True)

    for root, box in zip(yroots,boxes):
        box = ChebyshevSubdivisionSolver.TrackedInterval(box)
        assert box.__contains__(root) == True

def testoutside_neg1_pos1():
    """
    Let the search interval be larger than [-1,1]^n.
    Assert that each root is in its respective box.
    """
    f = lambda x,y: 2*x*y*np.cos(y**2)*np.cos(2*x)-np.cos(x*y)
    g = lambda x,y: 2*np.sin(x*y**2)*np.sin(3*x*y)-np.sin(x*y)
    a,b = np.array([-2,-2]), np.array([2,2])
    funcs = [f,g]
    
    yroots, boxes = yr.solve(funcs, a, b, returnBoundingBoxes=True)
    for root, box in zip(yroots,boxes):
        box = ChebyshevSubdivisionSolver.TrackedInterval(box)
        assert box.__contains__(root) == True