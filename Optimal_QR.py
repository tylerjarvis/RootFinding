import numpy as np
import time
from yroots.subdivision import solve
import pickle


def timeIt(method, funcs, a=np.array([-1,-1]), b=np.array([1,1]), trials=5):
        """ Runs the test multiple times and takes the average of the times.

        Parameters
        ----------
            funcs : lambda functions
                The functions to run the tests on.
            a : np.array
                Lower bounds of the interval.
            b : np.array
                Upper bounds of the interval.
            trials : int
                Number of times to time the solve function.

        Returns
        -------
            time : float
                The average time per trial it took.
            roots : numpy array
                The roots that the solver found.
        """
        time_list = list()
        for i in range(trials):
            start = time.time()
            solve(method, funcs, a, b)
            end = time.time()
            time_list.append(end - start)

        return sum(time_list)/trials

def norm_test(yroots, roots, tol=2.220446049250313e-13):
    """ Determines whether the roots given pass or fail the test according
        to whether or not their norms are within tol of the norms of the
        "actual" roots, which are determined either by previously known
        roots or Marching Squares roots.

    Parameters
    ----------
        yroots : numpy array
            The roots that yroots found.
        roots : numpy array
            "Actual" roots either obtained analytically or through Marching
            Squares.
        tol : float, optional
            Tolerance that determines how close the roots need to be in order
            to be considered close. Defaults to 1000*eps where eps is machine
            epsilon.

    Returns
    -------
         root_diff : float
            The norm of the max difference between the roots (either in x or y)

    """
    roots_sorted = np.sort(roots,axis=0)
    yroots_sorted = np.sort(yroots,axis=0)
    root_diff = roots_sorted - yroots_sorted
    return max(np.linalg.norm(root_diff[:,0]), np.linalg.norm(root_diff[:,1]))

def max_residuals(funcs, roots):
    """ Finds the residuals of the given function at the roots.

    Paramters
    ---------
        funcs : list of functions
            The functions to find the max residuals of.
        roots : numpy array
            The coordinates of the roots.

    Returns
    -------
        numpy array
            The residuals of the function.

    """
    return np.max([np.abs(func(roots[:,0],roots[:,1])) for func in funcs])

def get_results(method,funcs, a, b, comp_roots, n=-1):
    """ Runs the solver keeping track of the time, the max residuals, the norm
        difference to the actual roots (or Chebfun roots where no actual is
        available), and number of roots.

    Parameters
    ----------
        funcs : list of functions
            The functions to run the solver on.
        a : numpy array
            The lower bounds of the interval to check.
        b : numpy array
            The upper bounds of the interval to check.
        comp_roots : numpy array
            The roots to compare to when performing the norm test.
        n : int
            The number of the test of the Chebfun test suite this is performed
            on.

    Returns
    -------
        max_res: float
            The maximum residual. If no roots were found, returns -1.
        timing : float
            How long the tests took (on average).
        num_roots : int
            The number of roots found.
        norm_diff : float
            The difference between the norms according to the Chebfun norm test.
            If there are a different number of roots found, returns -1.
        n : int
            Which test this was run on or the degree of the polynomial
    """
    # Cast a and b as numpy arrays
    a = np.array(a)
    b = np.array(b)

    # Time, solve for the roots, and compute max resiudals.
    timing = timeIt(method, funcs, a, b)
    roots, cond, backcond, cond_eig, grad = solve(method, funcs, a, b)
    num_roots = len(roots)
    max_res = -1
    if num_roots > 0:
        max_res = max_residuals(funcs, roots)


    # The norm test can only be run if the same number of roots are found.
    # If a different number of roots are found, return -1.
    norm_diff = -1
    if num_roots == len(comp_roots):
        norm_diff = norm_test(yroots=roots, roots=comp_roots)

    return max_res, timing, norm_diff, num_roots, cond, backcond, cond_eig, grad, n

def test_roots_1_1(method):
    # Test 1.1
    f = lambda x,y: 144*(x**4+y**4)-225*(x**2+y**2) + 350*x**2*y**2+81
    g = lambda x,y: y-x**6
    # yroots = timeIt([f,g],[-1,-1],[1,1], plot=False)
    m_sq_roots = np.loadtxt('tests/chebfun_test_output/cftest1_1ms.csv', delimiter=',')
    chebfun_roots = np.loadtxt('tests/chebfun_test_output/cftest1_1.csv', delimiter=',')
    return get_results(method,[f,g], [-1, -1], [1, 1], m_sq_roots, 1)

def test_roots_1_2(method):
    # Test 1.2
    f = lambda x,y: (y**2-x**3)*((y-0.7)**2-(x-0.3)**3)*((y+0.2)**2-(x+0.8)**3)*((y+0.2)**2-(x-0.8)**3)
    g = lambda x,y: ((y+.4)**3-(x-.4)**2)*((y+.3)**3-(x-.3)**2)*((y-.5)**3-(x+.6)**2)*((y+0.3)**3-(2*x-0.8)**3)
    # yroots = timeIt([f,g],[-1,-1],[1,1], plot=False)
    chebfun_roots = np.loadtxt('tests/chebfun_test_output/cftest1_2.csv',delimiter=',')

    return get_results(method,[f,g], [-1, -1], [1, 1], chebfun_roots, 2)

def test_roots_1_3(method):
    # Test 1.3
    f = lambda x,y: y**2-x**3
    g = lambda x,y: (y+.1)**3-(x-.1)**2
    # yroots = timeIt([f,g],[-1,-1],[1,1], plot=False)
    chebfun_roots = np.loadtxt('tests/chebfun_test_output/cftest1_3.csv', delimiter=',')
    MSroots = np.loadtxt('tests/chebfun_test_output/cftest1_3ms.csv', delimiter=',')

    return get_results(method,[f,g], [-1, -1], [1, 1], chebfun_roots, 3)

def test_roots_1_4(method):
    # Test 1.4
    f = lambda x,y: x - y + .5
    g = lambda x,y: x + y
    # Single root has to be in matrix form because yroots
    # returns the roots in matrix form.
    a_roots = np.array([[-.25, .25]])

    return get_results(method,[f,g], [-1, -1], [1, 1], a_roots, 4)

def test_roots_1_5(method):
    # Test 1.5
    f = lambda x,y: y + x/2 + 1/10
    g = lambda x,y: y - 2.1*x + 2
    # Single root has to be in matrix form because yroots
    # returns the roots in matrix form.
    a_roots = np.array([[0.730769230769231, -0.465384615384615]])

    return get_results(method,[f,g], [-1, -1], [1, 1], a_roots, 5)

def test_roots_2_1(method):
    # Test 2.1
    f = lambda x,y: np.cos(10*x*y)
    g = lambda x,y: x + y**2
    m_sq_roots = np.loadtxt('tests/chebfun_test_output/cftest2_1ms.csv', delimiter=',')
    chebfun_roots = np.loadtxt('tests/chebfun_test_output/cftest2_1.csv', delimiter=',')

    return get_results(method,[f,g], [-1, -1], [1, 1], m_sq_roots, 6)

def test_roots_2_2(method):
    # Test 2.2
    f = lambda x,y: x
    g = lambda x,y: (x-.9999)**2 + y**2-1
    m_sq_roots = np.loadtxt('tests/chebfun_test_output/cftest2_2ms.csv', delimiter=',')
    chebfun_roots = np.loadtxt('tests/chebfun_test_output/cftest2_2.csv', delimiter=',')
    return get_results(method,[f,g], [-1, -1], [1, 1], m_sq_roots, 7)

def test_roots_2_3(method):
    # Test 2.3
    f = lambda x,y: np.sin(4*(x + y/10 + np.pi/10))
    g = lambda x,y: np.cos(2*(x-2*y+ np.pi/7))
    m_sq_roots = np.loadtxt('tests/chebfun_test_output/cftest2_3ms.csv', delimiter=',')
    chebfun_roots = np.loadtxt('tests/chebfun_test_output/cftest2_3.csv', delimiter=',')

    return get_results(method,[f,g], [-1, -1], [1, 1], m_sq_roots, 8)

def test_roots_2_4(method):
    # Test 2.4
    f = lambda x,y: np.exp(x-2*x**2-y**2)*np.sin(10*(x+y+x*y**2))
    g = lambda x,y: np.exp(-x+2*y**2+x*y**2)*np.sin(10*(x-y-2*x*y**2))
    m_sq_roots = np.loadtxt('tests/chebfun_test_output/cftest2_4ms.csv', delimiter=',')
    chebfun_roots = np.loadtxt('tests/chebfun_test_output/cftest2_4.csv', delimiter=',')

    return get_results(method,[f,g], [-1, -1], [1, 1], m_sq_roots, 9)

def test_roots_2_5(method):
    # Test 2.5
    f = lambda x,y: 2*y*np.cos(y**2)*np.cos(2*x)-np.cos(y)
    g = lambda x,y: 2*np.sin(y**2)*np.sin(2*x)-np.sin(x)
    m_sq_roots = np.loadtxt('tests/chebfun_test_output/cftest2_5ms.csv', delimiter=',')
    chebfun_roots = np.loadtxt('tests/chebfun_test_output/cftest2_5.csv', delimiter=',')

    # Notice this interval is different
    return get_results(method,[f,g], [-4,-4], [4,4], m_sq_roots, 10)

def test_roots_3_1(method):
    # Test 3.1
    # No MS roots to compare to, so we compare to Chebfun's roots.
    f = lambda x,y: ((x-.3)**2+2*(y+0.3)**2-1)
    g = lambda x,y: ((x-.49)**2+(y+.5)**2-1)*((x+0.5)**2+(y+0.5)**2-1)*((x-1)**2+(y-0.5)**2-1)
    chebfun_roots = np.loadtxt('tests/chebfun_test_output/cftest3_1.csv', delimiter=',')

    return get_results(method,[f,g], [-1, -1], [1, 1], chebfun_roots, 11)

def test_roots_3_2(method):
    # Test 3.2
    # No MS roots to compare to, so we compare to Chebfun's roots.
    f = lambda x,y: ((x-0.1)**2+2*(y-0.1)**2-1)*((x+0.3)**2+2*(y-0.2)**2-1)*((x-0.3)**2+2*(y+0.15)**2-1)*((x-0.13)**2+2*(y+0.15)**2-1)
    g = lambda x,y: (2*(x+0.1)**2+(y+0.1)**2-1)*(2*(x+0.1)**2+(y-0.1)**2-1)*(2*(x-0.3)**2+(y-0.15)**2-1)*((x-0.21)**2+2*(y-0.15)**2-1)
    chebfun_roots = np.loadtxt('tests/chebfun_test_output/cftest3_2.csv', delimiter=',')

    return get_results(method,[f,g], [-1, -1], [1, 1], chebfun_roots, 12)

def test_roots_4_1(method):
    # Test 4.1
    # This system has 4 true roots, but ms fails (finds 5).
    f = lambda x,y: np.sin(3*(x+y))
    g = lambda x,y: np.sin(3*(x-y))
    m_sq_roots = np.loadtxt('tests/chebfun_test_output/cftest4_1ms.csv',delimiter=',')
    chebfun_roots = np.loadtxt('tests/chebfun_test_output/cftest4_1.csv', delimiter=',')

    return get_results(method,[f,g], [-1, -1], [1, 1], m_sq_roots, 13)

def test_roots_4_2(method):
    # Test 4.2
    f = lambda x,y: ((90000*y**10 + (-1440000)*y**9 + (360000*x**4 + 720000*x**3 + 504400*x**2 + 144400*x + 9971200)*(y**8) +
                ((-4680000)*x**4 + (-9360000)*x**3 + (-6412800)*x**2 + (-1732800)*x + (-39554400))*(y**7) + (540000*x**8 +
                2160000*x**7 + 3817600*x**6 + 3892800*x**5 + 27577600*x**4 + 51187200*x**3 + 34257600*x**2 + 8952800*x + 100084400)*(y**6) +
                ((-5400000)*x**8 + (-21600000)*x**7 + (-37598400)*x**6 + (-37195200)*x**5 + (-95198400)*x**4 +
                (-153604800)*x**3 + (-100484000)*x**2 + (-26280800)*x + (-169378400))*(y**5) + (360000*x**12 + 2160000*x**11 +
                6266400*x**10 + 11532000*x**9 + 34831200*x**8 + 93892800*x**7 + 148644800*x**6 + 141984000*x**5 + 206976800*x**4 +
                275671200*x**3 + 176534800*x**2 + 48374000*x + 194042000)*(y**4) + ((-2520000)*x**12 + (-15120000)*x**11 + (-42998400)*x**10 +
                (-76392000)*x**9 + (-128887200)*x**8 + (-223516800)*x**7 + (-300675200)*x**6 + (-274243200)*x**5 + (-284547200)*x**4 +
                (-303168000)*x**3 + (-190283200)*x**2 + (-57471200)*x + (-147677600))*(y**3) + (90000*x**16 + 720000*x**15 + 3097600*x**14 +
                9083200*x**13 + 23934400*x**12 + 58284800*x**11 + 117148800*x**10 + 182149600*x**9 + 241101600*x**8 + 295968000*x**7 +
                320782400*x**6 + 276224000*x**5 + 236601600*x**4 + 200510400*x**3 + 123359200*x**2 + 43175600*x + 70248800)*(y**2) +
                ((-360000)*x**16 + (-2880000)*x**15 + (-11812800)*x**14 + (-32289600)*x**13 + (-66043200)*x**12 + (-107534400)*x**11 +
                (-148807200)*x**10 + (-184672800)*x**9 + (-205771200)*x**8 + (-196425600)*x**7 + (-166587200)*x**6 + (-135043200)*x**5 +
                (-107568800)*x**4 + (-73394400)*x**3 + (-44061600)*x**2 + (-18772000)*x + (-17896000))*y + (144400*x**18 + 1299600*x**17 +
                5269600*x**16 + 12699200*x**15 + 21632000*x**14 + 32289600*x**13 + 48149600*x**12 + 63997600*x**11 + 67834400*x**10 +
                61884000*x**9 + 55708800*x**8 + 45478400*x**7 + 32775200*x**6 + 26766400*x**5 + 21309200*x**4 + 11185200*x**3 + 6242400*x**2 +
                3465600*x + 1708800)))
    g = lambda x,y: 1e-4*(y**7 + (-3)*y**6 + (2*x**2 + (-1)*x + 2)*y**5 + (x**3 + (-6)*x**2 + x + 2)*y**4 + (x**4 + (-2)*x**3 + 2*x**2 +
                x + (-3))*y**3 + (2*x**5 + (-3)*x**4 + x**3 + 10*x**2 + (-1)*x + 1)*y**2 + ((-1)*x**5 + 3*x**4 + 4*x**3 + (-12)*x**2)*y +
                (x**7 + (-3)*x**5 + (-1)*x**4 + (-4)*x**3 + 4*x**2))
    m_sq_roots = np.loadtxt('tests/chebfun_test_output/cftest4_2ms.csv',delimiter=',')

    return get_results(method,[f,g], [-1, -1], [1, 1], m_sq_roots, 14)

def test_roots_5(method):
    # Test 5.1
    f = lambda x,y: 2*x*y*np.cos(y**2)*np.cos(2*x)-np.cos(x*y)
    g = lambda x,y: 2*np.sin(x*y**2)*np.sin(3*x*y)-np.sin(x*y)
    m_sq_roots = np.loadtxt('tests/chebfun_test_output/cftest5_1ms.csv',delimiter=',')
    chebfun_roots = np.loadtxt('tests/chebfun_test_output/cftest5_1.csv', delimiter=',')

    return get_results(method,[f,g], [-1, -1], [1, 1], m_sq_roots, 15)

def test_roots_6_1(method):
    # Test 6.1
    # No MS roots to compare to.
    f = lambda x,y: (y - 2*x)*(y+0.5*x)
    g = lambda x,y: x*(x**2+y**2-1)
    chebfun_roots = np.loadtxt('tests/chebfun_test_output/cftest6_1.csv', delimiter=',')

    return get_results(method,[f,g], [-1, -1], [1, 1], chebfun_roots, 16)

def test_roots_6_2(method):
    # Test 6.2
    # This one we find more than them, but it's the correct number of roots.
    # No MS roots to compare to.
    f = lambda x,y: (y - 2*x)*(y+.5*x)
    g = lambda x,y: (x-.0001)*(x**2+y**2-1)
    chebfun_roots = np.loadtxt('tests/chebfun_test_output/cftest6_2.csv', delimiter=',')
    actual_roots = np.array([[1/10000,-1/20000],[1/10000, 1/5000],[-2/np.sqrt(5),1/np.sqrt(5)],[-1/np.sqrt(5),-2/np.sqrt(5)],[1/np.sqrt(5),2/np.sqrt(5)],[2/np.sqrt(5),-1/np.sqrt(5)]])

    return get_results(method,[f,g], [-1, -1], [1, 1], actual_roots, 17)

def test_roots_6_3(method):
    # Test 6.3
    f = lambda x,y: 25*x*y - 12
    g = lambda x,y: x**2+y**2-1
    m_sq_roots = np.loadtxt('tests/chebfun_test_output/cftest6_3ms.csv',delimiter=',')
    chebfun_roots = np.loadtxt('tests/chebfun_test_output/cftest6_3.csv', delimiter=',')

    return get_results(method,[f,g], [-1, -1], [1, 1], m_sq_roots, 18)


def test_roots_7_1(method):
    # Test 7.1
    f = lambda x,y: (x**2+y**2-1)*(x-1.1)
    g = lambda x,y: (25*x*y-12)*(x-1.1)
    m_sq_roots = np.loadtxt('tests/chebfun_test_output/cftest7_1ms.csv',delimiter=',')
    chebfun_roots = np.loadtxt('tests/chebfun_test_output/cftest7_1.csv', delimiter=',')

    return get_results(method,[f,g], [-1, -1], [1, 1], m_sq_roots, 19)


def test_roots_7_2(method):
    # Test 7.2
    f = lambda x,y: y**4 + (-1)*y**3 + (2*x**2)*(y**2) + (3*x**2)*y + (x**4)
    h = lambda x,y: y**10-2*(x**8)*(y**2)+4*(x**4)*y-2
    g = lambda x,y: h(2*x,2*(y+.5))
    chebfun_roots = np.loadtxt('tests/chebfun_test_output/cftest7_2.csv',delimiter=',')

    return get_results(method,[f,g], [-1, -1], [1, 1], chebfun_roots, 20)


def test_roots_7_3(method):
    # Test 7.3
    c = 1.e-09
    f = lambda x,y: np.cos(x*y/(c**2))+np.sin(3*x*y/(c**2))
    g = lambda x,y: np.cos(y/c)-np.cos(2*x*y/(c**2))
    chebfun_roots = np.loadtxt('tests/chebfun_test_output/cftest7_3.csv',delimiter=',')

    return get_results(method,[f,g], [-1e-9, -1e-9],[1e-9, 1e-9], chebfun_roots, 21)


def test_roots_7_4(method):
    # Test 7.4
    f = lambda x,y: np.sin(3*np.pi*x)*np.cos(x*y)
    g = lambda x,y: np.sin(3*np.pi*y)*np.cos(np.sin(x*y))
    chebfun_roots = np.loadtxt('tests/chebfun_test_output/cftest7_4.csv',delimiter=',')

    return get_results(method,[f,g], [-1, -1], [1, 1], chebfun_roots, 22)


def test_roots_8_1(method):
    # Test 8.1
    f = lambda x,y: np.sin(10*x-y/10)
    g = lambda x,y: np.cos(3*x*y)
    m_sq_roots = np.loadtxt('tests/chebfun_test_output/cftest8_1ms.csv',delimiter=',')
    chebfun_roots =  np.loadtxt('tests/chebfun_test_output/cftest8_1.csv',delimiter=',')

    return get_results(method,[f,g], [-1, -1], [1, 1], m_sq_roots, 23)


def test_roots_8_2(method):
    # Test 8.2
    f = lambda x,y: np.sin(10*x-y/10) + y
    g = lambda x,y: np.cos(10*y-x/10) - x
    m_sq_roots = np.loadtxt('tests/chebfun_test_output/cftest8_2ms.csv',delimiter=',')
    chebfun_roots =  np.loadtxt('tests/chebfun_test_output/cftest8_2.csv',delimiter=',')

    return get_results(method,[f,g], [-1, -1], [1, 1], m_sq_roots, 24)


def test_roots_9_1(method):
    # Test 9.1
    f = lambda x,y: x**2+y**2-.9**2
    g = lambda x,y: np.sin(x*y)
    m_sq_roots = np.loadtxt('tests/chebfun_test_output/cftest9_1ms.csv',delimiter=',')
    chebfun_roots =  np.loadtxt('tests/chebfun_test_output/cftest9_1.csv',delimiter=',')

    return get_results(method,[f,g], [-1, -1], [1, 1], m_sq_roots, 25)


def test_roots_9_2(method):
    # Test 9.2
    f = lambda x,y: x**2+y**2-.49**2
    g = lambda x,y: (x-.1)*(x*y - .2)
    m_sq_roots = np.loadtxt('tests/chebfun_test_output/cftest9_2ms.csv',delimiter=',')
    chebfun_roots =  np.loadtxt('tests/chebfun_test_output/cftest9_2.csv',delimiter=',')

    return get_results(method,[f,g], [-1, -1], [1, 1], m_sq_roots, 26)


def test_roots_10(method):
    # Test 10.1
    f = lambda x,y: (x-1)*(np.cos(x*y**2)+2)
    g = lambda x,y: np.sin(8*np.pi*y)*(np.cos(x*y)+2)
    # yroots = timeIt([f,g],[-1,-1],[1,1], plot=False)
    chebfun_roots = np.loadtxt('tests/chebfun_test_output/cftest10_1.csv',delimiter=',')
    actual_roots = np.array([[1, -1.0], [1, -0.875], [1, -0.75], [1, -0.625], [1, -0.5], [1, -0.375],
                            [1, -0.25], [1, -0.125], [1, 0.0], [1, 0.125], [1, 0.25], [1, 0.375],
                            [1, 0.5], [1, 0.625], [1, 0.75], [1, 0.875], [1, 1.0]])

    return get_results(method,[f,g], [-1, -1], [1, 1], actual_roots, 27)

if __name__ == "__main__":
    # Put all the tests in a list
    tests = [test_roots_1_1,test_roots_1_2,test_roots_1_3,
            test_roots_1_4,test_roots_1_5,
            test_roots_2_1, test_roots_2_2,test_roots_2_3,
            test_roots_2_4,test_roots_2_5,
            test_roots_3_1,test_roots_3_2,
            test_roots_4_1, test_roots_4_2,
            test_roots_5,
            test_roots_6_1,test_roots_6_2,test_roots_6_3,
            test_roots_7_1,test_roots_7_2,test_roots_7_3, test_roots_7_4,
            test_roots_8_1,test_roots_8_2,
            test_roots_9_1,test_roots_9_2,
            test_roots_10]

    test_nums =[1.1, 1.2, 1.3, 1.4,1.5,
                2.1, 2.2, 2.3, 2.4, 2.5,
                3.1, 3.2,
                4.1, 4.2,
                5,
                6.1, 6.2, 6.3,
                7.1, 7.2, 7.3, 7.4,
                8.1, 8.2,
                9.1, 9.2,
                10]

    num_tests = len(test_nums)

    # Create the dictionary that maps n to the float test number
    test_num_dict = {n+1:test_nums[n] for n in range(num_tests)}

#debug for TVB
    methods = ['qrt','svd','tvb']
    results_dict = {method:dict() for method in methods}
    for method in methods:
        print('Using',method,end='\n\n')
        residual_dict = dict()
        timing_dict = dict()
        norm_dict = dict()
        num_roots_dict = dict()
        cond_dict = dict()
        backcond_dict = dict()
        cond_eig_dict = dict()
        grad_dict = dict()

        for i, test in enumerate(tests):
            print('Running test {}'.format(test_num_dict[i+1]))
            max_res, timing, norm_diff, num_roots, cond, backcond, cond_eig, grad, test_num = test(method)
            residual_dict[test_num] = max_res
            timing_dict[test_num] = timing
            norm_dict[test_num] = norm_diff
            num_roots_dict[test_num] = num_roots
            cond_dict[test_num] = cond
            backcond_dict[test_num] = backcond
            cond_eig_dict[test_num] = cond_eig
            grad_dict[test_num] = grad

        results_dict[method]['residuals'] = residual_dict
        results_dict[method]['timing'] = timing_dict
        results_dict[method]['norms'] = norm_dict
        results_dict[method]['num_roots'] = num_roots_dict
        results_dict[method]['cond'] = cond_dict
        results_dict[method]['backcond'] = backcond_dict
        results_dict[method]['cond_eig'] = cond_eig_dict
        results_dict[method]['gradient_info'] = grad_dict

        with open('tests/chebsuite_tests/chebsuite_result_{}}.pkl'.format(method), 'xb') as f:
            pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)

    with open('tests/chebsuite_tests/chebsuite_result.pkl', 'xb') as f:
        pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)
