import numpy as np
from yroots.Combined_Solver import solve
from time import time
from matplotlib import pyplot as plt
# TODO Description of where these tests come from, links to relevant papers,
# acknowledgements, etc.


def norm_pass_or_fail(yroots, roots, tol=2.220446049250313e-13):
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
         bool
            Whether or not all the roots were close enough.
    """
    roots_sorted = np.sort(roots,axis=0)
    yroots_sorted = np.sort(yroots,axis=0)
    root_diff = roots_sorted - yroots_sorted
    return np.linalg.norm(root_diff[:,0]) < tol and np.linalg.norm(root_diff[:,1]) < tol


def residuals(func, roots):
    """ Finds the residuals of the given function at the roots.
    Paramters
    ---------
        func : function
            The function to find the residuals of.
        roots : numpy array
            The coordinates of the roots.

    Returns
    -------
        numpy array
            The residuals of the function.
    """
    return np.abs(func(roots[:,0],roots[:,1]))


def residuals_pass_or_fail(funcs, roots, tol=2.220446049250313e-13):
    """ Determines whether the roots given pass or fail the test according
        to whether or not the maximal residuals are within a certain tolerance.
    Parameters
    ----------
        funcs : list of functions
            The functions to find the residuals of.
        roots : numpy array
            The roots to plug into the functions to get the residuals.
        tol : float, optional
            How close to 0 the maximal residual must be in order to pass.
            Defaults to 1000* eps where eps is machine epsilon.
    Returns
    -------
        bool
            True if the roots pass the test (are close enough to 0), False
            otherwise.
    """
    for func in funcs:
        if np.max(residuals(func, roots)) > tol:
            return False

    return True


def pass_or_fail(funcs, yroots, roots, test_num, test_type="norm", tol=2.220446049250313e-13):
    """Determines whether a test passes or fails bsed on the given criteria.
    Parameters
    ----------
        funcs : list of functions
            The functions to find the roots of.
        yroots : numpy array
            Roots found by yroots.
        roots : numpy array
            The list of "actual" or Marching Squares roots.
        test_num : float or string
            The number of the test. For example, test 9.2 one could pass in
            "9.2" or 9.2.
        test_type : string, optional
            What type of test to use to determine wheter it passes or fails.
             - "norm" -- runs norm_pass_or_fail, default
             - "residual" -- runs residual_pass_or_fail
        tol : float, optional
            The tolerance with which we want to run our tests. Defualts to
            1000*eps where eps is machine epsilon.
    Raises
    ------
        AssertionError
            If len(yroots) != len(roots) or if it fails the residual
            or norm tests.
        ValueError
            If test_type is not "norm" or "residual"
    """
    if (test_type not in ['norm','residual']):
        raise ValueError("test_type must be 'norm' or 'residual'.")

    if len(yroots) != len(roots):
        if len(yroots) > len(roots):
            raise AssertionError("Test " + str(test_num) +  ": YRoots found"
                                 " too many roots: " + str(len(yroots)) +
                                 " where " + str(len(roots)) + " were expected.")
        else:
            raise AssertionError("Test " + str(test_num) +  ": YRoots didn't"
                                 " find enough roots: " + str(len(yroots)) +
                                 " where " + str(len(roots)) + " were expected.")

    if test_type == 'norm':
        assert norm_pass_or_fail(yroots, roots, tol=tol), "Test " + str(test_num) + " failed."
    else:
        assert residuals_pass_or_fail(funcs, yroots, tol=tol), "Test " + str(test_num) + " failed."


def norm_pass_or_fail(yroots, roots, tol=2.220446049250313e-13):
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
         bool
            Whether or not all the roots were close enough.
    """
    roots_sorted = np.sort(roots,axis=0)
    yroots_sorted = np.sort(yroots,axis=0)
    root_diff = roots_sorted - yroots_sorted
    return np.linalg.norm(root_diff[:,0]) < tol and np.linalg.norm(root_diff[:,1]) < tol, np.linalg.norm(root_diff[:,0]), np.linalg.norm(root_diff[:,1])


def residuals(func, roots):
    """ Finds the residuals of the given function at the roots.
    Paramters
    ---------
        func : function
            The function to find the residuals of.
        roots : numpy array
            The coordinates of the roots.

    Returns
    -------
        numpy array
            The residuals of the function.
    """
    return np.abs(func(roots[:,0],roots[:,1]))


def residuals_pass_or_fail(funcs, roots, tol=2.220446049250313e-13):
    """ Determines whether the roots given pass or fail the test according
        to whether or not the maximal residuals are within a certain tolerance.
    Parameters
    ----------
        funcs : list of functions
            The functions to find the residuals of.
        roots : numpy array
            The roots to plug into the functions to get the residuals.
        tol : float, optional
            How close to 0 the maximal residual must be in order to pass.
            Defaults to 1000* eps where eps is machine epsilon.
    Returns
    -------
        bool
            True if the roots pass the test (are close enough to 0), False
            otherwise.
    """
    for func in funcs:
        if np.max(residuals(func, roots)) > tol:
            return False

    return True

def verbose_pass_or_fail(funcs, yroots, polished_roots, test_num, cheb_roots=None, tol=2.220446049250313e-13):
    """ Determines which tests pass and which fail.
    Parameters
    ----------
        funcs : list of functions
            The functions to find the roots of.
        yroots : numpy array
            Roots found by yroots.
        MSroots : numpy array
            The list of "actual" or Marching Squares roots.
        test_num : float or string
            The number of the test. For example, test 9.2 one could pass in
            "9.2" or 9.2.
        cheb_roots : numpy array
            Chebfun roots for extra comparison when MS are available.
        tol : float, optional
            The tolerance with which we want to run our tests. Defualts to
            1000*eps where eps is machine epsilon.
    Raises
    ------
        AssertionError
            If len(yroots) != len(roots) or if it fails the residual
            or norm tests.
    """
    print ("=========================================================")
    print("Test " + str(test_num))

    residuals_pass = residuals_pass_or_fail(funcs, yroots, tol)
    if residuals_pass:
        print("\t Residual test: pass")
    else:
        print("\t Residual test: fail")

    if cheb_roots is not None:
        if residuals_pass_or_fail(funcs, cheb_roots, tol):
            print("\t Chebfun passes residual test")
        else:
            print("\t Chebfun fails residual test")
        try:
            norm_pass, x_norm, y_norm = norm_pass_or_fail(polished_roots, cheb_roots, tol)
            if norm_pass:
                print("\t Chebfun norm test: pass")
            else:
                print("\t Chebfun norm test: fail")
            print("The norm of the difference in x values:", x_norm)
            print("The norm of the difference in y values:", y_norm)
        except ValueError as e:
            print("A different number of roots were found.")
            print ("Yroots: " + str(len(yroots)))
            print("Chebfun Roots: " + str(len(cheb_roots)))
    if polished_roots is not None:
        try:
            norm_pass, x_norm, y_norm = norm_pass_or_fail(yroots, polished_roots, tol)
            if norm_pass:
                print("\t YRoots norm test: pass")
            else:
                print("\t YRoots norm test: fail")
            print("The norm of the difference in x values:", x_norm)
            print("The norm of the difference in y values:", y_norm)
        except ValueError as e:
                print("A different number of roots were found.")
                print ("Yroots: " + str(len(yroots)))
                print("Polished: " + str(len(polished_roots)))
    print("YRoots max residuals:")
    YR_resid = list()
    for i, func in enumerate(funcs):
        YR_resid.append(residuals(func, yroots))
        print("\tf" + str(i) + ": " + str(np.max(residuals(func, yroots))))

    cheb_resid = None
    if cheb_roots is not None:
        cheb_resid = list()
        print("Chebfun max residuals:")
        for i, func in enumerate(funcs):
            cheb_resid.append(residuals(func, cheb_roots))
            print("\tf" + str(i) + ": " + str(np.max(residuals(func, cheb_roots))))
    if polished_roots is not None:
        print("Polished max residuals:")
        Other_resid = list()
        for i, func in enumerate(funcs):
            Other_resid.append(residuals(func, polished_roots))
            print("\tf" + str(i) + ": " + str(np.max(residuals(func, polished_roots))))

        if len(yroots) > len(polished_roots):
            print("YRoots found more roots.")
            print("=========================================================")
            return residuals_pass,norm_pass

    # print("Comparison of Residuals (YRoots <= Other)")
    num_smaller = 0
    if polished_roots is not None:
        for i in range(len(YR_resid)):
            comparison_array = (YR_resid[i] <= Other_resid[i])
            # print(comparison_array)
            num_smaller += np.sum(comparison_array)
        print("Number of YRoots residual values <= Polished residual values are: " + str(num_smaller))

    if cheb_resid is not None:
        if len(yroots) > len(cheb_roots):
            print("=========================================================")
            return residuals_pass,norm_pass

        for i in range(len(YR_resid)):
            comparison_array2 = (YR_resid[i] <= cheb_resid[i])
            num_smaller += np.sum(comparison_array2)
    print("Number of YRoots residual values <= to Chebfun residual values are: " + str(num_smaller))

    print("=========================================================")
    return residuals_pass,norm_pass

#TODO: include test cases for for returnbounding boxes, exact, rescale

def test_roots_1_1():
    # Test 1.1
    f = lambda x,y: 144*(x**4+y**4)-225*(x**2+y**2) + 350*x**2*y**2+81
    g = lambda x,y: y-x**6
    funcs = [f,g]
    a, b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_1.1.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_1.1.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 1.1, cheb_roots=chebfun_roots)


def test_roots_1_2():
    # Test 1.2
    f = lambda x,y: (y**2-x**3)*((y-0.7)**2-(x-0.3)**3)*((y+0.2)**2-(x+0.8)**3)*((y+0.2)**2-(x-0.8)**3)
    g = lambda x,y: ((y+.4)**3-(x-.4)**2)*((y+.3)**3-(x-.3)**2)*((y-.5)**3-(x+.6)**2)*((y+0.3)**3-(2*x-0.8)**3)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start

    #TODO: SPEAK TO KATE ABOUT THIS
    # Get Polished results (Newton polishing misses roots)
    # yroots2 = solve([f,g],[-1,-1],[1,1], abs_approx_tol=[1e-8, 1e-12], rel_approx_tol=[1e-15, 1e-18],\
    #             max_cond_num=[1e5, 1e2], good_zeros_factor=[100,100], min_good_zeros_tol=[1e-5, 1e-5],\
    #             check_eval_error=[True,True], check_eval_freq=[1,2], plot=False, target_tol=[1e-13, 1e-13])
    actual_roots = np.load('Polished_results/polished_1.2.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_1.2.csv', delimiter=',')

    #return t, verbose_pass_or_fail([f,g], yroots, yroots2, 1.2, cheb_roots=chebfun_roots, tol=2.220446049250313e-10)
    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 1.2, cheb_roots=chebfun_roots, tol=2.220446049250313e-10)


def test_roots_1_3():
    # Test 1.3
    f = lambda x,y: y**2-x**3
    g = lambda x,y: (y+.1)**3-(x-.1)**2
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_1.3.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_1.3.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 1.3, cheb_roots=chebfun_roots)

def test_roots_1_4():
    # Test 1.4
    f = lambda x,y: x - y + .5
    g = lambda x,y: x + y
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    # Single root has to be in matrix form because yroots
    # returns the roots in matrix form.
    a_roots = np.array([[-.25, .25]])
    chebfun_roots = np.array([np.loadtxt('Chebfun_results/test_roots_1.4.csv', delimiter=',')])

    return t, verbose_pass_or_fail([f,g], yroots, a_roots, 1.4, cheb_roots=chebfun_roots)

def test_roots_1_5():
    # Test 1.5
    f = lambda x,y: y + x/2 + 1/10
    g = lambda x,y: y - 2.1*x + 2
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    # Single root has to be in matrix form because yroots
    # returns the roots in matrix form.
    a_roots = np.array([[0.730769230769231, -0.465384615384615]])

    chebfun_roots = np.array([np.loadtxt('Chebfun_results/test_roots_1.5.csv', delimiter=',')])

    return t, verbose_pass_or_fail([f,g], yroots, a_roots, 1.5, cheb_roots=chebfun_roots)


def test_roots_2_1():
    # Test 2.1
    f = lambda x,y: np.cos(10*x*y)
    g = lambda x,y: x + y**2
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_2.1.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_2.1.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 2.1, cheb_roots=chebfun_roots)


def test_roots_2_2():
    # Test 2.2
    f = lambda x,y: x
    g = lambda x,y: (x-.9999)**2 + y**2-1
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_2.2.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_2.2.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 2.2, cheb_roots=chebfun_roots)


def test_roots_2_3():
    # Test 2.3
    f = lambda x,y: np.sin(4*(x + y/10 + np.pi/10))
    g = lambda x,y: np.cos(2*(x-2*y+ np.pi/7))
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_2.3.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_2.3.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 2.3, cheb_roots=chebfun_roots)


def test_roots_2_4():
    # Test 2.4
    f = lambda x,y: np.exp(x-2*x**2-y**2)*np.sin(10*(x+y+x*y**2))
    g = lambda x,y: np.exp(-x+2*y**2+x*y**2)*np.sin(10*(x-y-2*x*y**2))
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_2.4.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_2.4.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 2.4, cheb_roots=chebfun_roots)


def test_roots_2_5():
    # Test 2.5
    f = lambda x,y: 2*y*np.cos(y**2)*np.cos(2*x)-np.cos(y)
    g = lambda x,y: 2*np.sin(y**2)*np.sin(2*x)-np.sin(x)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_2.5.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_2.5.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 2.5, cheb_roots=chebfun_roots, tol=2.220446049250313e-12)



def test_roots_3_1():
    # Test 3.1
    f = lambda x,y: ((x-.3)**2+2*(y+0.3)**2-1)
    g = lambda x,y: ((x-.49)**2+(y+.5)**2-1)*((x+0.5)**2+(y+0.5)**2-1)*((x-1)**2+(y-0.5)**2-1)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_3.1.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_3.1.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 3.1, cheb_roots=chebfun_roots, tol=2.220446049250313e-11)

def test_roots_3_2():
    # Test 3.2
    f = lambda x,y: ((x-0.1)**2+2*(y-0.1)**2-1)*((x+0.3)**2+2*(y-0.2)**2-1)*((x-0.3)**2+2*(y+0.15)**2-1)*((x-0.13)**2+2*(y+0.15)**2-1)
    g = lambda x,y: (2*(x+0.1)**2+(y+0.1)**2-1)*(2*(x+0.1)**2+(y-0.1)**2-1)*(2*(x-0.3)**2+(y-0.15)**2-1)*((x-0.21)**2+2*(y-0.15)**2-1)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_3.2.npy')

    #speak to KATE about this
    # yroots2 = solve([f,g],[-1,-1],[1,1], abs_approx_tol=[1e-8, 1e-15], rel_approx_tol=[1e-12, 1e-29],\
    #             max_cond_num=[1e5, 1e2], good_zeros_factor=[100,100], min_good_zeros_tol=[1e-5, 1e-5],\
    #             check_eval_error=[True,True], check_eval_freq=[1,1], plot=False)

    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_3.2.csv', delimiter=',')
    actual_roots = chebfun_roots

    #return t, verbose_pass_or_fail([f,g], yroots, yroots2, 3.2, cheb_roots=chebfun_roots, tol=2.220446049250313e-11)
    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 3.2, cheb_roots=chebfun_roots, tol=2.220446049250313e-11)


def test_roots_4_1():
    # Test 4.1
    # This system hs 4 true roots, but ms fails (finds 5).
    f = lambda x,y: np.sin(3*(x+y))
    g = lambda x,y: np.sin(3*(x-y))
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_4.1.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_4.1.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 4.1, cheb_roots=chebfun_roots)

def test_roots_4_2():
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
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_4.2.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_4.2.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 4.2, cheb_roots=chebfun_roots)



def test_roots_5():
    # Test 5.1
    f = lambda x,y: 2*x*y*np.cos(y**2)*np.cos(2*x)-np.cos(x*y)
    g = lambda x,y: 2*np.sin(x*y**2)*np.sin(3*x*y)-np.sin(x*y)
    funcs = [f,g]
    a,b = np.array([-2,-2]), np.array([2,2])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_5.1.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_5.1.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 5.1, cheb_roots=chebfun_roots)


def test_roots_6_1():
    # Test 6.1
    f = lambda x,y: (y - 2*x)*(y+0.5*x)
    g = lambda x,y: x*(x**2+y**2-1)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_6.1.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_6.1.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 6.1, cheb_roots=chebfun_roots)




def test_roots_6_2():
    # Test 6.2
    f = lambda x,y: (y - 2*x)*(y+.5*x)
    g = lambda x,y: (x-.0001)*(x**2+y**2-1)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.array([[1/10000,-1/20000],[1/10000, 1/5000],[-2/np.sqrt(5),1/np.sqrt(5)],[-1/np.sqrt(5),-2/np.sqrt(5)],[1/np.sqrt(5),2/np.sqrt(5)],[2/np.sqrt(5),-1/np.sqrt(5)]])
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_6.2.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 6.2, cheb_roots=chebfun_roots, tol=2.220446049250313e-11)


def test_roots_6_3():
    # Test 6.3
    f = lambda x,y: 25*x*y - 12
    g = lambda x,y: x**2+y**2-1
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_6.3.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_6.3.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 6.3, cheb_roots=chebfun_roots)


def test_roots_7_1():
    # Test 7.1
    f = lambda x,y: (x**2+y**2-1)*(x-1.1)
    g = lambda x,y: (25*x*y-12)*(x-1.1)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_7.1.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_7.1.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 7.1, cheb_roots=chebfun_roots)


def test_roots_7_2():
    # Test 7.2
    f = lambda x,y: y**4 + (-1)*y**3 + (2*x**2)*(y**2) + (3*x**2)*y + (x**4)
    h = lambda x,y: y**10-2*(x**8)*(y**2)+4*(x**4)*y-2
    g = lambda x,y: h(2*x,2*(y+.5))
    funcs = [f,g,h]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_7.2.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_7.2.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 7.2, cheb_roots=chebfun_roots, tol=2.220446049250313e-10)


def test_roots_7_3():
    # Test 7.3
    c = 1.e-09
    f = lambda x,y: np.cos(x*y/(c**2))+np.sin(3*x*y/(c**2))
    g = lambda x,y: np.cos(y/c)-np.cos(2*x*y/(c**2))
    funcs = [f,g]
    a,b = np.array([-1e-9, -1e-9]), np.array([1e-9, 1e-9])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_7.3.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_7.3.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 7.3, cheb_roots=chebfun_roots,tol=2.220446049250313e-10)



def test_roots_7_4():
    # Test 7.4
    f = lambda x,y: np.sin(3*np.pi*x)*np.cos(x*y)
    g = lambda x,y: np.sin(3*np.pi*y)*np.cos(np.sin(x*y))
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_7.4.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_7.4.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 7.4, cheb_roots=chebfun_roots)

def test_roots_8_1():
    # Test 8.1
    f = lambda x,y: np.sin(10*x-y/10)
    g = lambda x,y: np.cos(3*x*y)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_8.1.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_8.1.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 8.1, cheb_roots=chebfun_roots)

def test_roots_8_2():
    # Test 8.2
    f = lambda x,y: np.sin(10*x-y/10) + y
    g = lambda x,y: np.cos(10*y-x/10) - x
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_8.2.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_8.2.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 8.2, cheb_roots=chebfun_roots)



def test_roots_9_1():
    # Test 9.1
    f = lambda x,y: x**2+y**2-.9**2
    g = lambda x,y: np.sin(x*y)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_9.1.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_9.1.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 9.1, cheb_roots=chebfun_roots)


def test_roots_9_2():
    # Test 9.2
    f = lambda x,y: x**2+y**2-.49**2
    g = lambda x,y: (x-.1)*(x*y - .2)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.load('Polished_results/polished_9.2.npy')
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_9.2.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 9.2, cheb_roots=chebfun_roots)


def test_roots_10():
    # Test 10.1
    f = lambda x,y: (x-1)*(np.cos(x*y**2)+2)
    g = lambda x,y: np.sin(8*np.pi*y)*(np.cos(x*y)+2)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = solve(funcs,a,b)
    t = time() - start
    actual_roots = np.array([[1, -1.0], [1, -0.875], [1, -0.75], [1, -0.625], [1, -0.5], [1, -0.375],
                            [1, -0.25], [1, -0.125], [1, 0.0], [1, 0.125], [1, 0.25], [1, 0.375],
                            [1, 0.5], [1, 0.625], [1, 0.75], [1, 0.875], [1, 1.0]])
    chebfun_roots = np.loadtxt('Chebfun_results/test_roots_10.1.csv', delimiter=',')

    return t, verbose_pass_or_fail([f,g], yroots, actual_roots, 10.1, cheb_roots=chebfun_roots)

def plot_timings(tests,timings):
    labels = [test.__name__[11:].replace('_','.') for test in tests]
    plt.figure(figsize=(8,5))
    plt.subplot(211)
    plt.bar(labels,timings)
    plt.xticks(rotation=45)
    plt.ylim(0,40)
    plt.subplot(212)
    plt.bar(labels,timings)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.ylim((10**-3,10**2))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run all the tests!
    tests = np.array([test_roots_1_1,
                        test_roots_1_2,
                        test_roots_1_3,
                        test_roots_1_4,
                        test_roots_1_5,
                        test_roots_2_1,
                        test_roots_2_2,
                        test_roots_2_3,
                        test_roots_2_4,
                        test_roots_2_5,
                        test_roots_3_1,
                        test_roots_3_2,
                        test_roots_4_1,
                        test_roots_4_2,
                        test_roots_5,
                        test_roots_6_1,
                        test_roots_6_2,
                        test_roots_6_3,
                        test_roots_7_1,
                        test_roots_7_2,
                        test_roots_7_3,
                        test_roots_7_4,
                        test_roots_8_1,
                        test_roots_8_2,
                        test_roots_9_1,
                        test_roots_9_2,
                        test_roots_10])
    res_passes = np.zeros_like(tests,dtype=bool)
    norm_passes = np.zeros_like(tests,dtype=bool)
    times = np.zeros_like(tests)
    for i,test in enumerate(tests):
        t, passes = test()
        res_pass,norm_pass = passes
        res_passes[i] = res_pass
        norm_passes[i] = norm_pass
        times[i] = t
    print('\n\nSummary')
    print(f'Residual Test: Passed {np.sum(res_passes)} of 27, {100*np.mean(res_passes)}%')
    where_failed_res = np.where(~res_passes)[0]
    failed_res_tests = tests[where_failed_res]
    print(f'Failed Residual Test on \n{[t.__name__ for t in failed_res_tests]}')
    print(f'Norm Test    : Passed {np.sum(norm_passes)} of 27, {100*np.mean(norm_passes)}%')
    where_failed_norm = np.where(~norm_passes)[0]
    failed_norm_tests = tests[where_failed_norm]
    print(f'Failed Norm Test on \n{[t.__name__ for t in failed_norm_tests]}')
    plot_timings(tests,times)
