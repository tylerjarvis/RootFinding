# Define the tests for pass/fail, residuals

import numpy as np

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

def verbose_pass_or_fail(funcs, yroots, MSroots, test_num, cheb_roots=None, tol=2.220446049250313e-13):
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

    if residuals_pass_or_fail(funcs, yroots, tol):
        print("\t Residual test: pass")
    else:
        print("\t Residual test: fail")
    
    if cheb_roots is not None:
        if residuals_pass_or_fail(funcs, cheb_roots, tol):
            print("\t Chebfun passes residual test")
        else:
            print("\t Chebfun fails residual test")
        try:
            result, x_norm, y_norm = norm_pass_or_fail(yroots, cheb_roots, tol)
            if result:
                print("\t Chebfun norm test: pass")
            else:
                print("\t Chebfun norm test: fail")
            print("The norm of the difference in x values:", x_norm)
            print("The norm of the difference in y values:", y_norm)
        except ValueError as e:
            print("A different number of roots were found.")
            print ("Yroots: " + str(len(yroots)))
            print("Chebfun Roots: " + str(len(cheb_roots)))
    if MSroots is not None:
        try:
            result, x_norm, y_norm = norm_pass_or_fail(yroots, MSroots, tol)
            if result:
                print("\t MS/Actual norm test: pass")
            else:
                print("\t MS/Actual norm test: fail")
            print("The norm of the difference in x values:", x_norm)
            print("The norm of the difference in y values:", y_norm)
        except ValueError as e:
                print("A different number of roots were found.")
                print ("Yroots: " + str(len(yroots)))
                print("MS/Actual: " + str(len(MSroots)))
        
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
    if MSroots is not None:
        print("MS/Actual max residuals:")
        Other_resid = list()
        for i, func in enumerate(funcs):
            Other_resid.append(residuals(func, MSroots))
            print("\tf" + str(i) + ": " + str(np.max(residuals(func, MSroots))))

        if len(yroots) > len(MSroots):
            print("YRoots found more roots.")
            print("=========================================================")
            return

    # print("Comparison of Residuals (YRoots <= Other)")
    num_smaller = 0
    if MSroots is not None:
        for i in range(len(YR_resid)):
            comparison_array = (YR_resid[i] <= Other_resid[i])
            # print(comparison_array)
            num_smaller += np.sum(comparison_array)
        print("Number of YRoots residual values <= MS/Actual residual values are: " + str(num_smaller))
    
    if cheb_resid is not None:
        if len(yroots) > len(cheb_roots):
            print("=========================================================")
            return
        
        for i in range(len(YR_resid)):
            comparison_array2 = (YR_resid[i] <= cheb_resid[i])
            num_smaller += np.sum(comparison_array2)
    print("Number of YRoots residual values <= to Chebfun residual values are: " + str(num_smaller))

    print("=========================================================")