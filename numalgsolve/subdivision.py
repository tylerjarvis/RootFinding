"""
Subdivision provides a solve function that finds roots of a set of functions
by approximating the functions with Chebyshev polynomials.
When the approximation is performed on a sufficiently small interval,
the approximation degree is small enough to be solved efficiently.

"""

import numpy as np
from numpy.fft.fftpack import rfftn
from numalgsolve.OneDimension import divCheb,divPower,multCheb,multPower,solve
from numalgsolve.Division import division
from numalgsolve.utils import clean_zeros_from_matrix, slice_top
from numalgsolve.polynomial import MultiCheb
from itertools import product

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def solve(funcs, a, b):
    '''
    Finds the real roots of the given list of functions on a given interval.

    Parameters
    ----------
    funcs : list of callable functions
        Functions to find the common roots of.
    a : numpy array
        The lower bound on the interval.
    b : numpy array
        The upper bound on the interval.
    returns
    -------
    roots : numpy array
        The common roots of the polynomials. Each row is a root.
    '''
    dim = len(a)
    if dim == 1:
        #one dimensional case
        zeros = np.unique(subdivision_solve_1d(funcs[0],a,b))
        #Finds the roots of each succesive function and checks which roots are common.
        for func in funcs[1:]:
            if len(zeros) == 0:
                break
            zeros2 = np.unique(subdivision_solve_1d(func,a,b))
            common_zeros = []
            tol = 1.e-10
            for zero in zeros2:
                spot = np.where(np.abs(zeros-zero)<tol)
                if len(spot[0]) > 0:
                    common_zeros.append(zero)
            zeros = common_zeros
        return zeros
    else:
        #multi-dimensional case
        #choose an appropriate max degree for the given dimension
        deg_dim = {2:10, 3:5, 4:4}
        if dim > 4:
            deg = 2
        else:
            deg = deg_dim[dim]

        result = subdivision_solve_nd(funcs,a,b,deg)
        print("Total intervals checked was {}".format(total_intervals))
        print("The percent thrown out by each checker was {}".format((100*thrown_out / total_intervals).round(2)))
        return result

def transform(x,a,b):
    """Transforms points from the interval [-1,1] to the interval [a,b].

    Parameters
    ----------
    x : numpy array
        The points to be tranformed.
    a : float or numpy array
        The lower bound on the interval. Float if one-dimensional, numpy array if multi-dimensional
    b : float or numpy array
        The upper bound on the interval. Float if one-dimensional, numpy array if multi-dimensional

    Returns
    -------
    transform : numpy array
        The transformed points.
    """
    return ((b-a)*x+(b+a))/2

def inv_transform(x,a,b):
    """Transforms points from the interval [a,b] to the interval [-1,1].

    Parameters
    ----------
    x : numpy array
        The points to be tranformed.
    a : float or numpy array
        The lower bound on the interval. Float if one-dimensional, numpy array if multi-dimensional
    b : float or numpy array
        The upper bound on the interval. Float if one-dimensional, numpy array if multi-dimensional

    Returns
    -------
    transform : numpy array
        The transformed points.
    """
    return (2*x-b-a)/(b-a)

def interval_approximate_1d(f,a,b,deg):
    """Finds the chebyshev approximation of a one-dimensional function on an interval.

    Parameters
    ----------
    f : function from R -> R
        The function to interpolate.
    a : float
        The lower bound on the interval.
    b : float
        The upper bound on the interval.
    deg : int
        The degree of the interpolation.

    Returns
    -------
    coeffs : numpy array
        The coefficient of the chebyshev interpolating polynomial.
    """
    extrema = transform(np.cos((np.pi*np.arange(2*deg))/deg),a,b)
    values = f(extrema)
    coeffs = np.real(np.fft.fft(values/deg))
    coeffs[0]/=2
    coeffs[deg]/=2
    return coeffs[:deg+1]

def interval_approximate_nd(f,a,b,degs):
    """Finds the chebyshev approximation of an n-dimensional function on an interval.

    Parameters
    ----------
    f : function from R^n -> R
        The function to interpolate.
    a : numpy array
        The lower bound on the interval.
    b : numpy array
        The upper bound on the interval.
    deg : numpy array
        The degree of the interpolation in each dimension.

    Returns
    -------
    coeffs : numpy array
        The coefficient of the chebyshev interpolating polynomial.
    """
    if len(a)!=len(b):
        raise ValueError("Interval dimensions must be the same!")

    dim = len(a)
    n = degs[0]

    if hasattr(f,"evaluate_grid"):
        #for polynomials, we can quickly evaluate all points in a grid
        #xyz does not contain points, but the nth column of xyz has the values needed
        #along the nth axis. The direct product of these values procuces the grid
        cheb_values = np.cos(np.arange(2*n)*np.pi/n)
        xyz = transform(np.column_stack([cheb_values]*dim), a, b)
        values = f.evaluate_grid(xyz)
    else:
        #if function f has no "evaluate_grid" method,
        #we evaluate each point individually
        cheb_values = np.cos(np.arange(2*n)*np.pi/n)
        cheb_grids = np.meshgrid(*([cheb_values]*dim), indexing='ij')

        flatten = lambda x: x.flatten()
        cheb_points = transform(np.column_stack(map(flatten, cheb_grids)), a, b)
        values = f(cheb_points).reshape(2*n,2*n)

    coeffs = rfftn(values/np.product(degs))

    for i in range(dim):
        #construct slices for the first and degs[i] entry in each dimension
        idx0 = [slice(None)] * dim
        idx0[i] = 0

        idx_deg = [slice(None)] * dim
        idx_deg[i] = degs[i]

        #halve the coefficients in each slice
        coeffs[idx0] /= 2
        coeffs[idx_deg] /= 2

    slices = []
    for i in range(dim):
        slices.append(slice(0,degs[i]+1))

    return coeffs[slices]

def get_subintervals(a,b,dimensions):
    """Gets the subintervals to divide a matrix into.

    Parameters
    ----------
    a : numpy array
        The lower bound on the interval.
    b : numpy array
        The upper bound on the interval.
    dimensions : numpy array
        The dimensions we want to cut in half.

    Returns
    -------
    subintervals : list
        Each element of the list is a tuple containing an a and b, the lower and upper bounds of the interval.
    """
    RAND = 0.5139303900908738
    subintervals = []
    diffs1 = ((b-a)*RAND)[dimensions]
    diffs2 = ((b-a)-(b-a)*RAND)[dimensions]

    for subset in product([False,True], repeat=len(dimensions)):
        subset = np.array(subset)
        aTemp = a.copy()
        bTemp = b.copy()
        aTemp[dimensions] += (~subset)*diffs1
        bTemp[dimensions] -= subset*diffs2
        subintervals.append((aTemp,bTemp))
    return subintervals

def full_cheb_approximate(f,a,b,deg,max_deg,tol=1.e-8, coeff = None):
    """Gives the full chebyshev approximation and checks if it's good enough.

    Called recursively.

    Parameters
    ----------
    f : function
        The function we approximate.
    a : numpy array
        The lower bound on the interval.
    b : numpy array
        The upper bound on the interval.
    deg : int
        The degree to approximate with.
    max_deg : int
        The maximum degree before giving up on interpolation and returning none
    tol : float
        How small the high degree terms must be to consider the approximation accurate.
    coeff : numpy array
        The starting interpolation. Used to recursively get a higher approximation faster.

    Returns
    -------
    coeff : numpy array
        The coefficient array of the interpolation. If it can't get a good approximation and needs to subdivide, returns None.
    """
    if deg > max_deg:
        return None
    dim = len(a)
    degs = np.array([deg]*dim)
    if coeff is None:
        coeff = interval_approximate_nd(f,a,b,degs)
    coeff2 = interval_approximate_nd(f,a,b,degs*2)
    coeff2[slice_top(coeff)] -= coeff
    clean_zeros_from_matrix(coeff2,1.e-16)
    if np.sum(np.abs(coeff2)) > tol:
        coeff2[slice_top(coeff)] += coeff
        return full_cheb_approximate(f,a,b,2*deg,max_deg,tol=tol, coeff = coeff2)
    else:
        return coeff

def good_zeros_nd(zeros, imag_tol = 1.e-10):
    """Get the real zeros in the -1 to 1 interval in each dimension.

    Parameters
    ----------
    zeros : numpy array
        The zeros to be checked.
    imag_tol : float
        How large the imaginary part can be to still have it be considered real.

    Returns
    -------
    good_zeros : numpy array
        The real zero in [-1,1] of the input zeros.
    """
    good_zeros = zeros[np.all(np.abs(zeros.imag) < imag_tol,axis = 1)]
    good_zeros = good_zeros[np.all(np.abs(good_zeros) <= 1,axis = 1)]
    return good_zeros


"""
The check functions are all functions that take in a coefficent matrix and run a quick check
to determine if there can ever be zeros on the unit box there. They are then put into the list
all_bound_check_functions in the order we want to run them (probably fastest first). These are
then all run to throw out intervals as possible.
"""

def ext_val3(test_coeff, maxx = True):
    a,b,c = test_coeff
    """Absolute value of max or min of a + bx + c(2x^2 - 1) on -1 to 1"""
    if np.abs(c) < 1.e-10:
        if maxx:
            return abs(a) + abs(b)
        else:
            if abs(b) > abs(a):
                return 0
            else:
                return abs(a) - abs(b)
    else:
        vals = [a - b + c, a + b + c] #at +-1
        if np.abs(b/c) < 4:
            vals.append(a - b**2/(8*c) - c) #at -b/(4c)
        if maxx:
            return max(np.abs(vals))
        else:
            vals = np.array(vals)
            if np.any(vals > 0) and np.any(vals < 0):
                return 0
            else:
                return min(np.abs(vals))

def ext_val4(test_coeff, maxx = True):
    a,b,c,d = test_coeff
    """Absolute value of max or min of a + bx + c(2x^2 - 1) + d*(4x^3 - 3x) on -1 to 1"""
    if np.abs(d) < 1.e-10:
        return ext_val3([a,b,c], maxx = maxx)
    else:
        vals = [a - b + c - d, a + b + c + d] #at +-1

        #The quadratic roots
        if 16*c**2 >= 48*d*(b-3*d):
            x1 = (-4*c + np.sqrt(16*c**2 - 48*d*(b-3*d))) / (24*d)
            x2 = (-4*c - np.sqrt(16*c**2 - 48*d*(b-3*d))) / (24*d)
            if np.abs(x1) < 1:
                vals.append(a + b*x1 + c*(2*x1**2 - 1) + d*(4*x1**3 - 3*x1))
            if np.abs(x2) < 1:
                vals.append(a + b*x2 + c*(2*x2**2 - 1) + d*(4*x2**3 - 3*x2))
        if maxx:
            return max(np.abs(vals))
        else:
            vals = np.array(vals)
            if np.any(vals > 0) and np.any(vals < 0):
                return 0
            else:
                return min(np.abs(vals))

def constant_term_check(test_coeff):
    """Quick check of zeros in the unit box.

    Checks if the constant term is bigger than all the other terms combined, using the fact that
    each Chebyshev monomial is bounded by 1.

    Parameters
    ----------
    coeff : numpy array
        The coefficient matrix of the polynomial to check

    Returns
    -------
    check1 : bool
        False if there are no zeros in the unit box, True otherwise
    """
    test_sum = np.sum(np.abs(test_coeff))
    if np.abs(test_coeff.flatten()[0]) * 2 > test_sum:
        return False
    else:
        return True

def check2(test_coeff):
    """Quick check of zeros in the unit box.

    Parameters
    ----------
    coeff : numpy array
        The coefficient matrix of the polynomial to check

    Returns
    -------
    check1 : bool
        False if there are no zeros in the unit box, True otherwise
    """
    start = ext_val3(test_coeff[0,0:3], maxx = False)
    rest = 0
    for i in range(1, test_coeff.shape[0]):
        rest += ext_val3(test_coeff[i,0:3])
    rest += np.sum(np.abs(test_coeff[:,3:]))
    if start > rest:
        return False

    start = ext_val3(test_coeff[0:3,0], maxx = False)
    rest = 0
    for i in range(1, test_coeff.shape[1]):
        rest += ext_val3(test_coeff[0:3,i])
    rest += np.sum(np.abs(test_coeff[3:]))
    if start > rest:
        return False

    return True

def check3(test_coeff):
    """Quick check of zeros in the unit box.

    Parameters
    ----------
    coeff : numpy array
        The coefficient matrix of the polynomial to check

    Returns
    -------
    check1 : bool
        False if there are no zeros in the unit box, True otherwise
    """
    start = ext_val4(test_coeff[0,0:4], maxx = False)
    rest = 0
    for i in range(1, test_coeff.shape[0]):
        rest += ext_val4(test_coeff[i,0:4])
    rest += np.sum(np.abs(test_coeff[:,4:]))
    if start > rest:
        return False

    start = ext_val4(test_coeff[0:4,0], maxx = False)
    rest = 0
    for i in range(1, test_coeff.shape[1]):
        rest += ext_val4(test_coeff[0:4,i])
    rest += np.sum(np.abs(test_coeff[4:]))
    if start > rest:
        return False

    return True

def linear_check(test_coeff_in, intervals, a, b):
    """Quick check of zeros in intervals.

    Parameters
    ----------
    test_coeff_in : numpy array
        The coefficient matrix of the polynomial to check
    intervals : list
        A list of the intervals we want to check before subdividing them
    a : numpy array
        The lower bound on the interval.
    b : numpy array
        The upper bound on the interval.

    Returns
    -------
    intervals : list
        The intervals we actually need to check. All the ones we can remove are gone.
    """
    test_coeff = test_coeff_in.copy()
    for i in range(len(intervals)):
        intervals[i] = tuple([inv_transform(intervals[i][j], a, b) for j in range(2)])

    mask = []
    for interval in intervals:
        spot = [0]*len(a)
        const = test_coeff[tuple(spot)]
        test_coeff[tuple(spot)] = 0
        if const == 0:
            mask.append(True)
            continue
        lin_change = 0
        for dim in range(len(a)):
            spot_temp = spot.copy()
            spot_temp[dim] = 1
            temp_coeff = test_coeff[tuple(spot_temp)]
            test_coeff[tuple(spot_temp)] = 0
            interval_temp = np.array(interval)
            if const < 0:
                lin_change += max(temp_coeff * interval_temp[:,dim])
            else:
                lin_change += min(temp_coeff * interval_temp[:,dim])

        if const > 0:
            if -lin_change > const:
                mask.append(True)
                continue
        else:
            if -lin_change < const:
                mask.append(True)
                continue

        const += lin_change
        if np.abs(const) > np.sum(np.abs(test_coeff)):
            mask.append(False)
            continue
        mask.append(True)

    old_intervals = intervals
    intervals = []
    for i in range(len(old_intervals)):
        if(mask[i]):
            intervals.append(old_intervals[i])

    for i in range(len(intervals)):
        intervals[i] = transform(intervals[i], a, b)
    return intervals

def quadratic_check1(test_coeff):
    """Quick check of zeros in the unit box using the x^2 terms

    Parameters
    ----------
    coeff : numpy array
        The coefficient matrix of the polynomial to check

    Returns
    -------
    check1 : bool
        False if there are no zeros in the unit box, True otherwise
    """
    #check using |c0 + c1x + c2y +c3x^2|
    c0 = test_coeff[0,0]
    c1 = test_coeff[1,0]
    c2 = test_coeff[0,1]
    c3 = test_coeff[2,0]

    #if c3 != 0, same as a linear check
    if np.isclose(c3, 0):
        return True

    def quadratic_formula_check(y):
        """given a fixed value of y, uses the quadratic formula
            to see if c0 + c1x + c2y +c3x^2 = 0
            for some x in [a0, b0]"""
        discriminant = c1**2 - 4*(c2*y+c0)*c3
        if np.isclose(discriminant, 0) and a[0] < -c1/2/c3 < b[0]:
             return True
        elif discriminant > 0 and \
              (a[0] < (-c1+np.sqrt(discriminant))/2/c3 < b[0] or \
               a[0] < (-c1-np.sqrt(discriminant))/2/c3 < b[0]):
            return True
        else:
            return False

    #If c0 + c1x + c2y +c3x^2 = 0 in the region, useless check.
    if np.isclose(c2, 0) and quadratic_formula_check(0):
        return True
    else:
        y = lambda x: (-c3 *x**2 - c1 * x - c0)/c2
        if a[1] < y(a[0]) < b[1] or a[1] < y(b[0]) < b[1]:
            return True
        elif quadratic_formula_check(a[0]) or quadratic_formula_check(b[0]):
            return True

    #function for evaluating |c0 + c1x + c2y +c3x^2|
    eval = lambda x,y: abs(c0 + c1*x + c2*y + c3 * x**2)

    #In this case, extrema only occur on the edges since there are no critical points
    #edges 1&2: x = a0, b0 --> potential extrema at corners
    #edges 3&4: y = a1, b1 --> potential extrema at x0 = -c1/2c3, if that's in [a0, b0]
    if a[0] < -c1/2/c3 < b[0]:
        potential_minimizers = np.array([[a[0],a[1]],
                                         [a[0],b[1]],
                                         [b[0],a[1]],
                                         [b[0],b[1]],
                                         [-c1/2/c3,a[1]],
                                         [-c1/2/c3,b[1]]])
    else:
        potential_minimizers = np.array([[a[0],a[1]],
                                         [a[0],b[1]],
                                         [b[0],a[1]],
                                         [b[0],b[1]]])

    #if min{|c0 + c1x + c2y +c3x^2|} > sum of other terms in test_coeff, no roots in the region
    if min(eval(potential_minimizers)) > np.sum(np.abs(test_coeff)) - abs(c0) - abs(c1) - abs(c2) - abs(c3):
        return False
    else:
        return True

def quadratic_check2(test_coeff):
    """Quick check of zeros in the unit box using the y^2 terms

    Parameters
    ----------
    coeff : numpy array
        The coefficient matrix of the polynomial to check

    Returns
    -------
    check1 : bool
        False if there are no zeros in the unit box, True otherwise
    """
    #very similar to quadratic_check_1, but switch x and y
    #check using |c0 + c1x + c2y +c3y^2|
    c0 = test_coeff[0,0]
    c1 = test_coeff[1,0]
    c2 = test_coeff[0,1]
    c3 = test_coeff[0,2]

    #if c3 != 0, same as a linear check
    if np.isclose(c3, 0):
        return True

    def quadratic_formula_check(x):
        """given a fixed value of x, uses the quadratic formula
            to see if c0 + c1x + c2y +c3y^2 = 0
            for some y in [a1, b1]"""
        discriminant = c2**2 - 4*(c1*x+c0)*c3
        if np.isclose(discriminant, 0) and a[1] < -c2/2/c3 < b[1]:
             return True
        elif discriminant > 0 and \
              (a[1] < (-c2+np.sqrt(discriminant))/2/c3 < b[1] or \
               a[1] < (-c2-np.sqrt(discriminant))/2/c3 < b[1]):
            return True
        else:
            return False

    #If c0 + c1x + c2y +c3y^2 = 0 in the region, useless
    if np.isclose(c1, 0) and quadratic_formula_check(0):
        return True
    else:
        x = lambda y: (-c3 *y**2 - c2 * y - c0)/c1
        if a[0] < x(a[1]) < b[0] or a[0] < x(b[1]) < b[0]:
            return True
        elif quadratic_formula_check(a[1]) or quadratic_formula_check(b[1]):
            return True

    #function for evaluating |c0 + c1x + c2y +c3y^2|
    eval = lambda x,y: abs(c0 + c1*x + c2*y + c3 * y**2)

    #In this case, extrema only occur on the edges since there are no critical points
    #edges 1&2: x = a0, b0 --> potential extrema at y0 = -c2/2c3, if that's in [a1, b1]
    #edges 3&4: y = a1, b1 --> potential extrema at corners
    if a[1] < -c2/2/c3 < b[1]:
        potential_minimizers = np.array([[a[0],a[1]],
                                         [a[0],b[1]],
                                         [b[0],a[1]],
                                         [b[0],b[1]],
                                         [a[0],-c2/2/c3],
                                         [b[0],-c2/2/c3]])
    else:
        potential_minimizers = np.array([[a[0],a[1]],
                                         [a[0],b[1]],
                                         [b[0],a[1]],
                                         [b[0],b[1]]])

    #if min{|c0 + c1x + c2y +c3y^2|} > sum of other terms in test_coeff, no roots in the region
    if min(eval(potential_minimizers)) > np.sum(np.abs(test_coeff)) - abs(c0) - abs(c1) - abs(c2) - abs(c3):
        return False
    else:
        return True

def quadratic_check3(test_coeff):
    """Quick check of zeros in the unit box using the xy terms

    Parameters
    ----------
    coeff : numpy array
        The coefficient matrix of the polynomial to check

    Returns
    -------
    check1 : bool
        False if there are no zeros in the unit box, True otherwise
    """
    #check using |c0 + c1x + c2y +c3xy|
    c0 = test_coeff[0,0]
    c1 = test_coeff[1,0]
    c2 = test_coeff[0,1]
    c3 = test_coeff[1,1]

    #if c3 != 0, same as a linear check
    if np.isclose(c3, 0):
        return True

    #If c0 + c1x + c2y +c3xy = 0 in the region, useless
    #testing the vertical sides of the interval
    vert_asymptote = -c2/c3
    x = lambda y: (-c0 + c2*y)/(c1 + c3*y)
    if np.isclose(a[1], vert_asymptote):
        if a[0] < x(b[1]) < b[0]:
            return True
    elif np.isclose(b[1], vert_asymptote):
        if a[0] < x(a[1]) < b[0]:
            return True
    elif a[0] < x(a[1]) < b[0] or a[0] < x(b[1]) < b[0]:
        return True

    #testing the horizontal sides of the interval
    horiz_asymptote = -c1/c3
    y = lambda x: (-c0 + c1*x)/(c2 + c3*x)
    if np.isclose(a[0], horiz_asymptote):
        if a[1] < y(b[0]) < b[1]:
            return True
    elif np.isclose(b[0], horiz_asymptote):
        if a[1] < y(a[0]) < b[1]:
            return True
    elif a[1] < y(a[0]) < b[1] or a[1] < y(b[0]) < b[1]:
        return True

    #function for evaluating |c0 + c1x + c2y +c3xy|
    eval = lambda x,y: abs(c0 + c1*x + c2*y + c3*x*y)

    #In this case, only critical point is saddle point, so all minima occur on the edges
    #On all the edges it becomes linear, so extrema always ocur at the corners
    potential_minimizers = np.array([[a[0],a[1]],
                                         [a[0],b[1]],
                                         [b[0],a[1]],
                                         [b[0],b[1]]])

    #if min{|c0 + c1x + c2y +c3xy|} > sum of other terms in test_coeff, no roots in the region
    if min(eval(potential_minimizers)) > np.sum(np.abs(test_coeff)) - np.sum(np.abs(test_coeff[:2,:2])):
        return False
    else:
        return True

#Note checks 2 and 3 only work on 2D systems
all_bound_check_functions = [constant_term_check, check2, check3, quadratic_check1, quadratic_check2, quadratic_check3]
all_interval_check_functions = [linear_check]
thrown_out = np.zeros(len(all_bound_check_functions) + len(all_interval_check_functions))
total_intervals = 0

def subdivision_solve_nd(funcs,a,b,deg,tol=1.e-4,tol2=1.e-12):
    """Finds the common zeros of the given functions.

    Parameters
    ----------
    funcs : list
        Each element of the list is a callable function.
    a : numpy array
        The lower bound on the interval.
    b : numpy array
        The upper bound on the interval.
    deg : int
        The degree to approximate with in the chebyshev approximation.

    Returns
    -------
    good_zeros : numpy array
        The real zero in [-1,1] of the input zeros.
    """
    global total_intervals, thrown_out
    division_var = 0
    try:
        if np.random.rand() > .999:
            print("Interval - ",a,b)
        dim = len(a)
        cheb_approx_list = []
        for func in funcs:
            coeff = full_cheb_approximate(func,a,b,deg,4*deg,tol=tol)

            #Subdivides if needed.
            if coeff is None:
                intervals = get_subintervals(a,b,np.arange(dim))
                return np.vstack([subdivision_solve_nd(funcs,interval[0],interval[1],deg,tol=tol,tol2=tol2) \
                                  for interval in intervals])
            elif coeff.shape[0] > deg + 1:
                #Subdivide but run some checks on the intervals first
                intervals = get_subintervals(a,b,np.arange(dim))
                func_num = 0
                for check_func in all_interval_check_functions:
                    curr_intervals = len(intervals)
                    intervals = check_func(coeff, intervals, a, b)
                    total_intervals += curr_intervals - len(intervals)
                    thrown_out[func_num + len(all_bound_check_functions)] += curr_intervals - len(intervals)
                    func_num+=1

                return np.vstack([subdivision_solve_nd(funcs,interval[0],interval[1],deg,tol=tol,tol2=tol2) \
                                  for interval in intervals])
            else:
                coeff = trim_coeff(coeff,tol=tol, tol2=tol2)
                #Run checks to try and throw out the interval
                total_intervals += 1
                func_num = 0
                for check_func in all_bound_check_functions:
                    if not check_func(coeff):
                        thrown_out[func_num] += 1
                        return np.zeros([0,dim])
                    func_num+=1

                cheb_approx_list.append(MultiCheb(coeff))

        zeros = np.array(division(cheb_approx_list, divisor_var = 0, tol = 1.e-6))
        #for i in range(len(zeros)):
        #    zeros[i] = root = newton_polish(cheb_approx_list,zeros[i],tol = 1.e-10)
        if len(zeros) == 0:
            return np.zeros([0,dim])
        zeros = transform(good_zeros_nd(zeros),a,b)
        return zeros
    except np.linalg.LinAlgError as e:
        while division_var < len(a):
            try:
                zeros = np.array(division(cheb_approx_list, divisor_var = 0, tol = 1.e-6))
                return zeros
            except np.linalg.LinAlgError as e:
                division_var += 1
        intervals = get_subintervals(a,b,np.arange(dim))
        return np.vstack([subdivision_solve_nd(funcs,interval[0],interval[1],deg,tol=tol,tol2=tol2) \
                              for interval in intervals])

def trim_coeff(coeff, tol=1.e-8, tol2=1.e-8, drop_off_tol=1.e-9):
    """Reduce the number of coefficients and the degree.

    Parameters
    ----------
    coeff : numpy array
        The Chebyshev coefficients for approximating a function.

    Returns
    -------
    coeff : numpy array
        The reduced degree Chebyshev coefficients for approximating a function.
    """
    np.set_printoptions(linewidth=500)
    print('before trim', coeff)
    plt.clf()
    plt.subplot(121)
    plt.title("Before Trim")
    plt.matshow(coeff)
    #Cuts down in the degree we are dividing by so the division matrix is stable.
    for spot in zip(*np.where(np.abs(coeff[0]) < tol2)): #spots in coeff where coeff[0]<tol2
        #??? Why is coeff[0] instead of coeff?
        #and abs(coeff[0] - coeff[1]) < drop_off_tol
            slices = []
            slices.append(slice(0,None))
            for s in spot:
                slices.append(s)
            coeff[slices] = 0

    dim = coeff.ndim

    #Cuts out the high diagonals as much as possible to minimize polynomial degree.
    if abs(coeff[tuple([-1]*dim)]) < tol:
        coeff[tuple([-1]*dim)] = 0
        deg = np.sum(coeff.shape)-dim-1
        while True:
            mons = mon_combos_limited([0]*dim,deg,coeff.shape)
            slices = [] #becomes the indices of the terms of degree deg
            mons = np.array(mons).T
            for i in range(dim):
                slices.append(mons[i])
            if np.sum(np.abs(coeff[slices])) < tol: #L1 norm
            # and abs(coeff[slices] - coeff[slices-1]) < drop_off_tol
                coeff[slices] = 0
            else:
                break
            deg -= 1
    print('after trim', coeff)
    plt.subplot(122)
    plt.title("After Trim")
    plt.matshow(coeff)
    return coeff

def mon_combos_limited(mon, remaining_degrees, shape, cur_dim = 0):
    '''Finds all the monomials of a given degree that fits in a given shape and returns them. Works recursively.

    Very similar to mon_combos, but only returns the monomials of the desired degree.

    Parameters
    --------
    mon: list
        A list of zeros, the length of which is the dimension of the desired monomials. Will change
        as the function searches recursively.
    remaining_degrees : int
        Initially the degree of the monomials desired. Will decrease as the function searches recursively.
    shape : tuple
        The limiting shape. The i'th index of the mon can't be bigger than the i'th index of the shape.
    cur_dim : int
        The current position in the list the function is iterating through. Defaults to 0, but increases
        in each step of the recursion.

    Returns
    -----------
    answers : list
        A list of all the monomials.
    '''
    answers = []
    if len(mon) == cur_dim+1: #We are at the end of mon, no more recursion.
        if remaining_degrees < shape[cur_dim]:
            mon[cur_dim] = remaining_degrees
            answers.append(mon.copy())
        return answers
    if remaining_degrees == 0: #Nothing else can be added.
        answers.append(mon.copy())
        return answers
    temp = mon.copy() #Quicker than copying every time inside the loop.
    for i in range(min(shape[cur_dim],remaining_degrees+1)): #Recursively add to mon further down.
        temp[cur_dim] = i
        answers += mon_combos_limited(temp, remaining_degrees-i, shape, cur_dim+1)
    return answers

def good_zeros(zeros, imag_tol = 1.e-10):
    """Get the real zeros in the -1 to 1 interval

    Parameters
    ----------
    zeros : numpy array
        The zeros to be checked.
    imag_tol : float
        How large the imaginary part can be to still have it be considered real.

    Returns
    -------
    good_zeros : numpy array
        The real zero in [-1,1] of the input zeros.
    """
    zeros = zeros[np.where(np.abs(zeros) <= 1)]
    zeros = zeros[np.where(np.abs(zeros.imag) < imag_tol)]
    return zeros

def subdivision_solve_1d(f,a,b,cheb_approx_tol=1.e-10,max_degree=128):
    """Finds the roots of a one-dimensional function using subdivision and chebyshev approximation.

    Parameters
    ----------
    f : function from R^n -> R
        The function to interpolate.
    a : numpy array
        The lower bound on the interval.
    b : numpy array
        The upper bound on the interval.
    deg : int
        The degree of the interpolation.

    Returns
    -------
    coeffs : numpy array
        The coefficient of the chebyshev interpolating polynomial.
    """
    n = 2
    intitial_approx = interval_approximate_1d(f,a,b,deg = n)
    while n<=max_degree:
        coeffsN = np.zeros(2*n+1)
        coeffsN[:n+1] = intitial_approx
        coeffs2N = interval_approximate_1d(f,a,b,deg = 2*n)
        #Check if the approximation is good enough
        if np.sum(np.abs(coeffs2N - coeffsN)) < cheb_approx_tol:
            coeffs = coeffsN[:n+1]
            #Division is faster after degree 75
            if n > 75:
                return transform(good_zeros(divCheb(coeffs)),a,b)
            else:
                return transform(good_zeros(multCheb(np.trim_zeros(coeffs.copy(),trim='b'))),a,b)
        intitial_approx = coeffs2N
        n*=2
    #Subdivide the interval and recursively call the function.
    div_length = (b-a)/2
    return np.hstack([subdivision_solve_1d(f,a,b-div_length,max_degree=max_degree),\
                      subdivision_solve_1d(f,a+div_length,b,max_degree=max_degree)])
