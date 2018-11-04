"""
Subdivision provides a solve function that finds roots of a set of functions
by approximating the functions with Chebyshev polynomials.
When the approximation is performed on a sufficiently small interval,
the approximation degree is small enough to be solved efficiently.

"""

import numpy as np
from numpy.fft.fftpack import fftn
from numalgsolve.OneDimension import divCheb,divPower,multCheb,multPower,solve
from numalgsolve.Division import division
from numalgsolve.utils import clean_zeros_from_matrix, slice_top
from numalgsolve.polynomial import MultiCheb
from itertools import product
from matplotlib import pyplot as plt
from matplotlib import patches
import itertools

def solve(funcs, a, b, interval_data = False):
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
    interval_checks = [constant_term_check,full_quad_check, curvature_check]#full_cubic_check,
    subinterval_checks = [linear_check,quadratic_check1,quadratic_check2,quadratic_check3]
    interval_results = []
    for i in range(len(interval_checks) + len(subinterval_checks) + 1):
        interval_results.append([])

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
        deg_dim = {2:5, 3:4, 4:3}
        if dim > 4:
            deg = 2
        else:
            deg = deg_dim[dim]

        #Output the interval percentages
        result = subdivision_solve_nd(funcs,a,b,deg,interval_results,interval_checks,subinterval_checks)

        #Plot what happened
        if interval_data:

            results_numbers = np.array([len(i) for i in interval_results])
            total_intervals = sum(results_numbers)
            checkers = [func.__name__ for func in interval_checks]+[func.__name__ for func in subinterval_checks]+["Division"]

            print("Total intervals checked was {}".format(total_intervals))
            print("Methods used were {}".format(checkers))
            print("The percent solved by each was {}".format((100*results_numbers / total_intervals).round(2)))

            if dim == 2:
                colors = ['b', 'g', 'r', 'm', 'c', 'y', 'k','w','pink','fuchsia']
                fig,ax = plt.subplots(1)
                fig.set_size_inches(10, 10)
                for i in range(len(interval_checks)):
                    results = interval_results[i]
                    first = True
                    for data in results:
                        a0,b0 = data
                        if first:
                            first = False
                            rect = patches.Rectangle((a0[0],a0[1]),b0[0]-a0[0],b0[1]-a0[1],linewidth=.001,\
                                                     edgecolor=colors[i],facecolor=colors[i]\
                                                     , label = interval_checks[i].__name__)
                        else:
                            rect = patches.Rectangle((a0[0],a0[1]),b0[0]-a0[0],b0[1]-a0[1],linewidth=.001\
                                                     ,edgecolor=colors[i],facecolor=colors[i])
                        ax.add_patch(rect)

                for i in range(len(interval_checks), len(interval_checks) + len(subinterval_checks)):
                    results = interval_results[i]
                    first = True
                    for data in results:
                        a0,b0 = data
                        if first:
                            first = False
                            rect = patches.Rectangle((a0[0],a0[1]),b0[0]-a0[0],b0[1]-a0[1],linewidth=.001\
                                                     ,edgecolor=colors[i],facecolor=colors[i]\
                                                     , label = subinterval_checks[i - len(interval_checks)].__name__)
                        else:
                            rect = patches.Rectangle((a0[0],a0[1]),b0[0]-a0[0],b0[1]-a0[1],linewidth=.001\
                                                     ,edgecolor=colors[i],facecolor=colors[i])
                        ax.add_patch(rect)

                i = len(interval_checks) +len(subinterval_checks)
                results = interval_results[i]
                first = True
                for data in results:
                    a0,b0 = data
                    if first:
                        first = False
                        rect = patches.Rectangle((a0[0],a0[1]),b0[0]-a0[0],b0[1]-a0[1],linewidth=.001\
                                                 ,edgecolor=colors[i],facecolor=colors[i], label = 'Division Solve')
                    else:
                        rect = patches.Rectangle((a0[0],a0[1]),b0[0]-a0[0],b0[1]-a0[1],linewidth=.001\
                                                 ,edgecolor=colors[i],facecolor=colors[i])
                    ax.add_patch(rect)

                plt.title('What happened to the intervals')
                plt.xlim(a[0],b[0])
                plt.ylim(a[1],b[1])
                plt.legend()
                plt.show()

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

def chebyshev_block_copy(values_block):
    """This functions helps avoid double evaluation of functions at
    interpolation points. It takes in a tensor of function evaluation values
    and copies these values to a new tensor appropriately to prepare for
    chebyshev interpolation.

    Parameters
    ----------
    block_values : numpy array
      block of values from function evaluation

    Returns
    -------
    cheb_values : numpy array
      chebyshev interpolation values
    """
    dim = values_block.ndim
    deg = values_block.shape[0] - 1
    values_cheb = np.empty(tuple([2*deg])*dim, dtype=values_block.dtype)

    for block in product([False,True],repeat=dim):
        cheb_idx = [slice(0,deg+1)]*dim
        block_idx = [slice(None)]*dim
        for i,flip_dim in enumerate(block):
            if flip_dim:
                cheb_idx[i] = slice(deg+1,None)
                block_idx[i] = slice(deg-1,0,-1)
        values_cheb[tuple(cheb_idx)] = values_block[tuple(block_idx)]
    return values_cheb

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
    deg = degs[0]

    if hasattr(f,"evaluate_grid"):
        #for polynomials, we can quickly evaluate all points in a grid
        #xyz does not contain points, but the nth column of xyz has the values needed
        #along the nth axis. The direct product of these values procuces the grid
        cheb_values = np.cos(np.arange(deg+1)*np.pi/deg)
        xyz = transform(np.column_stack([cheb_values]*dim), a, b)
        values_block = f.evaluate_grid(xyz)

    else:
        #if function f has no "evaluate_grid" method,
        #we evaluate each point individually
        cheb_values = np.cos(np.arange(deg+1)*np.pi/deg)
        cheb_grids = np.meshgrid(*([cheb_values]*dim), indexing='ij')

        flatten = lambda x: x.flatten()
        cheb_points = transform(np.column_stack(map(flatten, cheb_grids)), a, b)
        values_block = f(cheb_points).reshape(*([deg+1]*dim))

    values = chebyshev_block_copy(values_block)
    coeffs = np.real(fftn(values/np.product(degs)))

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

def get_subintervals(a,b,dimensions,subinterval_checks,interval_results,polys,check_subintervals=False):
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

    if check_subintervals:
        scaled_subintervals = get_subintervals(-np.ones_like(a),np.ones_like(a),dimensions,None,None,None)
        for check_num, check in enumerate(subinterval_checks):
            for poly in polys:
                mask = check(poly.coeff, scaled_subintervals)
                new_scaled_subintervals = []
                new_subintervals = []
                for i, result in enumerate(mask):
                    if result:
                        new_scaled_subintervals.append(scaled_subintervals[i])
                        new_subintervals.append(subintervals[i])
                    else:
                        interval_results[check_num-(1+len(subinterval_checks))].append(subintervals[i])
                scaled_subintervals = new_scaled_subintervals
                subintervals = new_subintervals

    return subintervals

def full_cheb_approximate(f,a,b,deg,tol=1.e-8):
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
    tol : float
        How small the high degree terms must be to consider the approximation accurate.

    Returns
    -------
    coeff : numpy array
        The coefficient array of the interpolation. If it can't get a good approximation and needs to subdivide, returns None.
    """
    dim = len(a)
    degs = np.array([deg]*dim)
    coeff = interval_approximate_nd(f,a,b,degs)
    coeff2 = interval_approximate_nd(f,a,b,degs*2)
    coeff2[slice_top(coeff)] -= coeff
    clean_zeros_from_matrix(coeff2,1.e-16)
    if np.sum(np.abs(coeff2)) > tol:
        return None
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

def quad_check(test_coeff):
    """Quick check of zeros in the unit box.

    Parameters
    ----------
    test_coeff : numpy array
        The coefficient matrix of the polynomial to check

    Returns
    -------
    quad_check : bool
        False if there are no zeros in the unit box, True otherwise
    """
    dim = test_coeff.ndim
    slices = []
    slices.append(slice(0,3))
    slice_direc = 0
    for i in range(dim-1):
        slices.append(0)

    start = ext_val3(test_coeff[slices], maxx = False)
    rest = 0

    shape = list(test_coeff.shape)
    shape[slice_direc] = 1
    for spots in itertools.product(*[np.arange(i) for i in shape]):
        if sum(spots) > 0:
            for i in range(1, dim):
                slices[i] = spots[i]
            rest += ext_val3(test_coeff[slices])

    while slice_direc < dim - 1:
        slice_direc += 1
        slices[slice_direc] = slice(0,3)

        shape = np.array(test_coeff.shape)
        shape[slice_direc] = 1
        shape_diff = np.zeros_like(shape)
        for i in range(slice_direc):
            shape_diff[i] = 3
        shape -= shape_diff
        for spots in itertools.product(*[np.arange(i) for i in shape]):
            spots += shape_diff
            for i in range(dim):
                if i != slice_direc:
                    slices[i] = spots[i]
            rest += ext_val3(test_coeff[slices])

    if start > rest:
        return False
    else:
        return True

def cubic_check(test_coeff):
    """Quick check of zeros in the unit box.

    Parameters
    ----------
    test_coeff : numpy array
        The coefficient matrix of the polynomial to check

    Returns
    -------
    cubic_check : bool
        False if there are no zeros in the unit box, True otherwise
    """
    dim = test_coeff.ndim
    slices = []
    slices.append(slice(0,4))
    slice_direc = 0
    for i in range(dim-1):
        slices.append(0)

    start = ext_val4(test_coeff[slices], maxx = False)
    rest = 0

    shape = list(test_coeff.shape)
    shape[slice_direc] = 1
    for spots in itertools.product(*[np.arange(i) for i in shape]):
        if sum(spots) > 0:
            for i in range(1, dim):
                slices[i] = spots[i]
            rest += ext_val4(test_coeff[slices])

    while slice_direc < dim - 1:
        slice_direc += 1
        slices[slice_direc] = slice(0,4)

        shape = np.array(test_coeff.shape)
        shape[slice_direc] = 1
        shape_diff = np.zeros_like(shape)
        for i in range(slice_direc):
            shape_diff[i] = 4
        shape -= shape_diff
        for spots in itertools.product(*[np.arange(i) for i in shape]):
            spots += shape_diff
            for i in range(dim):
                if i != slice_direc:
                    slices[i] = spots[i]
            rest += ext_val4(test_coeff[slices])

    if start > rest:
        return False
    else:
        return True

def full_quad_check(test_coeff):
    """Quick check of zeros in the unit box.

    Parameters
    ----------
    test_coeff : numpy array
        The coefficient matrix of the polynomial to check

    Returns
    -------
    full_quad_check : bool
        False if there are no zeros in the unit box, True otherwise
    """
    for perm in itertools.permutations(np.arange(test_coeff.ndim)):
        if not quad_check(test_coeff.transpose(perm)):
            return False
    return True

def full_cubic_check(test_coeff):
    """Quick check of zeros in the unit box.

    Parameters
    ----------
    test_coeff : numpy array
        The coefficient matrix of the polynomial to check

    Returns
    -------
    full_quad_check : bool
        False if there are no zeros in the unit box, True otherwise
    """
    for perm in itertools.permutations(np.arange(test_coeff.ndim)):
        if not cubic_check(test_coeff.transpose(perm)):
            return False
    return True

def linear_check(test_coeff_in, intervals):
    """Quick check of zeros in intervals.

    Parameters
    ----------
    test_coeff_in : numpy array
        The coefficient matrix of the polynomial to check
    intervals : list
        A list of the intervals we want to check before subdividing them

    Returns
    -------
    mask : list
        Masks out the intervals we don't want
    """
    dim = test_coeff_in.ndim
    coeff_abs_sum = np.sum(np.abs(test_coeff_in))
    mask = []
    for interval in intervals:
        test_coeff = test_coeff_in.copy()

        a,b = interval
        # abs_smallest_corner = test_coeff[tuple(spot)]

        idx = [0]*dim
        const = test_coeff_in[idx]
        lin_coeff = np.zeros(dim)
        for cur_dim in range(dim):
            if test_coeff_in.shape[cur_dim] < 2:
                continue
            idx[cur_dim] = 1
            lin_coeff[cur_dim] = test_coeff_in[tuple(idx)]
            idx[cur_dim] = 0

        corner_vals = []
        for corner_pt in product(*zip(a,b)):
            corner_vals.append(const + np.sum(np.array(corner_pt)*lin_coeff))
        corner_vals = np.array(corner_vals)

        # check if corners have mixed signs
        if not (corner_vals.min() < 0 < corner_vals.max()):
            mask.append(True)
            continue

        abs_smallest_corner = np.min(np.abs(corner_vals))
        if 2*abs_smallest_corner > coeff_abs_sum:
            # case: corner is far enough from 0
            mask.append(False)
        else:
            mask.append(True)

        # test_coeff[tuple(spot)] = 0
        # for dim in range(len(a)):
        #     spot[dim] = 1
        #     neg_most_corner += a[dim]*test_coeff[tuple(spot)]
        #     spot[dim] = 0
        #
        # lin_min = neg_most_corner
        # for dim in range(len(a)):
        #     spot[dim] = 1
        #     if np.sign(test_coeff[tuple(spot)])*np.sign(neg_most_corner) < 0:
        #         lin_min += (b[dim] - a[dim]) * test_coeff[tuple(spot)]
        #     test_coeff[tuple(spot)] = 0
        #     spot[dim] = 0
        #
        # if np.sign(lin_min)*np.sign(neg_most_corner) < 0:
        #     mask.append(True)
        # elif np.sum(np.abs(test_coeff)) >= np.abs(neg_most_corner):
        #     mask.append(True)
        # else:
        #     mask.append(False)
    return mask

def quadratic_check1(test_coeff, intervals,tol=1e-12):
    """Quick check of zeros in intervals using the x^2 terms.

    Parameters
    ----------
    test_coeff : numpy array
        The coefficient matrix of the polynomial to check
    intervals : list
        A list of the intervals we want to check before subdividing them

    Returns
    -------
    mask : list
        Masks out the intervals we don't want
    """
    if test_coeff.ndim > 2:
        return [True]*len(intervals)
    padding = [(0,max(0,3-i)) for i in test_coeff.shape]
    test_coeff = np.pad(test_coeff.copy(), padding, mode='constant')
    #check using |b0 + b1x + b2y +b3T_2(x)| = |(b0 - b3) + b1x + b2y + 2 b3x^2| = |c0 + c1x + c2y + c3x^2|
    constant = test_coeff[0,0] - test_coeff[2,0]
    c1 = test_coeff[1,0]
    c2 = test_coeff[0,1]
    c3 = 2*test_coeff[2,0]

    #if c3 != 0, same as a linear check
    if np.isclose(c3, 0, atol=tol) or np.isclose(c2, 0, atol=tol):
        return [True]*len(intervals)
    mask = []
    for interval in intervals:
        def quadratic_formula_check(y):
            """given a fixed value of y, uses the quadratic formula
                to see if constant + c1x + c2y +c3T_2(x) = 0
                for some x in [a0, b0]"""
            discriminant = c1**2 - 4*(c2*y+constant)*c3
            if np.isclose(discriminant, 0,atol=tol) and interval[0][0] < -c1/2/c3 < interval[1][0]:
                 return True
            elif discriminant > 0 and \
                  (interval[0][0] < (-c1+np.sqrt(discriminant))/2/c3 < interval[1][0] or \
                   interval[0][0] < (-c1-np.sqrt(discriminant))/2/c3 < interval[1][0]):
                return True
            else:
                return False
         #If constant + c1x + c2y +c3x^2 = 0 in the region, useless check.
        if np.isclose(c2, 0,atol=tol) and quadratic_formula_check(0):
            mask.append(True)
            continue
        else:
            y = lambda x: (-c3 *x**2 - c1 * x - constant)/c2
            if interval[0][1] < y(interval[0][0]) < interval[1][1] or interval[0][1] < y(interval[1][0]) < interval[1][1]:
                mask.append(True)
                continue
            elif quadratic_formula_check(interval[0][0]) or quadratic_formula_check(interval[1][0]):
                mask.append(True)
                continue

         #function for evaluating |constant + c1x + c2y +c3x^2|
        eval = lambda xy: abs(constant + c1*xy[:,0] + c2*xy[:,1] + c3 * xy[:,0]**2)
         #In this case, extrema only occur on the edges since there are no critical points
        #edges 1&2: x = a0, b0 --> potential extrema at corners
        #edges 3&4: y = a1, b1 --> potential extrema at x0 = -c1/2c3, if that's in [a0, b0]
        if interval[0][0] < -c1/2/c3 < interval[1][0]:
            potential_minimizers = np.array([[interval[0][0],interval[0][1]],
                                             [interval[0][0],interval[1][1]],
                                             [interval[1][0],interval[0][1]],
                                             [interval[1][0],interval[1][1]],
                                             [-c1/2/c3,interval[0][1]],
                                             [-c1/2/c3,interval[1][1]]])
        else:
            potential_minimizers = np.array([[interval[0][0],interval[0][1]],
                                             [interval[0][0],interval[1][1]],
                                             [interval[1][0],interval[0][1]],
                                             [interval[1][0],interval[1][1]]])
         #if min{|constant + c1x + c2y +c3x^2|} > sum of other terms in test_coeff, no roots in the region
        if min(eval(potential_minimizers)) > np.sum(np.abs(test_coeff)) - abs(constant) - abs(c1) - abs(c2) - abs(c3):
            mask.append(False)
        else:
            mask.append(True)
    return mask

def quadratic_check2(test_coeff, intervals,tol=1e-12):
    """Quick check of zeros in the unit box using the y^2 terms

     Parameters
     ----------
     test_coeff : numpy array
         The coefficient matrix of the polynomial to check
     intervals : list
         A list of the intervals we want to check before subdividing them

     Returns
     -------
     mask : list
         Masks out the intervals we don't want
    """
    if test_coeff.ndim > 2:
        return [True]*len(intervals)
    padding = [(0,max(0,3-i)) for i in test_coeff.shape]
    test_coeff = np.pad(test_coeff.copy(), padding, mode='constant')
    #very similar to quadratic_check_1, but switch x and y
    #check using |b0 + b1x + b2y +b3T_2(y)| = |b0 - b3 + b1x + b2y + 2 b3y^2| = |c0 + c1x + c2y + c3y^2|
    constant = test_coeff[0,0] - test_coeff[0,2]
    c1 = test_coeff[1,0]
    c2 = test_coeff[0,1]
    c3 = 2*test_coeff[0,2]

    #if c3 != 0, same as a linear check
    if np.isclose(c3, 0, atol=tol) or np.isclose(c1, 0, atol=tol):
        return[True]*len(intervals)
    mask = []
    for interval in intervals:
        def quadratic_formula_check(x):
            """given a fixed value of x, uses the quadratic formula
                to see if constant + c1x + c2y +c3y^2 = 0
                for some y in [a1, b1]"""
            discriminant = c2**2 - 4*(c1*x+constant)*c3
            if np.isclose(discriminant, 0,atol=tol) and interval[0][1] < -c2/2/c3 < interval[1][1]:
                 return True
            elif discriminant > 0 and \
                  (interval[0][1] < (-c2+np.sqrt(discriminant))/2/c3 < interval[1][1] or \
                   interval[0][1] < (-c2-np.sqrt(discriminant))/2/c3 < interval[1][1]):
                return True
            else:
                return False
         #If constant + c1x + c2y +c3y^2 = 0 in the region, useless
        if np.isclose(c1, 0) and quadratic_formula_check(0):
            mask.append(True)
            continue
        else:
            x = lambda y: (-c3 *y**2 - c2 * y - constant)/c1
            if interval[0][0] < x(interval[0][1]) < interval[1][0] or interval[0][0] < x(interval[1][1]) < interval[1][0]:
                mask.append(True)
                continue
            elif quadratic_formula_check(interval[0][1]) or quadratic_formula_check(interval[1][1]):
                mask.append(True)
                continue

        #function to evaluate |constant + c1x + c2y +c3y^2|
        eval = lambda xy: abs(constant + c1*xy[:,0] + c2*xy[:,1] + c3 * xy[:,1]**2)
        #In this case, extrema only occur on the edges since there are no critical points
        #edges 1&2: x = a0, b0 --> potential extrema at y0 = -c2/2c3, if that's in [a1, b1]
        #edges 3&4: y = a1, b1 --> potential extrema at corners
        if interval[0][1] < -c2/2/c3 < interval[1][1]:
            potential_minimizers = np.array([[interval[0][0],interval[0][1]],
                                             [interval[0][0],interval[1][1]],
                                             [interval[1][0],interval[0][1]],
                                             [interval[1][0],interval[1][1]],
                                             [interval[0][0],-c2/2/c3],
                                             [interval[1][0],-c2/2/c3]])
        else:
            potential_minimizers = np.array([[interval[0][0],interval[0][1]],
                                             [interval[0][0],interval[1][1]],
                                             [interval[1][0],interval[0][1]],
                                             [interval[1][0],interval[1][1]]])
         #if min{|constant + c1x + c2y +c3y^2|} > sum of other terms in test_coeff, no roots in the region
        if min(eval(potential_minimizers)) > np.sum(np.abs(test_coeff)) - abs(constant) - abs(c1) - abs(c2) - abs(c3):
            mask.append(False)
        else:
            mask.append(True)
    return mask

def quadratic_check3(test_coeff, intervals,tol=1e-12):
    """Quick check of zeros in the unit box using the xy terms

     Parameters
     ----------
     test_coeff : numpy array
         The coefficient matrix of the polynomial to check
     intervals : list
         A list of the intervals we want to check before subdividing them

     Returns
     -------
     mask : list
         Masks out the intervals we don't want
    """
    if test_coeff.ndim > 2:
        return [True]*len(intervals)
    padding = [(0,max(0,3-i)) for i in test_coeff.shape]
    test_coeff = np.pad(test_coeff.copy(), padding, mode='constant')
    #check using |constant + c1x + c2y +c3xy|
    constant = test_coeff[0,0]
    c1 = test_coeff[1,0]
    c2 = test_coeff[0,1]
    c3 = test_coeff[1,1]

    ##if c3 != 0, same as a linear check
    if np.isclose(c3, 0,atol=tol):
        return [True]*len(intervals)

    mask = []
    for interval in intervals:
        ##If constant + c1x + c2y +c3xy = 0 in the region, useless

        #testing the vertical sides of the interval
        vert_asymptote = -c2/c3
        x = lambda y: (-constant + c2*y)/(c1 + c3*y)
        if np.isclose(interval[0][1], vert_asymptote):
            if interval[0][0] < x(interval[1][1]) < interval[1][0]:
                mask.append(True)
                continue
        elif np.isclose(interval[1][1], vert_asymptote):
            if interval[0][0] < x(interval[0][1]) < interval[1][0]:
                mask.append(True)
                continue
        elif interval[0][0] < x(interval[0][1]) < interval[1][0] or interval[0][0] < x(interval[1][1]) < interval[1][0]:
            mask.append(True)
            continue

        #testing the horizontal sides of the interval
        horiz_asymptote = -c1/c3
        y = lambda x: (-constant + c1*x)/(c2 + c3*x)
        if np.isclose(interval[0][0], horiz_asymptote):
            if interval[0][1] < y(interval[1][0]) < interval[1][1]:
                mask.append(True)
                continue
        elif np.isclose(interval[1][0], horiz_asymptote):
            if interval[0][1] < y(interval[0][0]) < interval[1][1]:
                mask.append(True)
                continue
        elif interval[0][1] < y(interval[0][0]) < interval[1][1] or interval[0][1] < y(interval[1][0]) < interval[1][1]:
            mask.append(True)
            continue

        ##Find the minimum

        #function for evaluating |constant + c1x + c2y +c3xy|
        eval = lambda xy: abs(constant + c1*xy[:,0] + c2*xy[:,1] + c3*xy[:,0]*xy[:,1])

        #In this case, only critical point is saddle point, so all minima occur on the edges
        #On all the edges it becomes linear, so extrema always ocur at the corners
        potential_minimizers = np.array([[interval[0][0],interval[0][1]],
                                         [interval[0][0],interval[1][1]],
                                         [interval[1][0],interval[0][1]],
                                         [interval[1][0],interval[1][1]]])

        ##if min{|constant + c1x + c2y +c3xy|} > sum of other terms in test_coeff, no roots in the region
        if min(eval(potential_minimizers)) > np.sum(np.abs(test_coeff)) - np.sum(np.abs(test_coeff[:2,:2])):
            mask.append(False)
        else:
            mask.append(True)

    return mask

#This is all for Tyler's new function
from mpmath import iv
from itertools import product
from copy import copy
def lambda_s(a):
    return sum(iv.mpf([0,1])*max(ai.a**2,ai.b**2) for ai in a)

def beta(a,b):
    return iv.mpf([-1,1])*iv.sqrt(lambda_s(a)*lambda_s(b))

def lambda_t(a,b):
    return beta(a,b) + np.dot(a,b)

class TabularCompute:
    def __init__(self,a,b,dim=False,index=None):
        """Class for estimating the maximum curvature.
        Parameters
        ----------
            a (int) - the starting value of the interval
            b (int) - the ending value of the interval
            dim (bool or int) - False if this is not an interval for a dimension
                                integer indicating the number of dimensions
            index (int) - defines which dimension this interval corresponds to

        """
        self.iv = iv.mpf([a,b])
        self.iv_lambda = iv.mpf([0,0])
        if dim:
            assert isinstance(dim, int)
            assert isinstance(index, int) and 0<=index<dim
            self.iv_prime = np.array([iv.mpf([0,0]) for _ in range(dim)])
            self.iv_prime[index] = iv.mpf([1,1])
        else:
            self.iv_prime = iv.mpf([0,0])

    def copy(self):
        new_copy = TabularCompute(0,0)
        new_copy.iv = copy(self.iv)
        new_copy.iv_prime = copy(self.iv_prime)
        new_copy.iv_lambda = copy(self.iv_lambda)
        return new_copy

    def __add__(self, other):
        new = self.copy()
        if isinstance(other, TabularCompute):
            new.iv += other.iv
            new.iv_prime += other.iv_prime
            new.iv_lambda += other.iv_lambda
        else:
            new.iv += other
        return new

    def __mul__(self, other):
        new = TabularCompute(0,0)
        if isinstance(other, TabularCompute):
            new.iv = self.iv*other.iv
            tmp1 = np.array([self.iv])*other.iv_prime
            tmp2 = np.array([other.iv])*self.iv_prime
            new.iv_prime = tmp1 + tmp2
            new.iv_lambda = (self.iv*other.iv_lambda
                            + other.iv*self.iv_lambda
                            + lambda_t(self.iv_prime, other.iv_prime))
        else:
            new.iv = self.iv*other
            new.iv_prime = self.iv_prime*other
            new.iv_lambda = self.iv_lambda*other
        return new
    def __sub__(self, other):
        return self + (-1*other)
    def __rmul__(self, other):
        return self*other
    def __radd__(self, other):
        return self + other
    def __rsub__(self, other):
        return (-1*self) + other
    def __str__(self):
        return f"{self.iv}\n{self.iv_prime}\n{self.iv_lambda}"
    def __repr__(self):
        return str(self)

chebval = np.polynomial.chebyshev.chebval
def chebvalnd(intervals, poly):
    n = poly.dim
    c = poly.coeff
    c = chebval(intervals[0],c, tensor=True)
    for i in range(1,n):
        c = chebval(intervals[i],c, tensor=False)
    if len(poly.coeff) == 1:
        return c[0]
    else:
        return c

def can_eliminate(poly, a, b):
    assert len(a)==len(b)==poly.dim
    n = poly.dim
    h = (b-a)[0]
    assert np.allclose(b-a, h)

    corners = poly(list(product(*zip(a,b))))
    if not (all(corners>0) or all(corners<0)):
        return False

    min_corner = abs(min(corners))

    x = []
    n = len(a)
    for i,(ai,bi) in enumerate(zip(a,b)):
        x.append(TabularCompute(ai,bi,dim=n,index=i))
    x = np.array(x)

    max_curve = abs(chebvalnd(x, poly).iv_lambda)
#     print(max_curve * n * h**2/8)
    return min_corner > max_curve * n * h**2/8

def curvature_check(coeff):
    poly = MultiCheb(coeff)
    a = np.array([-1.]*poly.dim)
    b = np.array([1.]*poly.dim)
    return not can_eliminate(poly, a, b)

def subdivision_solve_nd(funcs,a,b,deg,interval_results,interval_checks = [],subinterval_checks=[],tol=1.e-3):
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
    division_var = 0
    cheb_approx_list = []
    try:
        if np.random.rand() > .999:
            print("Interval - ",a,b)
        dim = len(a)
        for func in funcs:
            coeff = full_cheb_approximate(func,a,b,deg,tol=tol)

            #Subdivides if needed.
            if coeff is None:
                intervals = get_subintervals(a,b,np.arange(dim),None,None,None)

                return np.vstack([subdivision_solve_nd(funcs,interval[0],interval[1],deg,interval_results\
                                                       ,interval_checks,subinterval_checks,tol=tol)
                                  for interval in intervals])
            else:
                coeff = trim_coeff(coeff,tol=tol)
                #Run checks to try and throw out the interval
                for func_num, func in enumerate(interval_checks):
                    if not func(coeff):
                        interval_results[func_num].append([a,b])
                        return np.zeros([0,dim])
                cheb_approx_list.append(MultiCheb(coeff))

        zeros = np.array(division(cheb_approx_list, get_divvar_coord_from_eigval = True, divisor_var = 0, tol = 1.e-6))
        interval_results[-1].append([a,b])
        if len(zeros) == 0:
            return np.zeros([0,dim])
        return transform(good_zeros_nd(zeros),a,b)

    except np.linalg.LinAlgError as e:
        while division_var < len(a):
            try:
                zeros = np.array(division(cheb_approx_list, get_divvar_coord_from_eigval = True, divisor_var = 0, tol = 1.e-6))
                return zeros
            except np.linalg.LinAlgError as e:
                division_var += 1

        #Subdivide but run some checks on the intervals first
        intervals = get_subintervals(a,b,np.arange(dim),subinterval_checks,interval_results\
                                     ,cheb_approx_list,check_subintervals=True)
        if len(intervals) == 0:
            return np.zeros([0,dim])
        else:
            return np.vstack([subdivision_solve_nd(funcs,interval[0],interval[1],deg,interval_results\
                                               ,interval_checks,subinterval_checks,tol=tol)
                          for interval in intervals])

def trim_coeff(coeff, tol=1.e-3):
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
    dim = coeff.ndim

    #Cuts out the high diagonals as much as possible to minimize polynomial degree.
    if abs(coeff[tuple([-1]*dim)]) < tol:
        coeff[tuple([-1]*dim)] = 0
        deg = np.sum(coeff.shape)-dim-1
        while deg > 2:
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

def subdivision_solve_1d(f,a,b,cheb_approx_tol=1.e-3,max_degree=128):
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
    cur_deg = 2
    initial_approx = interval_approximate_1d(f,a,b,deg = cur_deg)
    while cur_deg<=max_degree:
        coeffsN = np.zeros(2*cur_deg+1)
        coeffsN[:cur_deg+1] = initial_approx
        coeffs2N = interval_approximate_1d(f,a,b,deg = 2*cur_deg)
        #Check if the approximation is good enough
        # if np.sum(np.abs(coeffs2N - coeffsN)) < cheb_approx_tol:
        if np.sum(np.abs(coeffs2N[cur_deg+1:])) < cheb_approx_tol:
            coeffs = coeffsN[:cur_deg+1]
            #Division is faster after degree 75
            if cur_deg > 75:
                return transform(good_zeros(divCheb(coeffs)),a,b)
            else:
                return transform(good_zeros(multCheb(np.trim_zeros(coeffs.copy(),trim='b'))),a,b)
        initial_approx = coeffs2N
        cur_deg*=2
    #Subdivide the interval and recursively call the function.
    div_length = (b-a)/2
    return np.hstack([subdivision_solve_1d(f,a,b-div_length,max_degree=max_degree),\
                      subdivision_solve_1d(f,a+div_length,b,max_degree=max_degree)])
