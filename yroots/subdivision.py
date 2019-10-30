"""
Subdivision provides a solve function that finds roots of a set of functions
by approximating the functions with Chebyshev polynomials.
When the approximation is performed on a sufficiently small interval,
the approximation degree is small enough to be solved efficiently.

"""

import numpy as np
from numpy.fft.fftpack import fftn
from yroots.OneDimension import divCheb,divPower,multCheb,multPower,solve
from yroots.Division import division
from yroots.Multiplication import multiplication
from yroots.utils import clean_zeros_from_matrix, slice_top, MacaulayError, get_var_list, \
                        ConditioningError, Tolerances
from yroots.polynomial import MultiCheb
from yroots.IntervalChecks import IntervalData
from yroots.RootTracker import RootTracker
from itertools import product
from matplotlib import pyplot as plt
from scipy.linalg import lu
import itertools
import time
import warnings

def solve(funcs, a, b, rel_approx_tol=1.e-6, abs_approx_tol=1.e-10, max_cond_num=1e6, macaulay_zero_tol=1e-12, good_zeros_factor=100, min_good_zeros_tol=1e-5, plot = False, plot_intervals = False, deg = None, max_level=999):
    '''
    Finds the real roots of the given list of functions on a given interval.

    All of the tolerances can be passed in as numbers of iterable types. If multiple
    are passed in as iterable types they must have the same length. When the length is
    more than 1, they are used one after the other to polish the roots.

    Parameters
    ----------
    funcs : list of vectorized, callable functions
        Functions to find the common roots of.
        More efficient if functions have an 'evaluate_grid' method handle
        function evaluation at an grid of points.
    a : numpy array
        The lower bound on the interval.
    b : numpy array
        The upper bound on the interval.
    rel_approx_tol : float or list
        The relative tolerance used in the approximation tolerance. The error is bouned by
        error < abs_approx_tol + rel_approx_tol * inf_norm_of_approximation
    abs_approx_tol : float or list
        The absolute tolerance used in the approximation tolerance. The error is bouned by
        error < abs_approx_tol + rel_approx_tol * inf_norm_of_approximation
    max_cond_num : float or list
        The maximum condition number of the Macaulay Matrix Reduction
    macaulay_zero_tol : float or list
        What is considered 0 in the macaulay matrix reduction.
    good_zeros_factor : float or list
        Multiplying this by the approximation error gives how far outside of [-1,1] a root can
        be and still be considered inside the interval.
    min_good_zeros_tol : float or list
        The smallest the good_zeros_tol can be, which is how far outside of [-1,1] a root can
        be and still be considered inside the interval.
    plot : bool
        If True plots the zeros-loci of the functions along with the computed roots
    plot_intervals : bool
        If True, plot is True, and the functions are 2 dimensional, plots what check/method solved
        each part of the interval.
    deg : int
        The degree used for the approximation. If None, the following degrees are used.
        Degree 50 for 1D functions.
        Degree 9 for 2D functions.
        Degree 5 for 3D functions.
        Degree 3 for 4D functions.
        Degree 2 for 5D functions and above.
    max_level : int
        The maximum levels deep the recursion will go. Increasing it above 999 may result in recursion error!

    If finding roots of a univariate function, `funcs` does not need to be a list,
    and `a` and `b` can be floats instead of arrays.

    returns
    -------
    zeros : numpy array
        The common zeros of the polynomials. Each row is a root.
    '''
    #Detect the dimension
    if not isinstance(funcs,list):
        dim = 1
    else:
        dim = len(funcs)

    #make a and b the right type
    a = np.float64(a)
    b = np.float64(b)

    #choose an appropriate max degree for the given dimension if non is specified.
    if deg is None:
        deg_dim = {1: 50, 2:9, 3:5, 4:3}
        if dim > 4:
            deg = 2
        else:
            deg = deg_dim[dim]

    #Sets up the tolerances.
    tols = Tolerances(rel_approx_tol=rel_approx_tol, abs_approx_tol=abs_approx_tol, max_cond_num=max_cond_num, macaulay_zero_tol=macaulay_zero_tol, good_zeros_factor=good_zeros_factor, min_good_zeros_tol=min_good_zeros_tol)
    tols.nextTols()

    #Set up the interval data and root tracker classes
    interval_data = IntervalData(a,b)
    root_tracker = RootTracker()

    if dim == 1:
        solve_func = subdivision_solve_1d
        if isinstance(funcs,list):
            funcs = funcs[0]
    else:
        solve_func = subdivision_solve_nd

    #Initial Solve
    solve_func(funcs,a,b,deg,interval_data,root_tracker,tols,max_level)

    #Polishing
    while tols.nextTols():
        polish_intervals = root_tracker.get_polish_intervals()
        interval_data.add_polish_intervals(polish_intervals)
        for new_a, new_b in polish_intervals:
            interval_data.start_polish_interval()
            solve_func(funcs,new_a,new_b,deg,interval_data,root_tracker,tols,max_level)
    print("\rPercent Finished: 100%{}".format(' '*50))

    #Print results
    interval_data.print_results()

    #Plotting
    if plot:
        if dim == 1:
            x = np.linspace(a,b,1000)
            plt.plot(x,funcs(x),color='k')
            plt.plot(np.real(root_tracker.roots),np.zeros(len(root_tracker.roots)),'o',color = 'none',markeredgecolor='r')
            plt.show()
        elif dim == 2:
            interval_data.plot_results(funcs, root_tracker.roots, plot_intervals)

    return root_tracker.roots

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

def chebyshev_block_copy(values_block):
    """This functions helps avoid double evaluation of functions at
    interpolation points. It takes in a tensor of function evaluation values
    and copies these values to a new tensor appropriately to prepare for
    chebyshev interpolation.

    Parameters
    ----------
    values_block : numpy array
      block of values from function evaluation

    Returns
    -------
    values_cheb : numpy array
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

def interval_approximate_1d(f,a,b,deg,inf_norm=None):
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
    inf_norm : float
        The inf_norm of the function, if already computed
    Returns
    -------
    coeffs : numpy array
        The coefficient of the chebyshev interpolating polynomial.
    inf_norm : float
        The inf_norm of the function
    """
    extrema = transform(np.cos((np.pi*np.arange(2*deg))/deg),a,b)
    values = f(extrema)

    if inf_norm is not None:
        inf_norm = max(np.max(np.abs(values)), inf_norm)
    else:
        inf_norm = np.max(np.abs(values))

    coeffs = np.real(np.fft.fft(values/deg))
    coeffs[0]/=2
    coeffs[deg]/=2
    return coeffs[:deg+1], inf_norm

class Memoize:
    """
    A Memoization class taken from Stack Overflow
    https://stackoverflow.com/questions/1988804/what-is-memoization-and-how-can-i-use-it-in-python
    """
    def __init__(self, f):
        self.f = f
        self.memo = {}
    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]

@Memoize
def get_cheb_grid(deg, dim, has_eval_grid):
    """Helper function for interval_approximate_nd.

    Parameters
    ----------
    deg : int
        The interpolation degree.
    dim : int
        The interpolation dimension.

    Returns
    -------
    get_cheb_grid : numpy array
        The chebyshev grid used to evaluate the functions in interval_approximate_nd
    """
    if has_eval_grid:
        cheb_values = np.cos(np.arange(deg+1)*np.pi/deg)
        return np.column_stack([cheb_values]*dim)
    else:
        cheb_values = np.cos(np.arange(deg+1)*np.pi/deg)
        cheb_grids = np.meshgrid(*([cheb_values]*dim), indexing='ij')
        flatten = lambda x: x.flatten()
        return np.column_stack(tuple(map(flatten, cheb_grids)))

def interval_approximate_nd(f,a,b,deg,return_bools=False,inf_norm=None):
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
    return_bools: bool
        whether to return bools which indicate if the function changes sign or not
    inf_norm : float
        The inf_norm of the function, if already computed

    Returns
    -------
    coeffs : numpy array
        The coefficient of the chebyshev interpolating polynomial.
    change_sign: numpy array (Optional)
        list of which subintervals change sign
    inf_norm : float
        The inf_norm of the function
    """
    if len(a)!=len(b):
        raise ValueError("Interval dimensions must be the same!")

    dim = len(a)

    if hasattr(f,"evaluate_grid"):
        xyz = transform(get_cheb_grid(deg, dim, True), a, b)
        values_block = f.evaluate_grid(xyz)
    else:
        cheb_points = transform(get_cheb_grid(deg, dim, False), a, b)
        cheb_points = [cheb_points[:,i] for i in range(dim)]
        values_block = f(*cheb_points).reshape(*([deg+1]*dim))

    #figure out on which subintervals the function changes sign
    if return_bools:
        change_sign = np.zeros(2**dim, dtype=bool)

    values = chebyshev_block_copy(values_block)

    if inf_norm is not None:
        inf_norm = max(np.max(np.abs(values_block)), inf_norm)
    else:
        inf_norm = np.max(np.abs(values_block))

    coeffs = np.real(fftn(values/deg**dim))

    for i in range(dim):
        #construct slices for the first and degs[i] entry in each dimension
        idx0 = [slice(None)] * dim
        idx0[i] = 0

        idx_deg = [slice(None)] * dim
        idx_deg[i] = deg

        #halve the coefficients in each slice
        coeffs[tuple(idx0)] /= 2
        coeffs[tuple(idx_deg)] /= 2

    slices = [slice(0,deg+1)]*dim
    if return_bools:
        return coeffs[tuple(slices)], change_sign, inf_norm
    else:
        return coeffs[tuple(slices)], inf_norm

def get_subintervals(a,b,dimensions,interval_data,polys,change_sign,approx_error,check_subintervals=False):
    """Gets the subintervals to divide a search interval into.

    Parameters
    ----------
    a : numpy array
        The lower bound on the interval.
    b : numpy array
        The upper bound on the interval.
    dimensions : numpy array
        The dimensions we want to cut in half.
    interval_data : IntervalData
        A class to run the subinterval checks and keep track of the solve progress
    polys : list
        A list of MultiCheb polynomials representing the function approximations on the
        interval to subdivide. Used in the subinterval checks.
    change_sign : list
        A list of bools of whether we know the functions can change sign on the subintervals.
        Used in the subinterval checks.
    approx_error: float
        The bound of the sup norm error of the chebyshev approximation.
    check_subintervals : bool
        If True runs the subinterval checks to through out intervals where the functions are never 0.

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
        #get intervals -1 to 1
        scaled_subintervals = get_subintervals(-np.ones_like(a),np.ones_like(a),dimensions,None,None,None,approx_error)
        return interval_data.check_subintervals(subintervals, scaled_subintervals, polys, change_sign, approx_error)
    else:
        return subintervals

def full_cheb_approximate(f,a,b,deg,abs_approx_tol,rel_approx_tol,good_deg=None):
    """Gives the full chebyshev approximation and checks if it's good enough.

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
    rel_approx_tol : float or list
        The relative tolerance used in the approximation tolerance. The error is bouned by
        error < abs_approx_tol + rel_approx_tol * inf_norm_of_approximation
    abs_approx_tol : float or list
        The absolute tolerance used in the approximation tolerance. The error is bouned by
        error < abs_approx_tol + rel_approx_tol * inf_norm_of_approximation
    good_deg : numpy array
        Interpoation degree that is guaranteed to give an approximation valid to within approx_tol.

    Returns
    -------
    coeff : numpy array
        The coefficient array of the interpolation. If it can't get a good approximation and needs to subdivide, returns None.
    bools: numpy array
        (2^n, 1) array of bools corresponding to which subintervals the function changes sign in
        If it cannot get a good approximation, it returns which directions to subdivide in.
    inf_norm : float
        The inf norm of f on [a,b]
    error : float
        The approximation error
    """
    #We know what degree we want
#     if good_deg is not None:
#         coeff, bools, inf_norm = interval_approximate_nd(f,a,b,good_deg,return_bools=True)
#         return coeff, bools, inf_norm
    #Try degree deg and see if it's good enough
    coeff, inf_norm = interval_approximate_nd(f,a,b,deg)
    coeff2, bools, inf_norm = interval_approximate_nd(f,a,b,deg*2,return_bools=True, inf_norm=inf_norm)
    coeff2[slice_top(coeff)] -= coeff

    error = np.sum(np.abs(coeff2))
    if error > abs_approx_tol+rel_approx_tol*inf_norm:
        #Find the directions to subdivide
        dim = len(a)
        # TODO: Intelligent Subdivision.
        # div_dimensions = []
        # slices = [slice(0,None,None)]*dim
        # for d in range(dim):
        #     slices[d] = slice(deg+1,None,None)
        #     if np.sum(np.abs(coeff2[tuple(slices)])) > approx_tol/dim:
        #         div_dimensions.append(d)
        #     slices[d] = slice(0,None,None)
        # if len(div_dimensions) == 0:
        #     div_dimensions.append(0)
        # return None, np.array(div_dimensions)

        #for now, subdivide in every dimension
        return None, np.arange(dim), inf_norm, error
    else:
        return coeff, bools, inf_norm, error

def good_zeros_nd(zeros, imag_tol, real_tol):
    """Get the real zeros in the -1 to 1 interval in each dimension.

    Parameters
    ----------
    zeros : numpy array
        The zeros to be checked.
    imag_tol : float
        How large the imaginary part can be to still have it be considered real.
    real_tol : float
        How far the real part can be outside the interval [-1,1]^n and still be
        considered valid.

    Returns
    -------
    good_zeros : numpy array
        The real zeros in [-1,1]^n of the input zeros.
    """
    good_zeros = zeros[np.all(np.abs(zeros.imag) <= imag_tol,axis = 1)]
    good_zeros = good_zeros[np.all(np.abs(good_zeros) <= 1 + real_tol,axis = 1)]
    return good_zeros.real

def solve_linear(coeffs):
    """Finds the roots when the coeffs are **all** linear.

    Parameters
    ----------
    coeffs : list
        A list of the coefficient arrays. They should all be linear.

    Returns
    -------
    solve_linear : numpy array
        The root, if any.
    """
    dim = len(coeffs[0].shape)
    A = np.zeros([dim,dim])
    B = np.zeros(dim)
    for row in range(dim):
        coeff = coeffs[row]
        spot = tuple([0]*dim)
        B[row] = coeff[spot]
        var_list = get_var_list(dim)
        for col in range(dim):
            if coeff.shape[0] == 1:
                A[row,col] = 0
            else:
                A[row,col] = coeff[var_list[col]]

    #solve the system
    try:
        return np.linalg.solve(A,-B)
    except np.linalg.LinAlgError as e:
        if str(e) == 'Singular matrix':
            #if the system is dependent, then there are infinitely many roots
            #if the system is inconsistent, there are no roots
            #TODO: this should be more airtight than raising a warning

            #if the rightmost column of U from LU decomposition
            # is a pivot column, system is inconsistent
            # otherwise, it's dependent
            U = lu(np.hstack((A,B.reshape(-1,1))))[2]
            pivot_columns = [np.flatnonzero(U[i, :])[0] for i in range(U.shape[0]) if np.flatnonzero(U[i, :]).shape[0]>0]
            if not (U.shape[1]-1 in pivot_columns):
                #independent
                warnings.warn('System potentially has infinitely many roots')
            return np.zeros([0,dim])

def subdivision_solve_nd(funcs,a,b,deg,interval_data,root_tracker,tols,max_level,good_degs=None,level=0):
    """Finds the common zeros of the given functions.

    All the zeros will be stored in root_tracker.

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
    interval_data : IntervalData
        A class to run the subinterval checks and keep track of the solve progress
    root_tracker : RootTracker
        A class to keep track of the roots that are found.
    tols : Tolerances
        The tolerances to be used.
    max_level : int
        The maximum level for the recursion
    good_degs : numpy array
        Interpoation degrees that are guaranteed to give an approximation valid to within approx_tol.
    level : int
        The current level of the recursion.
    """
    if level > max_level:
        # TODO Refine case where there may be a root and it goes too deep.
        interval_data.track_interval("Too Deep", [a, b])
        # # Find residuals of the midpoint of the interval.
        # residual_samples = list()
        # for func in funcs:
        #     residual_samples.append(func(*(np.array(a) + np.array(b))/2))
        # # If all the residuals are within the tolerance, return midpoint approximation.
        # if np.all(residual < solve_tol for residual in residual_samples):
        #     return (np.array(a) + np.array(b))/2
        return

    cheb_approx_list = []
    interval_data.print_progress()
    dim = len(a)
    if good_degs is None:
        good_degs = [None]*len(funcs)
    inf_norms = []
    approx_errors = []
    #Get the chebyshev approximations
    for func, good_deg in zip(funcs, good_degs):
        coeff,change_sign,inf_norm,approx_error = full_cheb_approximate(func,a,b,deg,tols.abs_approx_tol,tols.rel_approx_tol, good_deg)
        inf_norms.append(inf_norm)
        approx_errors.append(approx_error)
        #Subdivides if a bad approximation
        if coeff is None:
            intervals = get_subintervals(a,b,change_sign,None,None,None,approx_errors)
            for new_a, new_b in intervals:
                subdivision_solve_nd(funcs,new_a,new_b,deg,interval_data,root_tracker,tols,max_level,level=level+1)
            return
        else:
            #if the function changes sign on at least one subinterval, skip the checks
            if np.any(change_sign):
                cheb_approx_list.append(coeff)
                continue
            #Run checks to try and throw out the interval
            if interval_data.check_interval(coeff, approx_error, a, b):
                return

            cheb_approx_list.append(coeff)

    #Make the system stable to solve
    coeffs, good_approx, approx_errors = trim_coeffs(cheb_approx_list, tols.abs_approx_tol, tols.rel_approx_tol, inf_norms, approx_errors)

    #Used if subdividing further.
    good_degs = [coeff.shape[0] - 1 for coeff in coeffs]
    good_zeros_tol = max(tols.min_good_zeros_tol, np.sum(np.abs(approx_errors))*tols.good_zeros_factor)

    #Check if everything is linear
    if np.all(np.array([coeff.shape[0] for coeff in coeffs]) == 2):
        zero = solve_linear(coeffs)
        #Store the information and exit
        zero = transform(good_zeros_nd(zero.reshape([1,dim]),good_zeros_tol,good_zeros_tol),a,b)
        interval_data.track_interval("Base Case", [a,b])
        root_tracker.add_roots(zero, a, b, "Base Case")

    #Check if anything is linear
    elif np.any(np.array([coeff.shape[0] for coeff in coeffs]) == 2):
        #Subdivide but run some checks on the intervals first
        intervals = get_subintervals(a,b,np.arange(dim),interval_data,cheb_approx_list,change_sign,approx_errors,True)
        for new_a, new_b in intervals:
            subdivision_solve_nd(funcs,new_a,new_b,deg,interval_data,root_tracker,tols,max_level,good_degs,level+1)

    #Runs the same things as above, but we want to get rid of that eventually so keep them seperate.
    elif np.any(np.array([coeff.shape[0] for coeff in coeffs]) > 5) or not good_approx:
        intervals = get_subintervals(a,b,np.arange(dim),interval_data,cheb_approx_list,change_sign,approx_errors,True)
        for new_a, new_b in intervals:
            subdivision_solve_nd(funcs,new_a,new_b,deg,interval_data,root_tracker,tols,max_level,good_degs,level+1)

    #Solve using spectral methods if stable.
    else:
        polys = [MultiCheb(coeff, lead_term = [coeff.shape[0]-1], clean_zeros = False) for coeff in coeffs]
        try:
            zeros = multiplication(polys, max_cond_num=tols.max_cond_num, macaulay_zero_tol= tols.macaulay_zero_tol)
            zeros = transform(good_zeros_nd(zeros,good_zeros_tol,good_zeros_tol),a,b)
            interval_data.track_interval("Spectral", [a,b])
            root_tracker.add_roots(zeros, a, b, "Spectral")
        except ConditioningError as e:
            #Subdivide but run some checks on the intervals first
            intervals = get_subintervals(a,b,np.arange(dim),interval_data,cheb_approx_list,change_sign,approx_errors,True)
            for new_a, new_b in intervals:
                subdivision_solve_nd(funcs,new_a,new_b,deg,interval_data,root_tracker,tols,max_level,good_degs,level+1)

def trim_coeffs(coeffs, abs_approx_tol, rel_approx_tol, inf_norms, errors):
    """Trim the coefficient matrices so they are stable and choose a direction to divide in.

    Parameters
    ----------
    coeffs : list
        The coefficient matrices of the Chebyshev polynomials we are solving.
    rel_approx_tol : float or list
        The relative tolerance used in the approximation tolerance. The error is bouned by
        error < abs_approx_tol + rel_approx_tol * inf_norm_of_approximation
    abs_approx_tol : float or list
        The absolute tolerance used in the approximation tolerance. The error is bouned by
        error < abs_approx_tol + rel_approx_tol * inf_norm_of_approximation
    inf_norms : list
        The inf norms of the functions
    errors : list
        The approximation errors of the functions
    Returns
    -------
    polys : list
        The reduced degree Chebyshev polynomials
    good_approx : bool
        Whether all the approximations were good
    """
    #we start with good approximations
    good_approx = True
    for num, coeff in enumerate(coeffs):
        #get the error inherent in the approximation
        error = errors[num]
        #zero out small spots in the coefficient matrix; increment the error accordingly
        spot = np.abs(coeff) < 1.e-10*np.max(np.abs(coeff))
        error += np.sum(np.abs(coeff[spot]))
        coeff[spot] = 0

        #try to zero out everything below the lower-reverse-hyperdiagonal
        #that's a fancy way of saying monomials that are more than the specified degree
        dim = coeff.ndim
        deg = np.sum(coeff.shape) - dim
        initial_mons = []
        for deg0 in range(coeff.shape[0], deg+1):
            initial_mons += mon_combos_limited_wrap(deg0, dim, coeff.shape)
        mons = np.array(initial_mons).T
        slices = [mons[i] for i in range(dim)]
        slice_error = np.sum(np.abs(coeff[tuple(slices)]))
        #increment error
        error += slice_error
        if error > abs_approx_tol+rel_approx_tol*inf_norms[num]:
            #FREAK OUT if we can't zero out everything below the lower-reverse-hyperdiagonal
            good_approx = False
        else:
            #try to increment the degree down
            coeff[tuple(slices)] = 0
            deg = coeff.shape[0]-1
            #stop when it gets linear...
            while deg > 1:
                #try to cut off another hyperdiagonal from the coefficient matrix
                mons = mon_combos_limited_wrap(deg, dim, coeff.shape)
                slices = [] #becomes the indices of the terms of degree deg
                mons = np.array(mons).T
                for i in range(dim):
                    slices.append(mons[i])
                slices = tuple(slices)
                slice_error = np.sum(np.abs(coeff[slices]))
                #if that introduces too much error, backtrack
                if slice_error + error > abs_approx_tol+rel_approx_tol*inf_norms[num]:
                    if deg < coeff.shape[0]-1:
                        slices = tuple([slice(0,deg+1)]*dim)
                        coeff = coeff[slices]
                    break
                #otherwise, increment the error
                else:
                    error += slice_error
                    coeff[slices] = 0
                    deg-=1
                    if deg == 1:
                        slices = tuple([slice(0,2)]*dim)
                        coeff = coeff[slices]
                        break
        coeffs[num] = coeff
        errors[num] = error

    return coeffs, good_approx, errors

@Memoize
def mon_combos_limited_wrap(deg, dim, shape):
    '''A wrapper for mon_combos_limited to memoize.

    Parameters
    --------
    deg: int
        Degree of the monomials desired.
    dim : int
        Dimension of the monomials desired.
    shape : tuple
        The limiting shape. The i'th index of the mon can't be bigger than the i'th index of the shape.

    Returns
    -----------
    mon_combo_limited_wrap : list
        A list of all the monomials.
    '''
    return mon_combos_limited([0]*dim,deg,shape)

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
        answers.extend(mon_combos_limited(temp, remaining_degrees-i, shape, cur_dim+1))
    return answers

def good_zeros_1d(zeros, imag_tol, real_tol):
    """Get the real zeros in the -1 to 1 interval

    Parameters
    ----------
    zeros : numpy array
        The zeros to be checked.
    imag_tol : float
        How large the imaginary part can be to still have it be considered real.
    real_tol : float
        How far the real part can be outside the interval [-1,1] and still be
        considered valid.

    Returns
    -------
    good_zeros : numpy array
        The real zeros in [-1,1] of the input zeros.
    """
    zeros = zeros[np.where(np.abs(zeros) <= 1 + real_tol)]
    zeros = zeros[np.where(np.abs(zeros.imag) < imag_tol)]
    return zeros.real

def subdivision_solve_1d(f,a,b,deg,interval_data,root_tracker,tols,max_level,level=0):
    """Finds the roots of a one-dimensional function using subdivision and chebyshev approximation.

    Parameters
    ----------
    f : function from R -> R
        The function to interpolate.
    a : numpy array
        The lower bound on the interval.
    b : numpy array
        The upper bound on the interval.
    deg : int
        The degree of the approximation.
    interval_data : IntervalData
        A class to run the subinterval checks and keep track of the solve progress
    root_tracker : RootTracker
        A class to keep track of the roots that are found.
    tols : Tolerances
        The tolerances to be used.
    max_level : int
        The maximum level for the recursion
    level : int
        The current level of the recursion.

    Returns
    -------
    coeffs : numpy array
        The coefficient of the chebyshev interpolating polynomial.
    """
    if level > max_level:
        # TODO Refine case where there may be a root and it goes too deep.
        interval_data.track_interval("Too Deep", [a, b])
        return

    RAND = 0.5139303900908738
    interval_data.print_progress()
    coeff, inf_norm = interval_approximate_1d(f,a,b,deg)
    coeff2, inf_norm = interval_approximate_1d(f,a,b,deg*2,inf_norm)
    coeff2[slice_top(coeff)] -= coeff
    error = np.sum(np.abs(coeff2))
    if error > tols.abs_approx_tol+tols.rel_approx_tol*inf_norm:
        #Subdivide the interval and recursively call the function.
        div_spot = a + (b-a)*RAND
        subdivision_solve_1d(f,a,div_spot,deg,interval_data,root_tracker,tols,max_level,level+1)
        subdivision_solve_1d(f,div_spot,b,deg,interval_data,root_tracker,tols,max_level,level+1)
    else:
        while error + abs(coeff[-1]) < tols.abs_approx_tol+tols.rel_approx_tol*inf_norm and len(coeff) > 5:
            error += abs(coeff[-1])
            coeff = coeff[:-1]
        try:
            good_zeros_tol = max(tols.min_good_zeros_tol, error*tols.good_zeros_factor)
            zeros = transform(good_zeros_1d(multCheb(coeff),good_zeros_tol,good_zeros_tol),a,b)
            interval_data.track_interval('Spectral', [a,b])
            root_tracker.add_roots(zeros, a, b, 'Spectral')
        except ConditioningError as e:
            div_spot = a + (b-a)*RAND
            subdivision_solve_1d(f,a,div_spot,deg,interval_data,root_tracker,tols,max_level,level+1)
            subdivision_solve_1d(f,div_spot,b,deg,interval_data,root_tracker,tols,max_level,level+1)
