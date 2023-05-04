"""
Subdivision provides a solve function that finds roots of a set of functions
by approximating the functions with Chebyshev polynomials.
When the approximation is performed on a sufficiently small interval,
the approximation degree is small enough to be solved efficiently.

"""

import numpy as np
from scipy.fftpack import fftn
from yroots.OneDimension import divCheb, divPower, multCheb, multPower
from yroots.Multiplication import multiplication
from yroots.utils import clean_zeros_from_matrix, slice_top, MacaulayError, \
                         get_var_list, ConditioningError, TooManyRoots, \
                         Tolerances, solve_linear, memoize, Memoize, transform
from yroots.polynomial import MultiCheb
from yroots.IntervalChecks import IntervalData
from yroots.RootTracker import RootTracker
from itertools import product
from matplotlib import pyplot as plt
from scipy.linalg import lu
import time
import warnings
from numba import jit
from math import log2, ceil

macheps = 2.220446049250313e-16

def subdivide_to_linear(funcs, a, b, rel_approx_tol=1.e-15, abs_approx_tol=1.e-12,
          max_cond_num=1e5, good_zeros_factor=100, min_good_zeros_tol=1e-5,
          check_eval_error=True, check_eval_freq=1, plot=False,
          plot_intervals=False, deg=None, target_deg=1,
          return_potentials=False, method='svd', target_tol=1.01*macheps,
          trust_small_evals=False, intervalReductions=["improveBound", "getBoundingParallelogram"]):
    """
    Finds the real roots of the given list of functions on a given interval.

    All of the tolerances can be passed in as numbers of iterable types. If
    multiple are passed in as iterable types they must have the same length.
    When the length is more than 1, they are used one after the other to polish
    the roots.

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
        Multiplying this by the approximation error gives how far outside of [-1, 1] a root can
        be and still be considered inside the interval.
    min_good_zeros_tol : float or list
        The smallest the good_zeros_tol can be, which is how far outside of [-1, 1] a root can
        be and still be considered inside the interval.
    check_eval_error : bool
        Whether to compute the evaluation error on the fly and replace the approx tol with it.
    check_eval_freq : int
        The evaluation error will be computed on levels that are multiples of this.
    plot : bool
        If True plots the zeros-loci of the functions along with the computed roots
    plot_intervals : bool
        If True, plot is True, and the functions are 2 dimensional, plots what check/method solved
        each part of the interval.
    deg : int
        The degree used for the approximation. If None, the following degrees
        are used.
        Degree 100 for 1D functions.
        Degree 20 for 2D functions.
        Degree 9 for 3D functions.
        Degree 9 for 4D functions.
        Degree 2 for 5D functions and above.
    target_deg : int
        The degree the approximation needs to be trimmed down to before the
        Macaulay solver is called. If unspecified, it will either be 5 (for 2D
        functions) or match the deg argument.
    return_potentials : bool
        If True, returns the potential roots. Else, it does not.
    method : str (optional)
        The method to use when reducing the Macaulay matrix. Valid options are
        svd, tvb, and qrt.
    target_tol : float
        The final absolute approximation tolerance to use before using any sort
        of solver (Macaulay, linear, etc).
    trust_small_evals : bool
        Whether or not to trust function evaluations that may give floats
        smaller than machine epsilon. This should only be set to True if the
        function evaluations are very accurate.
    intervalReductions : list
        A list specifying the types of interval reductions that should be performed
        on each subinterval. The order of methods in the list determines the order
        in which the interval reductions are performed. To stop any interval
        reduction method from being run, pass in an empty list to this parameter.

    If finding roots of a univariate function, `funcs` does not need to be a list,
    and `a` and `b` can be floats instead of arrays.

    Returns
    -------
    zeros : numpy array
        The common zeros of the polynomials. Each row is a root.
    """
    # Detect the dimension
    if isinstance(funcs, list):
        dim = len(funcs)
    elif callable(funcs):
        dim = 1
    else:
        raise ValueError('`funcs` must be a callable or list of callables.')


    # make a and b the right type
    a = np.float64(a)
    b = np.float64(b)

    # Choose an appropriate max degree for the given dimension if none is specified.
    if deg is None:
        deg_dim = {1: 100, 2:20, 3:9, 4:9}
        if dim > 4:
            deg = 2
        else:
            deg = deg_dim[dim]

    # Sets up the tolerances.
    if isinstance(abs_approx_tol, list):
        abs_approx_tol = [max(tol, 1.01*macheps) for tol in abs_approx_tol]
    else:
        abs_approx_tol = max(abs_approx_tol, 1.01*macheps)
    tols = Tolerances(rel_approx_tol=rel_approx_tol,
                      abs_approx_tol=abs_approx_tol,
                      max_cond_num=max_cond_num,
                      good_zeros_factor=good_zeros_factor,
                      min_good_zeros_tol=min_good_zeros_tol,
                      check_eval_error=check_eval_error,
                      check_eval_freq=check_eval_freq,
                      target_tol=target_tol)
    tols.nextTols()

    # Set up the interval data and root tracker classes and cheb blocky copy arr
    interval_data = IntervalData(a, b, intervalReductions)
    root_tracker = RootTracker()
    values_arr.memo = {}
    initialize_values_arr(dim, deg+3)

    if dim == 1:
        # In one dimension, we don't use target_deg; it's the same as deg
        target_deg = deg
        solve_func = subdivision_solve_1d
        if isinstance(funcs, list):
            funcs = funcs[0]
    else:
        solve_func = subdivision_solve_nd

    # TODO : Set the maximum number of subdivisions so that
    # intervals cannot possibly be smaller than 2^-51
    max_level = 52



    # Initial Solve
    solve_func(funcs, a, b, deg, target_deg, interval_data,
               root_tracker, tols, max_level, method=method,
               trust_small_evals=trust_small_evals)
    root_tracker.keep_possible_duplicates()

    # Polishing
    while tols.nextTols():
        polish_intervals = root_tracker.get_polish_intervals()
        interval_data.add_polish_intervals(polish_intervals)
        for new_a, new_b in polish_intervals:
            interval_data.start_polish_interval()
            solve_func(funcs, new_a, new_b, deg, target_deg, interval_data, root_tracker, tols, max_level, method=method)
            root_tracker.keep_possible_duplicates(),
    print("\rPercent Finished: 100%{}".format(' '*50))

    # Print results
    interval_data.print_results()

    # Plotting
    if plot:
        if dim == 1:
            x = np.linspace(a, b, 1000)
            plt.plot(x, funcs(x), color='k')
            plt.plot(np.real(root_tracker.roots), np.zeros(len(root_tracker.roots)), 'o', color = 'none', markeredgecolor='r')
            plt.show()
        elif dim == 2:
            interval_data.plot_results(funcs, root_tracker.roots, plot_intervals)

    if len(root_tracker.potential_roots) != 0:
        warnings.warn("Some intervals subdivided too deep and some potential roots were found. To access these roots, rerun the solver with the keyword return_potentials=True")

    if return_potentials:
        return root_tracker.roots, root_tracker.potential_roots
    else:
        return root_tracker.roots

@Memoize
def initialize_values_arr(dim, deg):
    """Helper function for chebyshev_block_copy.
    Initializes an array to use throughout the whole solve function.
    Builds one array corresponding to dim and deg that can be used for any
    block copy of degree less than deg

    Parameters
    ----------
    dim : int
        Dimension
    deg : int
        Degree

    Returns
    -------
    An empty numpy array that can be used to hold values for a chebyshev_block_copy
    of dimension dim degree < deg.
    """
    return np.empty(tuple([2*deg])*dim, dtype=np.float64)

@Memoize
def values_arr(dim):
    """Helper function for chebyshev_block_copy.
    Finds the array initialized by initialize_values_arr for dimension dim.
    Assumes the degree of the approximation is less than the degree used for
    initialize_values_arr.

    Parameters
    ----------
    dim : int
        Dimension

    Returns
    -------
    An empty numpy array that can be used to hold values for a chebyshev_block_copy
    of dimension dim and degree less than the degree used for initialize_values_arr.
    """
    keys = tuple(initialize_values_arr.memo.keys())
    for idx, k in enumerate(keys):
        if k[0]==dim:
            break
    return initialize_values_arr.memo[keys[idx]]

@memoize
def block_copy_slicers(dim, deg):
    """Helper function for chebyshev_block_copy.
    Builds slice objects to index into the evaluation array to copy
    in preparation for the fft.

    Parameters
    ----------
    dim : int
        Dimension
    dim : int
        Degree of approximation

    Returns
    -------
    block_slicers : list of tuples of slice objects
        Slice objects used to index into the evaluations
    cheb_slicers : list of tuples of slice objects
        Slice objects used to index into the array we're copying evaluations to
    slicer : tuple of slice objets
        Used to index into the portion of that array we're using for the fft input
    """
    block_slicers = []
    cheb_slicers = []
    full_arr_deg = 2*deg
    for block in product([False, True], repeat=dim):
        cheb_idx = [slice(0, deg+1)]*dim
        block_idx = [slice(0, full_arr_deg)]*dim
        for i, flip_dim in enumerate(block):
            if flip_dim:
                cheb_idx[i] = slice(deg+1, full_arr_deg)
                block_idx[i] = slice(deg-1, 0, -1)
        block_slicers.append(tuple(block_idx))
        cheb_slicers.append(tuple(cheb_idx))
    return block_slicers, cheb_slicers, tuple([slice(0, 2*deg)]*dim)

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
    values_cheb = values_arr(dim)
    block_slicers, cheb_slicers, slicer = block_copy_slicers(dim, deg)

    for cheb_idx, block_idx in zip(cheb_slicers, block_slicers):
        try:
            values_cheb[cheb_idx] = values_block[block_idx]
        except ValueError as e:
            if str(e)[:42] == 'could not broadcast input array from shape':
                values_arr.memo[(dim, )] = np.empty(tuple([2*deg])*dim, dtype=np.float64)
                values_cheb = values_arr(dim)
                values_cheb[cheb_idx] = values_block[block_idx]
            else:
                raise ValueError(e)
    return values_cheb[slicer]

def interval_approximate_1d(f, a, b, deg, return_bools=False, return_inf_norm=False):
    """Finds the chebyshev approximation of a one-dimensional function on an
    interval.

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
    return_inf_norm : bool
        Whether to return the inf norm of the function
    Returns
    -------
    coeffs : numpy array
        The coefficient of the chebyshev interpolating polynomial.
    inf_norm : float
        The inf_norm of the function
    """
    extrema = transform(np.cos((np.pi*np.arange(2*deg))/deg), a, b)
    values = f(extrema)

    if return_inf_norm:
        inf_norm = np.max(np.abs(values))

    coeffs = np.real(np.fft.fft(values/deg))
    coeffs[0]/=2
    coeffs[deg]/=2

    if return_bools:
        # Check to see if the sign changes on the interval
        is_positive = values > 0
        sign_change = any(is_positive) and any(~is_positive)
        if return_inf_norm: return coeffs[:deg+1], sign_change, inf_norm
        else:               return coeffs[:deg+1], sign_change
    else:
        if return_inf_norm: return coeffs[:deg+1], inf_norm
        else:               return coeffs[:deg+1]

@memoize
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
        The chebyshev grid used to evaluate the functions in
        interval_approximate_nd
    """
    if has_eval_grid:
        cheb_values = np.cos(np.arange(deg+1)*np.pi/deg)
        return np.column_stack([cheb_values]*dim)
    else:
        cheb_values = np.cos(np.arange(deg+1)*np.pi/deg)
        cheb_grids = np.meshgrid(*([cheb_values]*dim), indexing='ij')
        flatten = lambda x: x.flatten()
        return np.column_stack(tuple(map(flatten, cheb_grids)))

def interval_approximate_nd(f, a, b, deg, return_inf_norm=False):
    """Finds the chebyshev approximation of an n-dimensional function on an
    interval.

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
    return_inf_norm : bool
        whether to return the inf norm of the function

    Returns
    -------
    coeffs : numpy array
        The coefficient of the chebyshev interpolating polynomial.
    inf_norm : float
        The inf_norm of the function
    """
    dim = len(a)
    if dim != len(b):
        raise ValueError("Interval dimensions must be the same!")

    if hasattr(f, "evaluate_grid"):
        cheb_points = transform(get_cheb_grid(deg, dim, True), a, b)
        values_block = f.evaluate_grid(cheb_points)
    else:
        cheb_points = transform(get_cheb_grid(deg, dim, False), a, b)
        values_block = f(*cheb_points.T).reshape(*([deg+1]*dim))

    values = chebyshev_block_copy(values_block)

    if return_inf_norm:
        inf_norm = np.max(np.abs(values_block))

    x0_slicer, deg_slicer, slices, rescale = interval_approx_slicers(dim, deg)
    coeffs = fftn(values/rescale).real
    for x0sl, degsl in zip(x0_slicer, deg_slicer):
        # halve the coefficients in each slice
        coeffs[x0sl] /= 2
        coeffs[degsl] /= 2

    if return_inf_norm:
        return coeffs[tuple(slices)], inf_norm
    else:
        return coeffs[tuple(slices)]

@memoize
def interval_approx_slicers(dim, deg):
    """Helper function for interval_approximate_nd. Builds slice objects to index
    into the output of the fft and divide some of the values by 2 and turn them into
    coefficients of the approximation.

    Parameters
    ----------
    dim : int
        The interpolation dimension.
    deg : int
        The interpolation degree.

    Returns
    -------
    x0_slicer : list of tuples of slice objects
        Slice objects used to index into the the degree 1 monomials
    deg_slicer : list of tuples of slice objects
        Slice objects used to index into the the degree d monomials
    slices : tuple of slice objets
        Used to index into the portion of the array that are coefficients
    rescale : int
        amount to rescale the evaluations by in order to feed them into the fft
    """
    x0_slicer = [tuple([slice(None) if i != d else 0 for i in range(dim)])
                  for d in range(dim)]
    deg_slicer = [tuple([slice(None) if i != d else deg for i in range(dim)])
                  for d in range(dim)]
    slices = tuple([slice(0, deg+1)]*dim)
    return x0_slicer, deg_slicer, slices, deg**dim

def full_cheb_approximate(f, a, b, deg, abs_approx_tol, rel_approx_tol, good_deg=None):
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
        Interpoation degree that is guaranteed to give an approximation valid
        to within approx_tol.

    Returns
    -------
    coeff : numpy array
        The coefficient array of the interpolation. If it can't get a good
        approximation and needs to subdivide, returns None.
    inf_norm : float
        The inf norm of f on [a, b]
    error : float
        The approximation error
    """
    # We don't know what degree we want
    if good_deg is None:
        good_deg = deg
    # Try degree deg and see if it's good enough
    coeff = interval_approximate_nd(f, a, b, good_deg)
    coeff2, inf_norm = interval_approximate_nd(f, a, b, good_deg*2, return_inf_norm=True)
    coeff2[slice_top(coeff.shape)] -= coeff

    error = np.sum(np.abs(coeff2))
    if error > abs_approx_tol+rel_approx_tol*inf_norm:
        return None, inf_norm, error
    else:
        return coeff, inf_norm, error


def zeros_in_interval(zeros, a, b, dim, within_interval_tol=1e-9):
    """Returns the zeros that are only in the interval [a, b].

    Parameters
    ----------
        zeros : numpy array
            The zeros found using the solver.
        a : numpy array
            The lower bounds of the interval for each variable.
        b : numpy array
            The upper bounds of the interval for each variable.
        dim : int
            The dimension of the system.

    Returns
    -------
        zeros : numpy array
            The zeros that are in the interval [a, b]
    """
    # Check along each axis to ensure roots are within the boundaries
    for i in range(dim):
        zeros = zeros[zeros[:, i] - a[i] >= -within_interval_tol]
        zeros = zeros[zeros[:, i] - b[i] <= within_interval_tol]

    return zeros


def good_zeros_nd(zeros, imag_tol, real_tol):
    """Get the real zeros in the -1 to 1 interval in each dimension.

    Parameters
    ----------
    zeros : numpy array
        The zeros to be checked.
    imag_tol : float
        How large the imaginary part can be to still have it be considered real.
    real_tol : float
        How far the real part can be outside the interval [-1, 1]^n and still be
        considered valid.

    Returns
    -------
    good_zeros : numpy array
        The real zeros in [-1, 1]^n of the input zeros.
    """
    # Take care of the case where we found only 1 root
    if len(zeros.shape) == 1:
        mask = np.all(np.abs(zeros.imag) <= imag_tol, axis = 0)
        mask *= np.all(np.abs(zeros.real) <= 1 + real_tol, axis = 0)
    else:
        mask = np.all(np.abs(zeros.imag) <= imag_tol, axis = 1)
        mask *= np.all(np.abs(zeros.real) <= 1 + real_tol, axis = 1)
    return zeros[mask].real

def get_abs_approx_tol(func, deg, a, b, dim):
    """ Gets an absolute approximation tolerance based on the assumption that
        on the interval of size linearization_size * 2, the function can be
        perfectly approximated by a low degree Chebyshev polynomial.

        Parameters
        ----------
            func : function
                Function to approximate.
            deg : int
                The degree to use to approximate the function on the interval.
            a : numpy array
                The lower bounds of the interval on which to approximate.
            b : numpy array
                The upper bounds of the interval on which to approximate.

        Returns
        -------
            abs_approx_tol : float
                The calculated absolute approximation tolerance based on the
                noise of the function on the small interval.
    """
    # Half the width of the smaller interval
    linearization_size = 1e-14


    # Get a random small interval from [-1, 1] and transform so it's
    # within [a, b]
    x = transform(random_point(dim), a, b)
    a2 = np.array(x - linearization_size)
    b2 = np.array(x + linearization_size)

    # Approximate with a low degree Chebyshev polynomial
    coeff = interval_approximate_nd(func, a2, b2, 2*deg)
    coeff[deg_slices(deg, dim)] = 0

    # Sum up coeffieicents that are assumed to be just noise
    abs_approx_tol = np.sum(np.abs(coeff))

    # Divide by the number of spots that were summed up.
    numSpots = (deg*2)**dim - (deg)**dim

    # Multiply by 10 to give a looser tolerance (speed-up)
    # print(abs_approx_tol*10 / numSpots)
    return abs_approx_tol*10 / numSpots

@memoize
def deg_slices(deg, dim):
    """Helper function for get_abs_approx_tol. Returns a slice object for
    accessing all the terms of total degree less than deg in a coefficient
    tensor.

    Parameters
    ----------
        deg : int
            The degree of the Chebsyhev interpolation.
        dim : int
            The dimension of the system.

    Returns
    -------
        slice
            The slice that accesses all the coefficients of degree less than
            deg.
    """
    return (slice(0, deg), )*dim

@memoize
def random_point(dim):
    """Gets a random point from [-1, 1]^dim that's used for get_abs_approx_tol.
    Since this is memoized, subsequent calls will be a lot faster.

    Parameters
    ----------
        dim : int
            The dimension of the system/how many samples to take from [0, 1].

    Returns
    -------
        numpy array
            The random point that haas dim entries.
    """
    np.random.seed(0)
    # Scale the points so that they're each within [-1, 1]
    return np.random.rand(dim)*2 - 1

def subdivision_solve_nd(funcs, a, b, deg, target_deg, interval_data,
                         root_tracker, tols, max_level,good_degs=None, level=0,
                         method='svd', use_target_tol=False,
                         trust_small_evals=False):
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
    target_deg : int
        The degree to subdivide down to before building the Macaulay matrix.
    interval_data : IntervalData
        A class to run the subinterval checks and keep track of the solve
        progress
    root_tracker : RootTracker
        A class to keep track of the roots that are found.
    tols : Tolerances
        The tolerances to be used.
    max_level : int
        The maximum level for the recursion
    good_degs : numpy array
        Interpoation degrees that are guaranteed to give an approximation valid
        to within approx_tol.
    level : int
        The current level of the recursion.
    method : str (optional)
        The method to use when reducing the Macaulay matrix. Valid options are
        svd, tvb, and qrt.
    use_target_tol : bool
        Whether or not to use tols.target_tol when making approximations. This
        is necessary to get a sufficiently accurate approximation from which to
        build the Macaulay matrix and run the solver.
    """

    if level >= max_level:
        # TODO Refine case where there may be a root and it goes too deep.
        interval_data.track_interval("Too Deep", [a, b])
        # Return potential roots if the residuals are small
        root_tracker.add_potential_roots((a + b)/2, a, b, "Too Deep.")
        return

    dim = len(a)

    if tols.check_eval_error:
        # Using the first abs_approx_tol
        if not use_target_tol:
            tols.abs_approx_tol = tols.abs_approx_tols[tols.currTol]
            if level%tols.check_eval_freq == 0:
                numSpots = (deg*2)**len(a) - (deg)**len(a)
                for func in funcs:
                    tols.abs_approx_tol = max(tols.abs_approx_tol, numSpots * get_abs_approx_tol(func, 3, a, b, dim))
        # Using target_tol
        else:
            tols.target_tol = tols.target_tols[tols.currTol]
            if level%tols.check_eval_freq == 0:
                numSpots = (deg*2)**len(a) - (deg)**len(a)
                for func in funcs:
                    tols.target_tol = max(tols.target_tol, numSpots * get_abs_approx_tol(func, 3, a, b, dim))

    # Buffer the interval to solve on a larger interval to account for
    # corners. Right now, it's set to be 5e-10 so that on [-1, 1], the
    # buffer goes out 1e-9 around the initial search interval.
    # DETERMINED BY EXPERIMENTATION
    interval_buffer_size = (b - a) * 5e-10
    og_a = a.copy()
    og_b = b.copy()
    a -= interval_buffer_size
    b += interval_buffer_size

    cheb_approx_list = []
    interval_data.print_progress()
    if good_degs is None:
        good_degs = [None]*len(funcs)
    inf_norms = []
    approx_errors = []
    # Get the chebyshev approximations
    num_funcs = len(funcs)
    for func_num, (func, good_deg) in enumerate(zip(funcs, good_degs)):
        if use_target_tol:
            coeff, inf_norm, approx_error = full_cheb_approximate(func, a, b, deg, tols.target_tol, tols.rel_approx_tol, good_deg)
        else:
            coeff, inf_norm, approx_error = full_cheb_approximate(func, a, b, deg, tols.abs_approx_tol, tols.rel_approx_tol, good_deg)
        inf_norms.append(inf_norm)
        approx_errors.append(approx_error)
        # Subdivides if a bad approximation
        if coeff is None:
            if not trust_small_evals:
                approx_errors = [max(err,macheps) for err in approx_errors]
            intervals = interval_data.get_subintervals(og_a, og_b, cheb_approx_list, approx_errors, False)

            #reorder funcs. TODO: fancier things like how likely it is to pass checks
            funcs2 = funcs.copy()
            if func_num + 1 < num_funcs:
                del funcs2[func_num]
                funcs2.append(func)
            for new_a, new_b in intervals:
                subdivision_solve_nd(funcs2,new_a,new_b,deg,target_deg,interval_data,root_tracker,tols,max_level,level=level+1, method=method, trust_small_evals=trust_small_evals)
            return
        else:
            # Run checks to try and throw out the interval
            if not trust_small_evals:
                approx_error = max(approx_error, macheps)
            if interval_data.check_interval(coeff, approx_error, og_a, og_b):
                return

            cheb_approx_list.append(coeff)

    # Reduce the degree of the approximations while not introducing too much error
    coeffs, good_approx, approx_errors = trim_coeffs(cheb_approx_list, tols.abs_approx_tol, tols.rel_approx_tol, inf_norms, approx_errors)
    if not trust_small_evals:
        approx_errors = [max(err, macheps) for err in approx_errors]
    # Used if subdividing further.
    # Only choose good_degs if the approximation after trim_coeffs is good.
    if good_approx:
        # good_degs are assumed to be 1 higher than the current approximation
        # but no larger than the initial degree for more accurate performance.
        good_degs = [min(coeff.shape[0], deg) for coeff in coeffs]
        good_zeros_tol = max(tols.min_good_zeros_tol, sum(np.abs(approx_errors))*tols.good_zeros_factor)

    # Check if the degree is small enough or if trim_coeffs introduced too much error
    if np.any(np.array([coeff.shape[0] for coeff in coeffs]) > target_deg + 1) or not good_approx:
        intervals = interval_data.get_subintervals(og_a, og_b, cheb_approx_list, approx_errors, True)
        for new_a, new_b in intervals:
            subdivision_solve_nd(funcs, new_a, new_b, deg, target_deg, interval_data, root_tracker, tols, max_level, good_degs, level+1, method=method, trust_small_evals=trust_small_evals, use_target_tol=True)

    # Check if any approx error is greater than target_tol for Macaulay method
    elif np.any(np.array(approx_errors) > np.array(tols.target_tol) + tols.rel_approx_tol*np.array(inf_norms)):
        intervals = interval_data.get_subintervals(og_a, og_b, cheb_approx_list, approx_errors, True)
        for new_a, new_b in intervals:
            subdivision_solve_nd(funcs, new_a, new_b, deg, target_deg, interval_data, root_tracker, tols, max_level, good_degs, level+1, method=method, trust_small_evals=trust_small_evals, use_target_tol=True)

    # Check if everything is linear
    elif np.all(np.array([coeff.shape[0] for coeff in coeffs]) == 2):
        if deg != 2:
            subdivision_solve_nd(funcs, a, b, 2, target_deg, interval_data, root_tracker, tols, max_level, good_degs, level, method=method, trust_small_evals=trust_small_evals, use_target_tol=True)
            return
        zero, cond = solve_linear(coeffs)
        # Store the information and exit
        zero = good_zeros_nd(zero, good_zeros_tol, good_zeros_tol)
        zero = transform(zero, a, b)
        zero = zeros_in_interval(zero, og_a, og_b, dim)
        interval_data.track_interval("Base Case", [a, b])
        root_tracker.add_roots(zero, a, b, "Base Case")

    # Solve using spectral methods if stable.
    else:
        polys = [MultiCheb(coeff, lead_term = [coeff.shape[0]-1], clean_zeros = False) for coeff in coeffs]
        res = multiplication(polys, max_cond_num=tols.max_cond_num, method=method)
        #check for a conditioning error
        if res[0] is None:
            # Subdivide but run some checks on the intervals first
            intervals = interval_data.get_subintervals(og_a, og_b, cheb_approx_list, approx_errors, True)
            for new_a, new_b in intervals:
                subdivision_solve_nd(funcs, new_a, new_b, deg, target_deg, interval_data, root_tracker, tols, max_level, good_degs, level+1, method=method, trust_small_evals=trust_small_evals, use_target_tol=True)
        else:
            zeros = res
            zeros = good_zeros_nd(zeros, good_zeros_tol, good_zeros_tol)
            zeros = transform(zeros, a, b)
            zeros = zeros_in_interval(zeros, og_a, og_b, dim)
            interval_data.track_interval("Macaulay", [a, b])
            root_tracker.add_roots(zeros, a, b, "Macaulay")

def trim_coeffs(coeffs, abs_approx_tol, rel_approx_tol, inf_norms, errors):
    """Trim the coefficient matrices to reduce the degree by zeroing out any
    entries in the coefficient matrix above a certain degree.

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
    # Assume we start with good approximations
    good_approx = True
    for num, coeff in enumerate(coeffs):
        # Get the error inherent in the approximation
        error = errors[num]

        # Try to zero out everything below the lower-reverse-hyperdiagonal
        # that's a fancy way of saying monomials that are more than the specified degree
        dim = coeff.ndim
        deg = np.sum(coeff.shape) - dim - 1
        initial_mons = []
        for deg0 in range(coeff.shape[0], deg+1):
            initial_mons += mon_combos_limited_wrap(deg0, dim, coeff.shape)
        mons = np.array(initial_mons).T
        slices = tuple(mons[:dim])
        slice_error = np.sum(np.abs(coeff[slices]))
        # increment error
        error += slice_error
        if error > abs_approx_tol+rel_approx_tol*inf_norms[num]:
            # FREAK OUT if we can't zero out everything below the lower-reverse-hyperdiagonal
            good_approx = False
        else:
            # try to increment the degree down
            coeff[slices] = 0
            deg = coeff.shape[0]-1
            # stop when it gets linear...
            while deg > 1:
                # try to cut off another hyperdiagonal from the coefficient matrix
                mons = mon_combos_limited_wrap(deg, dim, coeff.shape)
                mons = np.array(mons).T
                slices = tuple(mons[:dim])
                slice_error = np.sum(np.abs(coeff[slices]))
                # if that introduces too much error, backtrack
                if slice_error + error > abs_approx_tol+rel_approx_tol*inf_norms[num]:
                    if deg < coeff.shape[0]-1:
                        slices = tuple([slice(0, deg+1)]*dim)
                        coeff = coeff[slices]
                    break
                # otherwise, increment the error
                else:
                    error += slice_error
                    coeff[slices] = 0
                    deg-=1
                    if deg == 1:
                        slices = tuple([slice(0, 2)]*dim)
                        coeff = coeff[slices]
                        break
        coeffs[num] = coeff
        errors[num] = error

    return coeffs, good_approx, errors

@memoize
def mon_combos_limited_wrap(deg, dim, shape):
    """A wrapper for mon_combos_limited to memoize.

    Parameters
    --------
    deg: int
        Degree of the monomials desired.
    dim : int
        Dimension of the monomials desired.
    shape : tuple
        The limiting shape. The i'th index of the mon can't be bigger than the
        i'th index of the shape.

    Returns
    -----------
    mon_combo_limited_wrap : list
        A list of all the monomials.
    """
    return mon_combos_limited([0]*dim, deg, shape)

def mon_combos_limited(mon, remaining_degrees, shape, cur_dim = 0):
    """Finds all the monomials of a given degree that fits in a given shape and
     returns them. Works recursively.

    Very similar to mon_combos, but only returns the monomials of the desired
    degree.

    Parameters
    --------
    mon: list
        A list of zeros, the length of which is the dimension of the desired
        monomials. Will change as the function searches recursively.
    remaining_degrees : int
        Initially the degree of the monomials desired. Will decrease as the
        function searches recursively.
    shape : tuple
        The limiting shape. The i'th index of the mon can't be bigger than the
        i'th index of the shape.
    cur_dim : int
        The current position in the list the function is iterating through.
        Defaults to 0, but increases in each step of the recursion.

    Returns
    -----------
    answers : list
        A list of all the monomials.
    """
    answers = []
    if len(mon) == cur_dim+1: # We are at the end of mon, no more recursion.
        if remaining_degrees < shape[cur_dim]:
            mon[cur_dim] = remaining_degrees
            answers.append(mon.copy())
        return answers
    if remaining_degrees == 0: # Nothing else can be added.
        answers.append(mon.copy())
        return answers
    temp = mon.copy() # Quicker than copying every time inside the loop.
    for i in range(min(shape[cur_dim], remaining_degrees+1)): # Recursively add to mon further down.
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
        How far the real part can be outside the interval [-1, 1] and still be
        considered valid.

    Returns
    -------
    good_zeros : numpy array
        The real zeros in [-1, 1] of the input zeros.
    """
    zeros = zeros[np.where(np.abs(zeros) <= 1 + real_tol)]
    zeros = zeros[np.where(np.abs(zeros.imag) < imag_tol)]
    return zeros.real

def subdivision_solve_1d(f, a, b, deg, target_deg, interval_data, root_tracker,
                         tols, max_level, level=0, method='svd',
                         trust_small_evals=False):
    """Finds the roots of a one-dimensional function using subdivision and
    chebyshev approximation.

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
    target_deg : int
        The degree to subdivide down to before building the Macauly matrix.
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


    # Determine the point at which to subdivide the interval
    RAND = 0.5139303900908738
    interval_data.print_progress()

    # Approximate the function using Chebyshev polynomials
    coeff = interval_approximate_1d(f, a, b, deg)
    coeff2, sign_change, inf_norm = interval_approximate_1d(f, a, b, deg*2, return_bools=True, return_inf_norm=True)

    coeff2[slice_top(coeff.shape)] -= coeff

    # Calculate the approximate error between the deg and 2*deg approximations
    error = np.sum(np.abs(coeff2))
    allowed_error = tols.abs_approx_tol+tols.rel_approx_tol*inf_norm

    if error > allowed_error:
        # Subdivide the interval and recursively call the function.
        div_spot = a + (b-a)*RAND
        good_deg = deg
        subdivision_solve_1d(f, a, div_spot, good_deg, target_deg, interval_data, root_tracker, tols, max_level, level+1)
        subdivision_solve_1d(f, div_spot, b, good_deg, target_deg, interval_data, root_tracker, tols, max_level, level+1)
    else:
        # Trim the coefficient array (reduce the degree) as much as we can.
        # This identifies a 'good degree' with which to approximate the function
        # if it is less than the given approx degree.
        last_coeff_size = abs(coeff[-1])
        new_error = error + last_coeff_size
        while new_error < allowed_error:
            if len(coeff) == 1:
                break
            #maybe a list pop here? idk if worth it to switch away from arrays
            coeff = coeff[:-1]
            last_coeff_size = abs(coeff[-1])
            error = new_error
            new_error = error + last_coeff_size
        if not trust_small_evals:
            error = max(error, macheps)
        good_deg = max(len(coeff) - 1, 1)

        # Run interval checks to eliminate regions
        if not sign_change: # Skip checks if there is a sign change
            if interval_data.check_interval(coeff, error, a, b):
                return

        try:
            good_zeros_tol = max(tols.min_good_zeros_tol, error*tols.good_zeros_factor)
            zeros = transform(good_zeros_1d(multCheb(coeff), good_zeros_tol, good_zeros_tol), a, b)
            interval_data.track_interval("Macaulay", [a, b])
            root_tracker.add_roots(zeros, a, b, "Macaulay")
        except (ConditioningError, TooManyRoots) as e:
            div_spot = a + (b-a)*RAND
            subdivision_solve_1d(f, a, div_spot, good_deg, target_deg, interval_data, root_tracker, tols, max_level, level+1)
            subdivision_solve_1d(f, div_spot, b, good_deg, target_deg, interval_data, root_tracker, tols, max_level, level+1)
