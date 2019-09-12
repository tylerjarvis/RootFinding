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
                        ConditioningError
from yroots.polynomial import MultiCheb
from yroots.IntervalChecks import IntervalData
from itertools import product
from matplotlib import pyplot as plt
from scipy.linalg import lu
import itertools
import time
import warnings

def solve(funcs, a, b, plot = False, plot_intervals = False, polish = False, approx_tol=1.e-6):
    '''
    Finds the real roots of the given list of functions on a given interval.

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
    plot : bool
        If True plots the zeros-loci of the functions along with the computed roots
    plot_intervals : bool
        If True, plot is True, and the functions are 2 dimensional, plots what check/method solved
        each part of the interval.
    polish : bool
        If True resolves for each root on a smaller interval with a finer approximation to give a
        more accurate answer.

    If finding roots of a univariate function, `funcs` does not need to be a list,
    and `a` and `b` can be floats instead of arrays.

    returns
    -------
    zeros : numpy array
        The common zeros of the polynomials. Each row is a root.
    '''
    if not isinstance(funcs,list):
        funcs = [funcs]
        dim = 1
    else:
        dim = len(funcs)

    if dim == 1:
        #one dimensional case
        interval_data = IntervalData(a,b)
        zeros = subdivision_solve_1d(funcs[0],a,b,interval_data, cheb_approx_tol=approx_tol)
        if plot:
            x = np.linspace(a,b,1000)
            for f in funcs:
                plt.plot(x,f(x),color='k')
            plt.plot(np.real(zeros),np.zeros(len(zeros)),'o',color = 'none',markeredgecolor='r')
            plt.show()
        print("\rPercent Finished: 100%       ")
        interval_data.print_results()
        return zeros
    else:
        #multidimensional case

        #make a and b the right type
        a = np.float64(a)
        b = np.float64(b)

        interval_data = IntervalData(a,b)

        #choose an appropriate max degree for the given dimension
        deg_dim = {2:9, 3:5, 4:3}
        if dim > 4:
            deg = 2
        else:
            deg = deg_dim[dim]

        #Output the interval percentages
        zeros = subdivision_solve_nd(funcs,a,b,deg,interval_data,polish=polish, approx_tol=approx_tol)

        print("\rPercent Finished: 100%       ")
        interval_data.print_results()
        #Plot what happened
        if plot and dim == 2:
#             interval_data.print_results()
            interval_data.plot_results(funcs, zeros, plot_intervals)
        return zeros

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

    multiplier = None
    if multiplier is None:
        if np.max(np.abs(values)) < 1.e-50:
            multiplier = 1.e50
        else:
            multiplier = 1./np.max(np.abs(values))
    multiplier = max(1, multiplier)
#     multiplier = 1.
    values *= multiplier

    coeffs = np.real(np.fft.fft(values/deg))
    coeffs[0]/=2
    coeffs[deg]/=2
    return coeffs[:deg+1]

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

def interval_approximate_nd(f,a,b,deg,return_bools=False,multiplier=None):
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
        whether to return bools which indicate if the funtion changes sign or not

    Returns
    -------
    coeffs : numpy array
        The coefficient of the chebyshev interpolating polynomial.
    change_sign: numpy array (Optional)
        list of which subintervals change sign
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
        #Checks are fast enough that this isn't worth it

#         change_sign = np.ones(2**dim, dtype=bool)

#         split = 0.027860780181747646 #from RAND below
#         split_point = len(np.where(cheb_values>split)[0])

#         for k, subinterval in enumerate(product([False,True], repeat=dim)):
#             slicer = []*dim
#             for i in range(dim):
#                 if subinterval[i]:
#                     slicer.append(slice(split_point,None))
#                 else:
#                     slicer.append(slice(None,split_point))

#             if np.all(values_block[tuple(slicer)]>0) or np.all(values_block[tuple(slicer)]<0):
#                 change_sign[k] = False

    values = chebyshev_block_copy(values_block)

#     if multiplier is None:
#         if np.max(np.abs(values)) < 1.e-5:
#             multiplier = 1.e5
#         else:
#             multiplier = 1./np.max(np.abs(values))
#     multiplier = max(1, multiplier)
    multiplier = 1.
    values *= multiplier

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
        return coeffs[tuple(slices)], change_sign, multiplier
    else:
        return coeffs[tuple(slices)], multiplier

def get_subintervals(a,b,dimensions,interval_data,polys,change_sign,approx_tol,check_subintervals=False):
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
    approx_tol: float
        The bound of the sup norm error of the chebyshev approximation. Because trim_coeff
        introduces this much error again, 2*approx_tol is passed into the subintervals checks.
    check_subintervals : bool
        If True runs the subinterval checks to throw out intervals where the functions are never 0.

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
        scaled_subintervals = get_subintervals(-np.ones_like(a),np.ones_like(a),dimensions,None,None,None,approx_tol)
        #Uses 2*approx_tol because this much error can be added in the approximation and the trim_coeff
        # TODO Should be approx_tol/2? Or not since we account for that earlier?
        # TODO Mabye should be (200/87)* approx tol because it's the original tolerance given,
        # not necessarily the modified tolerance.
        return interval_data.check_subintervals(subintervals, scaled_subintervals, polys, change_sign, approx_tol)
    else:
        return subintervals

def full_cheb_approximate(f,a,b,deg,approx_tol,good_deg=None):
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
    approx_tol : float
        How small the high degree terms must be to consider the approximation accurate.
    good_deg : numpy array
        Interpoation degree that is guaranteed to give an approximation valid to within approx_tol.

    Returns
    -------
    coeff : numpy array
        The coefficient array of the interpolation. If it can't get a good approximation and needs to subdivide, returns None.
    bools: numpy array
        (2^n, 1) array of bools corresponding to which subintervals the function changes sign in
    """
    #We know what degree we want
    if good_deg is not None:
        coeff, bools, multiplier = interval_approximate_nd(f,a,b,good_deg,return_bools=True)
        return coeff, bools
        
    # This is for trim_coeffs (*1/2) 
    approx_tol /= 2

    # Machine Epsilon
    #eps = 7./3. - 4./3 - 1

    #Try degree deg and see if it's good enough
    coeff, multiplier = interval_approximate_nd(f,a,b,deg)
    coeff2, bools, multiplier = interval_approximate_nd(f,a,b,deg*2,return_bools=True, multiplier=multiplier)
    coeff2[slice_top(coeff)] -= coeff

    if np.sum(np.abs(coeff2)) > approx_tol: 
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
        return None, np.arange(dim)
    else:
        return coeff, bools

def good_zeros_nd(zeros, imag_tol = 1.e-5, real_tol = 1.e-5):
    """Get the real zeros in the -1 to 1 interval in each dimension.

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
    good_zeros = zeros[np.all(np.abs(zeros.imag) <= imag_tol,axis = 1)]
    good_zeros = good_zeros[np.all(np.abs(good_zeros) <= 1 + real_tol,axis = 1)]
    return good_zeros.real

def subdivision_solve_nd(funcs,a,b,deg,interval_data,approx_tol=1.e-12,solve_tol=1.e-8, polish=False, good_degs=None, level=0, max_level=25):
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
    interval_data : IntervalData
        A class to run the subinterval checks and keep track of the solve progress
    approx_tol: float
        The bound of the sup norm error of the chebyshev approximation.
    solve_tol : float
        The tolerance to pass into division solve.
    polish : bool
        If True resolves for each root on a smaller interval with a finer approximation to give a
        more accurate answer.
    good_degs : numpy array
        Interpoation degrees that are guaranteed to give an approximation valid to within approx_tol.

    Returns
    -------
    zeros : numpy array
        The real zeros of the functions in the interval [a,b]
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
        return np.zeros([0,len(a)])

    cheb_approx_list = []
    interval_data.print_progress()
    dim = len(a)
    if good_degs is None:
        good_degs = [None]*len(funcs)

    for func, good_deg in zip(funcs, good_degs):
        coeff, change_sign = full_cheb_approximate(func,a,b,deg,approx_tol,good_deg)
#         print(change_sign)
        #Subdivides if a bad approximation
        if coeff is None:
#             print("Bad Approx")
            intervals = get_subintervals(a,b,change_sign,None,None,None,approx_tol)
#             print(intervals, good_degs)
            return np.vstack([subdivision_solve_nd(funcs,interval[0],interval[1],deg,interval_data,\
                                                   approx_tol,solve_tol,polish,level=level+1) for interval in intervals])
        else:
            #if the function changes sign on at least one subinterval, skip the checks
            if np.any(change_sign):
                cheb_approx_list.append(coeff)
                continue
            #Run checks to try and throw out the interval
            if interval_data.check_interval(coeff, approx_tol, a, b):
                return np.zeros([0,dim])

            cheb_approx_list.append(coeff)

#     print("Valid Approx")
    #Make the system stable to solve
    
    coeffs, all_triangular = trim_coeffs(cheb_approx_list, approx_tol)

    #Check if everything is linear
#     print([coeff.shape[0] for coeff in coeffs])
    if np.all(np.array([coeff.shape[0] for coeff in coeffs]) == 2):
#         print("Linear")
#        if approx_tol > 1.e-6:
#            return subdivision_solve_nd(funcs,a,b,deg,interval_data,1.e-8,1.e-8,polish,level=level)
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
            zero = np.linalg.solve(A,-B)
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
                if U.shape[1]-1 in pivot_columns:
                    #dependent
                    return np.zeros([0,dim])
                else:
                    #independent
                    warnings.warn('System potentially has infinitely many roots')
                    return np.zeros([0,dim])

        interval_data.track_interval("Base Case", [a,b])
        if polish:
            polish_tol = (b[0]-a[0])/10
            return polish_zeros(transform(good_zeros_nd(zero.reshape([1,dim])),a,b), funcs, polish_tol)
        else:
            return transform(good_zeros_nd(zero.reshape([1,dim])),a,b)
    #Check if anything is linear
    elif np.any(np.array([coeff.shape[0] for coeff in coeffs]) == 2):
        #Subdivide but run some checks on the intervals first
        intervals = get_subintervals(a,b,np.arange(dim),interval_data,cheb_approx_list,change_sign,\
                                             approx_tol,check_subintervals=True)
        if len(intervals) == 0:
            return np.zeros([0,dim])
        else:
            good_degs = [coeff.shape[0] - 1 for coeff in coeffs]
            return np.vstack([subdivision_solve_nd(funcs,interval[0],interval[1],deg,interval_data,\
                                                   approx_tol,solve_tol,polish,good_degs,level=level+1) for interval in intervals])

    if np.any(np.array([coeff.shape[0] for coeff in coeffs]) > 5) or not all_triangular:
        #Subdivide but run some checks on the intervals first
        intervals = get_subintervals(a,b,np.arange(dim),interval_data,cheb_approx_list,\
                                             change_sign,approx_tol,check_subintervals=True)
        if len(intervals) == 0:
            return np.zeros([0,dim])
        else:
            good_degs = [coeff.shape[0] - 1 for coeff in coeffs]
            return np.vstack([subdivision_solve_nd(funcs,interval[0],interval[1],deg,interval_data,\
                                                   approx_tol,solve_tol,polish,good_degs,level=level+1)\
                                                   for interval in intervals])

    polys = [MultiCheb(coeff, lead_term = [coeff.shape[0]-1], clean_zeros = False) for coeff in coeffs]
    
    # zeros = division(polys,divisor_var,solve_tol)
    try:
        zeros = multiplication(polys, approx_tol=approx_tol, solve_tol=solve_tol)
        zeros = np.array(zeros)
        interval_data.track_interval("Spectral", [a,b])
        if len(zeros) == 0:
            return np.zeros([0,dim])
        if polish:
            polish_tol = (b[0]-a[0])/10
            return polish_zeros(transform(good_zeros_nd(zeros),a,b), funcs, polish_tol)
        else:
            return transform(good_zeros_nd(zeros),a,b)
    except ConditioningError as e:
        # COMMENT OUT IF NOT USING DIVISION
        # divisor_var += 1
        # while divisor_var < dim:
        #     if not good_direc(coeffs,divisor_var,solve_tol):
        #         divisor_var += 1
        #         continue
        #     zeros = division(polys, divisor_var, solve_tol)
        #     if isinstance(zeros, int):
        #         divisor_var += 1
        #         continue
        #     zeros = np.array(zeros)
        #     interval_data.track_interval("Division", [a,b])
        #     if len(zeros) == 0:
        #         return np.zeros([0,dim])
        #     if polish:
        #         polish_tol = (b[0]-a[0])
        #         return polish_zeros(transform(good_zeros_nd(zeros),a,b),funcs,polish_tol)
        #     else:
        #         return transform(good_zeros_nd(zeros),a,)b
        # END COMMENT OUT FOR DIVISION

        #Subdivide but run some checks on the intervals first
        intervals = get_subintervals(a,b,np.arange(dim),interval_data,cheb_approx_list,change_sign,\
                                             approx_tol,check_subintervals=True)
        if len(intervals) == 0:
            return np.zeros([0,dim])
        else:
            good_degs = [poly.coeff.shape[0] - 1 for poly in polys]
            return np.vstack([subdivision_solve_nd(funcs,interval[0],interval[1],deg,interval_data,\
                                                   approx_tol,solve_tol,polish,good_degs,level=level+1) for interval in intervals])

def good_direc(coeffs, dim, solve_tol):
    """Determines if this is a good direction to try solving with division.

    Parameters
    ----------
    coeffs : list
        The coefficient matrices of the polynomials to solve.
    dim : int
        The direction to divide by in division.
    solve_tol : float
        How small spots that will be in the Macaulay diagonal are allowed to get before we determine it is unstable.

    Returns
    -------
    good_direc : bool
        If True running division should be stable. If False, probably not.
    """
#     return True
    tol = solve_tol*100
    slices = []
    for i in range(coeffs[0].ndim):
        if i == dim:
            slices.append(0)
        else:
            slices.append(slice(None))
    vals = [coeff[tuple(slices)] for coeff in coeffs]
    degs = [val.shape[0] for val in vals]

    min_vals = np.zeros([len(vals),*vals[np.argmax(degs)].shape])

    for num, val in enumerate(vals):
        deg = degs[num]
        slices = [num]+[slice(0,deg) for i in range(val.ndim)]
        min_vals[tuple(slices)] = val

    min_vals[min_vals==0] = 1
    if np.any(np.min(np.abs(min_vals),axis=0) < tol):
        return False
    return True

def polish_zeros(zeros, funcs, tol=1.e-1):
    """Polishes the given zeros of the functions to a better accuracy.

    Resolves with finer tolerances in a box around the zeros.

    Parameters
    ----------
    zeros : numpy array
        The given zeros.
    funcs : list
        The functions to find the zeros of.
    tol : float
        How big of a box around the found zeros to solve on.

    Returns
    -------
    polish_zeros : numpy
        The polished zeros.
    """
    import warnings
    warnings.warn("Polishing may return duplicate zeros.")
    
    if len(zeros) == 0:
        return zeros
    dim = zeros.shape[1]
    polished_zeros = []

    for zero in zeros:
        a = np.array(zero) - tol
        b = np.array(zero) + 1.1*tol #Keep the root away from 0
        interval_data = IntervalData(a,b)
        interval_data.polishing = True
        # TODO : Change the approx_tol to make polishing much more accurate.
        polished_zero = subdivision_solve_nd(funcs,a,b,5,interval_data,approx_tol=1.e-13,\
                                                 solve_tol=1.e-15,polish=False)
        polished_zeros.append(polished_zero)
    return np.vstack(polished_zeros)

def trim_coeffs(coeffs, approx_tol):
    """Trim the coefficient matrices so they are stable and choose a direction to divide in.

    Parameters
    ----------
    coeffs : list
        The coefficient matrices of the Chebyshev polynomials we are solving.
    approx_tol: float
        The bound of the sup norm error of the chebyshev approximation.

    Returns
    -------
    polys : list
        The reduced degree Chebyshev polynomials
    divisor_var : int
        What direction to do the division in to be stable. -1 means we should subdivide.
    """
    # This is for trim_coeffs 
    approx_tol /= 2

    all_triangular = True
    for num, coeff in enumerate(coeffs):
        error = 0.
        # Potentially hard-coded tolerance: 1e-10 gets things that aren't really small
        spot = np.abs(coeff) < 1.e-10*np.max(np.abs(coeff))
        error = np.sum(np.abs(coeff[spot]))
        coeff[spot] = 0
        
        dim = coeff.ndim
        deg = np.sum(coeff.shape) - dim
        initial_mons = []
        for deg0 in range(coeff.shape[0], deg+1):
            initial_mons += mon_combos_limited_wrap(deg0, dim, coeff.shape)
        mons = np.array(initial_mons).T
        slices = [mons[i] for i in range(dim)]
        slice_error = np.sum(np.abs(coeff[tuple(slices)]))
        if slice_error + error > approx_tol:
            all_triangular = False
        else:
            coeff[tuple(slices)] = 0
            deg = coeff.shape[0]-1
            while deg > 1:
                mons = mon_combos_limited_wrap(deg, dim, coeff.shape)
                slices = [] #becomes the indices of the terms of degree deg
                mons = np.array(mons).T
                for i in range(dim):
                    slices.append(mons[i])
                slices = tuple(slices)
                slice_error = np.sum(np.abs(coeff[slices]))
                if slice_error + error > approx_tol:
                    if deg < coeff.shape[0]-1:
                        slices = tuple([slice(0,deg+1)]*dim)
                        coeff = coeff[slices]
                    break
                else:
                    error += slice_error
                    coeff[slices] = 0
                    deg-=1
                    if deg == 1:
                        slices = tuple([slice(0,2)]*dim)
                        coeff = coeff[slices]
                        break
        coeffs[num] = coeff

    return coeffs, all_triangular

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

def good_zeros_1d(zeros, imag_tol = 1.e-10):
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

def subdivision_solve_1d(f,a,b,interval_data,cheb_approx_tol=1.e-5,max_degree=128):
    """Finds the roots of a one-dimensional function using subdivision and chebyshev approximation.

    Parameters
    ----------
    f : function from R^n -> R
        The function to interpolate.
    a : numpy array
        The lower bound on the interval.
    b : numpy array
        The upper bound on the interval.
    cheb_approx_tol : float
        The bound of the sup norm error of the chebyshev approximation.
    max_degree : int
        The degree of the interpolation before subdividing.

    Returns
    -------
    coeffs : numpy array
        The coefficient of the chebyshev interpolating polynomial.
    """
    cur_deg = 2
    interval_data.print_progress()
    initial_approx = interval_approximate_1d(f,a,b,deg = cur_deg)
    while cur_deg<=max_degree:
        coeffsN = np.zeros(2*cur_deg+1)
        coeffsN[:cur_deg+1] = initial_approx
        coeffs2N = interval_approximate_1d(f,a,b,deg = 2*cur_deg)
        #Check if the approximation is good enough
        # if np.sum(np.abs(coeffs2N - coeffsN)) < cheb_approx_tol:
        if np.sum(np.abs(coeffs2N[cur_deg+1:])) < np.sum(np.abs(coeffs2N[:cur_deg+1]))*cheb_approx_tol:
            coeffs = coeffsN[:cur_deg+1]
            #const interval check
            if interval_data.check_interval(coeffs, cheb_approx_tol, a, b):
                return np.zeros([0])
            #Division is faster after degree 75
            if cur_deg > 75:
                interval_data.track_interval('Spectral', [a,b])
                return transform(good_zeros_1d(divCheb(coeffs)),a,b)
            else:
                interval_data.track_interval('Spectral', [a,b])
                return transform(good_zeros_1d(multCheb(np.trim_zeros(coeffs.copy(),trim='b'))),a,b)
        initial_approx = coeffs2N
        cur_deg*=2
    #Subdivide the interval and recursively call the function.
    div_length = (b-a)/2
    return np.hstack([subdivision_solve_1d(f,a,b-div_length,interval_data,max_degree=max_degree),\
                      subdivision_solve_1d(f,a+div_length,b,interval_data,max_degree=max_degree)])
