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
from yroots.utils import clean_zeros_from_matrix, slice_top, MacaulayError, get_var_list
from yroots.polynomial import MultiCheb
from yroots.IntervalChecks import IntervalData
from itertools import product
from matplotlib import pyplot as plt
import itertools
import time

def solve(funcs, a, b, plot = False, plot_intervals = False, polish = False):
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

    If finding roots of a univariate function, `funcs` does not need to be a list,
    and `a` and `b` can be floats instead of arrays.

    returns
    -------
    roots : numpy array
        The common roots of the polynomials. Each row is a root.
    '''
    interval_data = IntervalData(a,b)

    if not isinstance(funcs,list):
        funcs = [funcs]
        dim = 1
    elif not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        dim = 1
    else:
        dim = len(a)
    if dim == 1:
        #one dimensional case
        zeros = subdivision_solve_1d(funcs[0],a,b)
        if plot:
            x = np.linspace(a,b,1000)
            for f in funcs:
                plt.plot(x,f(x),color='k')
            plt.plot(np.real(zeros),np.zeros(len(zeros)),'o',color = 'none',markeredgecolor='r')
            plt.show()
        return zeros
    else:
        #multidimensional case

        #make a and b the right type
        a = np.float64(a)
        b = np.float64(b)
        #choose an appropriate max degree for the given dimension
        deg_dim = {2:9, 3:8, 4:3}
        if dim > 4:
            deg = 2
        else:
            deg = deg_dim[dim]

        #Output the interval percentages
        zeros = subdivision_solve_nd(funcs,a,b,deg,interval_data,polish=polish)

        print("\rPercent Finished: 100%       ")
        interval_data.print_results()
        #Plot what happened
        if plot and dim == 2:
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

def interval_approximate_nd(f,a,b,degs,return_bools=False):
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
        whether to return bools which indicate if a funtion changes sign or not

    Returns
    -------
    coeffs : numpy array
        The coefficient of the chebyshev interpolating polynomial.
    change_sign: numpy array
        list of which subintervals change sign
    """
    if len(a)!=len(b):
        raise ValueError("Interval dimensions must be the same!")

    dim = len(a)
    deg = degs[0]

    if hasattr(f,"evaluate_grid"):
        cheb_values = np.cos(np.arange(deg+1)*np.pi/deg)
        xyz = transform(np.column_stack([cheb_values]*dim), a, b)
        values_block = f.evaluate_grid(xyz)
    else:
        cheb_values = np.cos(np.arange(deg+1)*np.pi/deg)
        cheb_grids = np.meshgrid(*([cheb_values]*dim), indexing='ij')

        flatten = lambda x: x.flatten()
        cheb_points = transform(np.column_stack(map(flatten, cheb_grids)), a, b)
        cheb_points = [cheb_points[:,i] for i in range(dim)]
        values_block = f(*cheb_points).reshape(*([deg+1]*dim))

    slices = []
    for i in range(dim):
        slices.append(slice(0,degs[i]+1))

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
    coeffs = np.real(fftn(values/np.product(degs)))

    for i in range(dim):
        #construct slices for the first and degs[i] entry in each dimension
        idx0 = [slice(None)] * dim
        idx0[i] = 0

        idx_deg = [slice(None)] * dim
        idx_deg[i] = degs[i]

        #halve the coefficients in each slice
        coeffs[tuple(idx0)] /= 2
        coeffs[tuple(idx_deg)] /= 2

    if return_bools:
        return coeffs[tuple(slices)], change_sign
    else:
        return coeffs[tuple(slices)]

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
        scaled_subintervals = get_subintervals(-np.ones_like(a),np.ones_like(a),dimensions,None,None,None,approx_tol,False)
        return interval_data.check_subintervals(subintervals, scaled_subintervals, polys, change_sign, approx_tol)
    else:
        return subintervals

def full_cheb_approximate(f,a,b,deg,tol=1.e-8,good_degs=None):
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
    bools: numpy array
        (2^n, 1) array of bools corresponding to which subintervals the function changes sign in
    """
    dim = len(a)
    degs = np.array([deg]*dim)
    #We know what degree we want
    if good_degs is not None:
        coeff, bools = interval_approximate_nd(f,a,b,good_degs,return_bools=True)
        clean_zeros_from_matrix(coeff,1.e-16)
        return coeff, bools
    #Try degree deg and see if it's good enough
    coeff = interval_approximate_nd(f,a,b,degs)
    coeff2, bools = interval_approximate_nd(f,a,b,degs*2,return_bools=True)
    coeff2[slice_top(coeff)] -= coeff
    clean_zeros_from_matrix(coeff2,1.e-16)
    if np.sum(np.abs(coeff2)) > tol:
        return None, None
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

    Returns
    -------
    good_zeros : numpy array
        The real zero in [-1,1] of the input zeros.
    """
    good_zeros = zeros[np.all(np.abs(zeros.imag) < imag_tol,axis = 1)]
    good_zeros = good_zeros[np.all(np.abs(good_zeros) <= 1 + real_tol,axis = 1)]
    return good_zeros.real

def subdivision_solve_nd(funcs,a,b,deg,interval_data,approx_tol=1.e-4,solve_tol=1.e-8, polish=False, good_degs=None):
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
    cheb_approx_list = []
    try:
        interval_data.print_progress()
        dim = len(a)

        for func_num, func in enumerate(funcs):
            if good_degs is not None:
                coeff, change_sign = full_cheb_approximate(func,a,b,deg,approx_tol,good_degs[func_num])
            else:
                coeff, change_sign = full_cheb_approximate(func,a,b,deg,approx_tol)
            
            #Subdivides if a bad approximation
            if coeff is None:
                intervals = get_subintervals(a,b,np.arange(dim),None,None,None,approx_tol,None)
                return np.vstack([subdivision_solve_nd(funcs,interval[0],interval[1],deg,interval_data,\
                                                       approx_tol,solve_tol,polish) for interval in intervals])
            else:
                #if the function changes sign on at least one subinterval, skip the checks
                if np.any(change_sign):
                    cheb_approx_list.append(coeff)
                    continue
                #Run checks to try and throw out the interval
                if interval_data.check_intervals(coeff, approx_tol, a, b):
                    return np.zeros([0,dim])

                cheb_approx_list.append(coeff)
        
        #Make the system stable to solve
        coeffs, divisor_var = trim_coeffs(cheb_approx_list, approx_tol = approx_tol)
        
        #Check if everything is linear
        if np.all(np.array([coeff.shape[0] for coeff in coeffs]) == 2):
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
            if np.linalg.matrix_rank(A) < dim:
                #FIX THIS
                raise ValueError("I have no idea what to do here")
            zero = np.linalg.solve(A,-B)
            interval_data.track_interval("Base Case", [a,b])
            if polish:
                polish_tol = (b[0]-a[0])/100
                return polish_zeros(transform(good_zeros_nd(zero.reshape([1,dim])),a,b), funcs,\
                                    interval_data,tol=polish_tol)
            else:
                return transform(good_zeros_nd(zero.reshape([1,dim])),a,b)
        elif np.any(np.array([coeff.shape[0] for coeff in coeffs]) == 2):
            #Subdivide but run some checks on the intervals first
            intervals = get_subintervals(a,b,np.arange(dim),interval_data,cheb_approx_list,change_sign,\
                                                 approx_tol,check_subintervals=True)
            if len(intervals) == 0:
                return np.zeros([0,dim])
            else:
                good_degs = [np.array(coeff.shape) - 1 for coeff in coeffs]
                return np.vstack([subdivision_solve_nd(funcs,interval[0],interval[1],deg,interval_data,\
                                                       approx_tol,solve_tol,polish,good_degs) for interval in intervals])
        
        if np.any([coeff.shape[0] > 5 for coeff in coeffs]):
            divisor_var = -1
        if divisor_var < 0:
            #Subdivide but run some checks on the intervals first
            intervals = get_subintervals(a,b,np.arange(dim),interval_data,cheb_approx_list,\
                                                 change_sign,approx_tol,check_subintervals=True)
            if len(intervals) == 0:
                return np.zeros([0,dim])
            else:
                good_degs = [np.array(coeff.shape) - 1 for coeff in coeffs]
                return np.vstack([subdivision_solve_nd(funcs,interval[0],interval[1],deg,interval_data,\
                                                       approx_tol,solve_tol,polish,good_degs) for interval in intervals])
        
        polys = [MultiCheb(coeff, lead_term = [coeff.shape[0]-1], clean_zeros = False) for coeff in coeffs]
        zeros = np.array(division(polys,divisor_var,solve_tol))
        interval_data.track_interval("Division", [a,b])
        if len(zeros) == 0:
            return np.zeros([0,dim])
        if polish:
            polish_tol = (b[0]-a[0])/100
            return polish_zeros(transform(good_zeros_nd(zeros),a,b), funcs, interval_data, polish_tol)
        else:
            return transform(good_zeros_nd(zeros),a,b)

    except np.linalg.LinAlgError as e:
        #Try in other directions
        divisor_var += 1
        while divisor_var < dim:
            if not good_direc(coeffs,divisor_var): #TEMP
                break
            try:
                zeros = np.array(division(polys, divisor_var, solve_tol))
                interval_data.track_interval("Division", [a,b])
                if len(zeros) == 0:
                    return np.zeros([0,dim])
                if polish:
                    polish_tol = (b[0]-a[0])/100
                    return polish_zeros(transform(good_zeros_nd(zeros),a,b),funcs,interval_data,polish_tol)
                else:
                    return transform(good_zeros_nd(zeros),a,b)
            except np.linalg.LinAlgError as e:
                divisor_var += 1
        #Subdivide but run some checks on the intervals first
        intervals = get_subintervals(a,b,np.arange(dim),interval_data,cheb_approx_list,change_sign,\
                                             approx_tol,check_subintervals=True)
        if len(intervals) == 0:
            return np.zeros([0,dim])
        else:
            good_degs = [np.array(poly.coeff.shape) - 1 for poly in polys]
            return np.vstack([subdivision_solve_nd(funcs,interval[0],interval[1],deg,interval_data,\
                                                   approx_tol,solve_tol,polish,good_degs) for interval in intervals])

def good_direc(coeffs, dim, tol=1e-6):
    slices = []
    for i in range(coeffs[0].ndim):
        if i == dim:
            slices.append(0)
        else:
            slices.append(slice(None))
    vals = [coeff[slices] for coeff in coeffs]
    degs = [val.shape[0] for val in vals]
    
    min_vals = np.zeros([len(vals),*vals[np.argmax(degs)].shape])

    for num, val in enumerate(vals):
        deg = degs[num]
        slices = [num]+[slice(0,deg) for i in range(val.ndim)]
        min_vals[slices] = val

    min_vals[min_vals==0] = 1
    if np.any(np.min(np.abs(min_vals),axis=0) < tol):
        return False
    return True    

def polish_zeros(zeros, funcs, interval_data, tol=1.e-2):
    if len(zeros) == 0:
        return zeros
    dim = zeros.shape[1]
    polished_zeros = []
    interval_data = IntervalData(np.array([0]),np.array([0]))
    interval_data.polishing = True
    
    for zero in zeros:
        a = np.array(zero) - tol
        b = np.array(zero) + 1.1*tol #Keep the root away from 0
        polished_zero = subdivision_solve_nd(funcs,a,b,5,interval_data,approx_tol=1.e-7,\
                                                 solve_tol=1.e-8,polish=False)
        polished_zeros.append(polished_zero)
    return np.vstack(polished_zeros)

def trim_coeffs(coeffs, approx_tol):
    """Trim the coefficient matrices so they are stable and choose a direction to divide in.

    Parameters
    ----------
    coeffs : list
        The coefficient matrices of the Chebyshev polynomials we are solving.

    Returns
    -------
    polys : list
        The reduced degree Chebyshev polynomials
    divisor_var : int
        What direction to do the division in to be stable. -1 means we should subdivide.
    """
    all_triangular = True
    for num, coeff in enumerate(coeffs):
        error = 0.
        dim = coeff.ndim
        deg = np.sum(coeff.shape) - dim
        initial_mons = []
        for deg0 in range(coeff.shape[0], deg+1):
            initial_mons += mon_combos_limited([0]*dim,deg0,coeff.shape)
        mons = np.array(initial_mons).T
        slices = [mons[i] for i in range(dim)]
        slice_error = np.sum(np.abs(coeff[slices]))
        if slice_error + error > approx_tol:
            all_triangular = False
        else:
            coeff[slices] = 0
            deg = coeff.shape[0]-1
            while True:
                mons = mon_combos_limited([0]*dim,deg,coeff.shape)
                slices = [] #becomes the indices of the terms of degree deg
                mons = np.array(mons).T
                for i in range(dim):
                    slices.append(mons[i])
                slice_error = np.sum(np.abs(coeff[slices]))
                if slice_error + error > approx_tol:
                    if deg < coeff.shape[0]-1:
                        slices = [slice(0,deg+1)]*dim
                        coeff = coeff[slices]
                    break
                else:
                    error += slice_error
                    coeff[slices] = 0
                    deg-=1
                    if deg == 1:
                        slices = [slice(0,2)]*dim
                        coeff = coeff[slices]
                        break
        coeffs[num] = coeff
    
    if not all_triangular:
        return coeffs, -1
    else:
        for divisor_var in range(coeffs[0].ndim):
            if good_direc(coeffs,divisor_var):
                return coeffs, divisor_var
        return coeffs, -1

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
                return transform(good_zeros_1d(divCheb(coeffs)),a,b)
            else:
                return transform(good_zeros_1d(multCheb(np.trim_zeros(coeffs.copy(),trim='b'))),a,b)
        initial_approx = coeffs2N
        cur_deg*=2
    #Subdivide the interval and recursively call the function.
    div_length = (b-a)/2
    return np.hstack([subdivision_solve_1d(f,a,b-div_length,max_degree=max_degree),\
                      subdivision_solve_1d(f,a+div_length,b,max_degree=max_degree)])
