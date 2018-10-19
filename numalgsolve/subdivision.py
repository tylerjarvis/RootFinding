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
        deg_dim = {2:10, 3:7, 4:4}
        if dim > 4:
            deg = 2
        else:
            deg = deg_dim[dim]

        return subdivision_solve_nd(funcs,a,b,deg)

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

    degs = np.array([deg]*dim)+1
    coeff = interval_approximate_nd(f,a,b,degs)
    coeff2 = interval_approximate_nd(f,a,b,degs*2) #Check against an approximation of degree twice as high, to make sure a reasonable approximation
    coeff2[slice_top(coeff)] -= coeff
    #print(np.sum(np.abs(coeff2)))
    clean_zeros_from_matrix(coeff2,1.e-16)
    #print(np.sum(np.abs(coeff2)))
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

def subdivision_solve_nd(funcs,a,b,deg,tol=1.e-8,tol2=1.e-8):
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
    print("Interval - ",a,b)
    dim = len(a)
    cheb_approx_list = []
    for func in funcs:
        coeff = full_cheb_approximate(func,a,b,deg,tol=tol)
        #Subdivides if needed.
        if coeff is None:
            intervals = get_subintervals(a,b,np.arange(dim))
            return np.vstack([subdivision_solve_nd(funcs,interval[0],interval[1],deg,tol=tol,tol2=tol2) \
                              for interval in intervals]) #subdivide and proceed recursively if too high degree
        coeff = trim_coeff(coeff,tol=tol, tol2=tol2)
        cheb_approx_list.append(MultiCheb(coeff)) #any zeros in the edges of the coeff matrix disappear here
    zeros = np.array(division(cheb_approx_list))
    if len(zeros) == 0:
        return np.zeros([0,dim])
    zeros = transform(good_zeros_nd(zeros),a,b)
    return zeros

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
