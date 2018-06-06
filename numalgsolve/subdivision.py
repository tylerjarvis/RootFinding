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
        zeros = np.unique(subdivide_solve_1d(funcs[0],a,b))
        #Finds the roots of each succesive function and checks which roots are common.
        for func in funcs[1:]:
            if len(zeros) == 0:
                break
            zeros2 = np.unique(subdivide_solve_1d(func,a,b))
            common = list()
            tol = 1.e-10
            for zero in zeros2:
                spot = np.where(np.abs(zeros-zero)<tol)
                if len(spot[0]) > 0:
                    common.append(zero)
            zeros = common
        return zeros
    else:
        if dim == 2:
            deg = 10
        elif dim == 3:
            deg = 7
        elif dim == 4:
            deg = 4
        elif dim == 5:
            deg = 2
        else:
            deg = 2
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
    def get_cheb_spot(k,n):
        return np.cos(np.pi*k/n)

    n = degs[0]

    if hasattr(f,"evaluate_grid"):
        #for polynomials, we can quickly evaluate all points in a grid
        #xyz does not contain points, but the nth column of xyz has the values needed
        #along the nth axis. The direct product of these values procuces the grid
        X = np.cos(np.arange(2*n)*np.pi/n)
        xyz = transform(np.column_stack([X]*dim), a, b)
        values = f.evaluate_grid(xyz)
    else:
        if dim == 2:
            X = np.cos(np.arange(2*n)*np.pi/n)
            y,x = np.meshgrid(X,X)
            cheb_spots = transform(np.vstack([x.flatten(),y.flatten()]).T,a,b)
            values = f(cheb_spots).reshape(2*n,2*n)
            #print(jit)
            #values = polyvalnd(cheb_spots,f.coeff).reshape(2*n,2*n)
        elif dim == 3:
            X = np.cos(np.arange(2*n)*np.pi/n)
            y,x,z = np.meshgrid(X,X,X)
            cheb_spots = transform(np.vstack([x.flatten(),y.flatten(),z.flatten()]).T,a,b)
            values = f(cheb_spots).reshape(2*n,2*n,2*n)

        else:
            values = np.zeros(2*degs)
            cheb_spots = list()
            for spot,val in np.ndenumerate(values):
                cheb_spots.append(transform([get_cheb_spot(spot[i],degs[i]) for i in range(dim)],a,b))
            vals = f(cheb_spots)
            i=0
            for spot,val in np.ndenumerate(values):
                values[spot] = vals[i]
                i+=1

    coeffs = np.real(fftn(values/np.product(degs)))

    for i in range(dim):
        idx0 = [slice(None)] * (dim)
        idx0[i] = 0
        idx00 = [slice(None)] * (dim)
        idx00[i] = degs[i]
        coeffs[idx0] = coeffs[idx0]/2
        coeffs[idx00] = coeffs[idx00]/2

    slices = list()
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
    subintervals = list()
    perms = interval_perms(list(), np.zeros(len(dimensions)), 0)
    diffs1 = ((b-a)*RAND)[dimensions]
    diffs2 = ((b-a)-(b-a)*RAND)[dimensions]
    for perm in perms:
        aTemp = a.copy()
        bTemp = b.copy()
        aTemp[dimensions] += (1-perm)*diffs1
        bTemp[dimensions] -= perm*diffs2
        subintervals.append(tuple([aTemp,bTemp]))
    return subintervals

def interval_perms(perms,perm,spot):
    """A helper function for get_subintervals. Finds the different ways we can split the dimensions.

    Called recursively.

    Parameters
    ----------
    perms : list
        The ways we have found so far.
    perm : numpy array
        The way we are currently dynamically computing.
    spot : int
        Where in perm we are changing.

    Returns
    -------
    subintervals : list
        Each element of the list is a tuple containing an a and b, the lower and upper bounds of the interval.
    """
    if spot == len(perm)-1:
        perms.append(perm.copy())
        perm[spot] = 1
        perms.append(perm.copy())
        return perms
    else:
        perms = interval_perms(perms,perm.copy(),spot+1)
        perm[spot] = 1
        perms = interval_perms(perms,perm.copy(),spot+1)
        return perms

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
    coeff2 = interval_approximate_nd(f,a,b,degs*2)
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
    dim = zeros.shape[1]
    good_zeros = zeros[np.where(np.sum(np.abs(zeros.imag) < imag_tol,axis = 1) == dim)[0]]
    good_zeros = good_zeros[np.where(np.sum(np.abs(good_zeros) <= 1,axis = 1) == dim)[0]]
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
    chebs = list()
    for func in funcs:
        coeff = full_cheb_approximate(func,a,b,deg,tol=tol)
        #Subdivides if needed.
        if coeff is None:
            intervals = get_subintervals(a,b,np.arange(dim))
            return np.vstack([subdivision_solve_nd(funcs,interval[0],interval[1],deg,tol=tol,tol2=tol2) \
                              for interval in intervals])
        coeff = trim_coeff(coeff,tol=tol, tol2=tol2)
        chebs.append(MultiCheb(coeff))
    zeros = np.array(division(chebs))
    if len(zeros) == 0:
        return np.zeros([0,dim])
    zeros = transform(good_zeros_nd(zeros),a,b)
    return zeros

def trim_coeff(coeff, tol=1.e-8, tol2=1.e-8):
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
    #Cuts down in the degree we are dividing by so the division matrix is stable.
    for spot in zip(*np.where(np.abs(coeff[0]) < tol2)):
        slices = list()
        slices.append(slice(0,None))
        for s in spot:
            slices.append(s)
        coeff[slices] = 0

    dim = len(coeff.shape)

    #Cuts out the high diagonals as much as possible to minimize polynomial degree.
    if abs(coeff[tuple([-1]*dim)]) < tol:
        coeff[tuple([-1]*dim)] = 0
        deg = np.sum(coeff.shape)-dim-1
        while True:
            mons = mon_combos_limited([0]*dim,deg,coeff.shape)
            slices = list()
            mons = np.array(mons).T
            for i in range(dim):
                slices.append(mons[i])
            if np.sum(np.abs(coeff[slices])) < tol:
                coeff[slices] = 0
            else:
                break
            deg -= 1

    return coeff

def mon_combos_limited(mon, numLeft, shape, spot = 0):
    '''Finds all the monomials of a given degree that fits in a given shape and returns them. Works recursively.

    Very similar to mon_combos, but only returns the monomials of the desired degree.

    Parameters
    --------
    mon: list
        A list of zeros, the length of which is the dimension of the desired monomials. Will change
        as the function searches recursively.
    numLeft : int
        The degree of the monomials desired. Will decrease as the function searches recursively.
    shape : tuple
        The limiting shape. The i'th spot of the mon can't be bigger than the i'th spot of the shape.
    spot : int
        The current position in the list the function is iterating through. Defaults to 0, but increases
        in each step of the recursion.

    Returns
    -----------
    answers : list
        A list of all the monomials.
    '''
    answers = list()
    if len(mon) == spot+1: #We are at the end of mon, no more recursion.
        if numLeft < shape[spot]:
            mon[spot] = numLeft
            answers.append(mon.copy())
        return answers
    if numLeft == 0: #Nothing else can be added.
        answers.append(mon.copy())
        return answers
    temp = mon.copy() #Quicker than copying every time inside the loop.
    for i in range(min(shape[spot],numLeft+1)): #Recursively add to mon further down.
        temp[spot] = i
        answers += mon_combos_limited(temp, numLeft-i, shape, spot+1)
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

def subdivide_solve_1d(f,a,b,cheb_approx_tol=1.e-10,max_degree=128):
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
    return np.hstack([subdivide_solve_1d(f,a,b-div_length,max_degree=max_degree),\
                      subdivide_solve_1d(f,a+div_length,b,max_degree=max_degree)])
