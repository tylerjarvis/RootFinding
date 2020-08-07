"""
This file contains functions that used to be part of IntervalChecks.py but are
not being used as of May 2020. This functions may be revamped in the future, but
in their current implementations they are not fast enough to be useful. If these
functions were carefully optimized, it's possible they could become useful, and
the ideas behind the checks could be used to develop better checks in the future.
A general description of each check is provided.
"""
import numpy as np
from itertools import product
import itertools
from yroots.polynomial import MultiCheb
from matplotlib import pyplot as plt
from yroots.polynomial import MultiCheb, Polynomial
from matplotlib import patches

"""
LINEAR CHECK
This is a fast and simple check. It compares the range of the linear terms on
the intervals to the sum of the absolute values of the other coefficients.
If min(linear-part) > other-coef-sum or max(linear-part) < -other-coef-sum,
there cannot be any roots.
Although this check is very efficient, we found that quadractic_check in IntervalChecks.py
removed most of the regions that linear_check removed plus some more. Experimentally,
it wasn't worth the time in 2D to run the linear check before the quadratic check because
quadratic check would throw out the region anyway.
**Because the nd-quadratic check is slow, it's quite possible that this check could be
useful in dimensions 3+ ***
Originally a subinterval check
"""
def linear_check(test_coeff, intervals, tol):
    """One of subinterval_checks

    Checks the max of the linear part of the approximation and compares to the sum of the other terms.

    Parameters
    ----------
    test_coeff : numpy array
        The coefficient matrix of the polynomial to check
    intervals : list
        A list of the intervals to check.
    tol: float
        The bound of the sup norm error of the chebyshev approximation.

    Returns
    -------
    mask : list
        A list of the results of each interval. False if the function is guarenteed to never be zero
        in the unit box, True otherwise
    """
    dim = test_coeff.ndim
    coeff_abs_sum = np.sum(np.abs(test_coeff))

    #Get the linear and constant terms
    idx = [0]*dim
    const = test_coeff[tuple(idx)]
    lin_coeff = np.zeros(dim)
    for cur_dim in range(dim):
        if test_coeff.shape[cur_dim] < 2:
            continue
        idx[cur_dim] = 1
        lin_coeff[cur_dim] = test_coeff[tuple(idx)]
        idx[cur_dim] = 0

    coeff_abs_sum -= np.sum(np.abs(lin_coeff))
    mask = []

    for i, interval in enumerate(intervals):

        corner_vals = []
        for ints in product(interval, repeat = dim):
            corner_vals.append(const + np.array([ints[i][i] for i in range(dim)])@lin_coeff)
        corner_vals = np.array(corner_vals)

        # check if corners have mixed signs
        if (corner_vals.min() < 0 < corner_vals.max()):
            mask.append(True)
            continue

        abs_smallest_corner = np.min(np.abs(corner_vals))
        if abs_smallest_corner > coeff_abs_sum + tol:
            # case: corner is far enough from 0
            mask.append(False)
        else:
            mask.append(True)

    return mask
"""
QUAD CHECK
Although it's confusingly named, the quad_check is NOT the same
as the quadratic_check. This function splits up the polynomial into
smaller chunks--polynomials with fewer terms, each of which is quadratic in one variable.
There cannot be a root if the minimum of the absolute value of the first chunk is greater
than the sum of the maximums of the absolute values of the other chunks.
Warning: this has quite a few issues. This check needs to be more mathematically
rigorous to be functional. There are coefficients that are ignored right now, which
must be added in if this check is to be airtight. The mathematical basis of how
extreme_val_3 is used on monomial multiples of quadratics should be explored
before using this check. For now, print statements are included to give some intuition
on how the chunks are made.
Originally an interval check
"""
def extreme_val3(test_coeff, maxx = True):
    ''' Finds the extreme value of test_coeff on -1 to 1, used by quad_check

    test_coeff is [a,b,c] and represents the funciton a + bx + c(2x^2 - 1).
    Basic calculus can be used to find the extreme values.

    Parameters
    ----------
    test_coeff : numpy array
        Array representing [a,b,c]
    maxx: bool
        If true returns the max of the absolute value of the funciton, otherwise returns
        the min of the absolute value of the function.
    Returns
    -------
    extreme_val3 : float
        The extreme value (max or min) of the absolute value of a + bx + c(2x^2 - 1).
    '''
    a,b,c = test_coeff
    #CAREFUL: There's a hard coded tolerance here...
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

def quad_check(test_coeff, tol):
    """One of interval_checks

    Like the constant term check, but splits the coefficient matrix into a one dimensional
    quadratics and uses the extreme values of those to get a better bound.

    Parameters
    ----------
    test_coeff : numpy array
        The coefficient matrix of the polynomial to check
    tol: float
        The bound of the sup norm error of the chebyshev approximation.

    Returns
    -------
    quad_check : bool
        False if the function is guarenteed to never be zero in the unit box, True otherwise
    """
    #The check fails if the test_coeff isn't at least quadratic
    if np.any(np.array(test_coeff.shape) < 3):
        return True
    dim = test_coeff.ndim
    slices = []
    slices.append(slice(0,3))
    slice_direc = 0
    for i in range(dim-1):
        slices.append(0)

    print('coef:',np.round(test_coeff,2),sep='\n')
    print('start coef:',test_coeff[tuple(slices)],sep='\n')
    #Get the min of the quadratic including the constant term
    start = extreme_val3(test_coeff[tuple(slices)], maxx = False)
    print('start coef min:',start,sep='\n')
    rest = 0

    #Get the max's of the other quadratics
    shape = list(test_coeff.shape)
    shape[slice_direc] = 1
    for spots in itertools.product(*[np.arange(i) for i in shape]):
        if sum(spots) > 0:
            for i in range(1, dim):
                slices[i] = spots[i]
            rest += extreme_val3(test_coeff[tuple(slices)])
            print('chunk coef:',test_coeff[tuple(slices)],sep='\n')
            print('chunk coef min:',extreme_val3(test_coeff[tuple(slices)]),'rest:',rest)

    #Tries the one-dimensional slices in other directions
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
            rest += extreme_val3(test_coeff[tuple(slices)])
            print('chunk coef:',test_coeff[tuple(slices)],sep='\n')
            print('chunk coef min:',extreme_val3(test_coeff[tuple(slices)]),'rest:',rest)

    if start > rest + tol:
        return False
    else:
        return True

def full_quad_check(test_coeff, tol):
    """One of interval_checks

    Runs the quad_check in each possible direction to get as much out of it as possible.

    Parameters
    ----------
    test_coeff : numpy array
        The coefficient matrix of the polynomial to check
    tol: float
        The bound of the sup norm error of the chebyshev approximation.

    Returns
    -------
    full_quad_check : bool
        False if the function is guarenteed to never be zero in the unit box, True otherwise
    """
    for perm in itertools.permutations(np.arange(test_coeff.ndim)):
        if not quad_check(test_coeff.transpose(perm), tol):
            return False
    return True

"""
CUBIC CHECK
Although it's confusingly named, the cubic_check is more similar to quad_check than
to any of the quadratic_check functions. This function splits up the polynomial into
smaller chunks--polynomials with fewer terms, each of which is cubic in one variable.
There cannot be a root if the minimum of the absolute value of the first chunk is greater
than the sum of the maximums of the absolute values of the other chunks.
Warning: this has quite a few issues. This check needs to be more mathematically
rigorous to be functional. There are coefficients that are ignored right now, which
must be added in if this check is to be airtight. The mathematical basis of how
extreme_val_3 is used on monomial multiples of quadratics should be explored
before using this check.
Originally an interval check
"""
def extreme_val4(test_coeff, maxx = True):
    ''' Finds the extreme value of test_coeff on -1 to 1, used by cubic_check

    test_coeff is [a,b,c,d] and represents the funciton a + bx + c(2x^2 - 1) + d*(4x^3 - 3x).
    Basic calculus can be used to find the extreme values.

    Parameters
    ----------
    test_coeff : numpy array
        Array representing [a,b,c,d]
    maxx: bool
        If true returns the absolute value of the max of the funciton, otherwise returns
        the absolute value of the min of the function.
    Returns
    -------
    extreme_val4 : float
        The extreme value (max or min) of the absolute value of a + bx + c(2x^2 - 1) + d*(4x^3 - 3x).
    '''
    a,b,c,d = test_coeff
    if np.abs(d) < 1.e-10:
        return extreme_val3([a,b,c], maxx = maxx)
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

def cubic_check(test_coeff, tol):
    """One of interval_checks

    Like the constant_term, but splits the coefficient matrix into a one dimensional
    cubics and uses the extreme values of those to get a better bound.

    Parameters
    ----------
    test_coeff : numpy array
        The coefficient matrix of the polynomial to check
    tol: float
        The bound of the sup norm error of the chebyshev approximation.

    Returns
    -------
    cubic_check : bool
        False if the function is guarenteed to never be zero in the unit box, True otherwise
    """
    #The check fails if the test_coeff isn't at least cubic
    if np.any(np.array(test_coeff.shape) < 4):
        return True
    dim = test_coeff.ndim
    slices = []
    slices.append(slice(0,4))
    slice_direc = 0
    for i in range(dim-1):
        slices.append(0)

    #Get the min of the cubic including the constant term
    start = extreme_val4(test_coeff[tuple(slices)], maxx = False)
    rest = 0

    #Get the max's of the other cubics
    shape = list(test_coeff.shape)
    shape[slice_direc] = 1
    for spots in itertools.product(*[np.arange(i) for i in shape]):
        if sum(spots) > 0:
            for i in range(1, dim):
                slices[i] = spots[i]
            rest += extreme_val4(test_coeff[tuple(slices)])

    #Tries the one-dimensional slices in other directions
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
            rest += extreme_val4(test_coeff[tuple(slices)])

    if start > rest + tol:
        return False
    else:
        return True

def full_cubic_check(test_coeff, tol):
    """One of interval_checks

    Runs the cubic_check in each possible direction to get as much out of it as possible.

    Parameters
    ----------
    test_coeff : numpy array
        The coefficient matrix of the polynomial to check
    tol: float
        The bound of the sup norm error of the chebyshev approximation.

    Returns
    -------
    full_cubic_check : bool
        False if the function is guarenteed to never be zero in the unit box, True otherwise
    """
    for perm in itertools.permutations(np.arange(test_coeff.ndim)):
        if not cubic_check(test_coeff.transpose(perm), tol):
            return False
    return True
"""
This check uses the curvature of the polynomial and interval arithmetic
to determine the range of the polynomial over the specified domain.
Like the other checks in this graveyard, it wasn't fast enough to use.
We've had issues with the first import statement in this check preventing
installation of yroots, so since this code is no longer used, the import
is currently commented out. If you decide to include this check, it's a
good idea to add it back into the unit test as well, since currently
it's not being tested due to this import error.
Originally an interval check
"""
# from mpmath import iv
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
        return "{}\n{}\n{}".format(self.iv,self.iv_prime,self.iv_lambda)
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

def can_eliminate(poly, a, b, tol):
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
    return min_corner > max_curve * n * h**2/8 + tol

def curvature_check(coeff, tol):
    poly = MultiCheb(coeff)
    a = np.array([-1.]*poly.dim)
    b = np.array([1.]*poly.dim)
    return not can_eliminate(poly, a, b, tol)
