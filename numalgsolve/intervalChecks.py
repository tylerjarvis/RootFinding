"""
The check functions are all functions that take in a coefficent matrix and run a quick check
to determine if there can ever be zeros on the unit box there. They are then put into the list
all_bound_check_functions in the order we want to run them (probably fastest first). These are
then all run to throw out intervals as possible.
"""

import numpy as np
from itertools import product
import itertools

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

def constant_term_check(test_coeff, tol):
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
    if np.abs(test_coeff.flatten()[0]) * 2 > test_sum + tol:
        return False
    else:
        return True

def quad_check(test_coeff, tol):
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

    if start > rest + tol:
        return False
    else:
        return True

def cubic_check(test_coeff, tol):
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

    if start > rest + tol:
        return False
    else:
        return True

def full_quad_check(test_coeff, tol):
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
        if not quad_check(test_coeff.transpose(perm), tol):
            return False
    return True

def full_cubic_check(test_coeff, tol):
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
        if not cubic_check(test_coeff.transpose(perm), tol):
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
    mask = []
    for interval in intervals:
        test_coeff = test_coeff_in.copy()
        
        a,b = interval
        spot = [0]*len(a)
        neg_most_corner = test_coeff[tuple(spot)]
        test_coeff[tuple(spot)] = 0
        for dim in range(len(a)):
            spot[dim] = 1
            neg_most_corner += a[dim]*test_coeff[tuple(spot)]
            spot[dim] = 0
        
        lin_min = neg_most_corner
        for dim in range(len(a)):
            spot[dim] = 1
            if np.sign(test_coeff[tuple(spot)])*np.sign(neg_most_corner) < 0:
                lin_min += (b[dim] - a[dim]) * test_coeff[tuple(spot)]
            test_coeff[tuple(spot)] = 0
            spot[dim] = 0
        
        if np.sign(lin_min)*np.sign(neg_most_corner) < 0:
            mask.append(True)
        elif np.sum(np.abs(test_coeff)) >= np.abs(neg_most_corner):
            mask.append(True)
        else:
            mask.append(False)
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

def TylersFunction(coeff):
    poly = MultiCheb(coeff)
    a = np.array([-1.]*poly.dim)
    b = np.array([1.]*poly.dim)
    return not can_eliminate(poly, a, b)
