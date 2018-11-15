"""
The check functions are all functions that take in a coefficent matrix and run a quick check
to determine if there can ever be zeros on the unit box there. They are then put into the list
all_bound_check_functions in the order we want to run them (probably fastest first). These are
then all run to throw out intervals as possible.
"""

import numpy as np
from itertools import product
import itertools
from numalgsolve.polynomial import MultiCheb

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
#     print(max_curve * n * h**2/8)
    return min_corner > max_curve * n * h**2/8 + tol

def curvature_check(coeff, tol):
    poly = MultiCheb(coeff)
    a = np.array([-1.]*poly.dim)
    b = np.array([1.]*poly.dim)
    return not can_eliminate(poly, a, b, tol)
