import numpy as np
import itertools
from scipy import linalg as la
from math import fabs

def get_fixed_vars(dim):
    """Used in quadratic_check_nd to iterate through the boundaries of the domain.

    Parameters
    ----------
    dim : int
        The dimension of the domain/system.

    Returns
    -------
    list of tuples
        A list of tuples indicating which variables to fix in each iteration,
        starting at fixing dim-1 of them and ending with fixing 1 of them. This
        intentionally excludes combinations that correspond to the corners of the
        domain and the interior extremum.
    """
    return list(itertools.chain.from_iterable(itertools.combinations(range(dim), r)\
                                             for r in range(dim-1,0,-1)))

def quadratic_check(test_coeff, tol, nd_check=False):
    if test_coeff.ndim == 2 and not nd_check:
        return quadratic_check_2D(test_coeff, tol)
    elif test_coeff.ndim == 3 and not nd_check:
        return quadratic_check_3D(test_coeff, tol)
    else:
        return quadratic_check_nd(test_coeff, tol)
    #return quadratic_check_nd(test_coeff, tol)

def quadratic_check_2D(test_coeff, tol):
    """One of subinterval_checks

    Finds the min of the absolute value of the quadratic part, and compares to the sum of the
    rest of the terms. There can't be a root if min(extreme_values) > other_sum	or if
    max(extreme_values) < -other_sum. We can short circuit and finish
    faster as soon as we find one value that is < other_sum and one value that > -other_sum.

    Parameters
    ----------
    test_coeff : numpy array
        The coefficient matrix of the polynomial to check
    tol: float
        The bound of the sup norm error of the chebyshev approximation.

    Returns
    -------
    True if the function is guaranteed to never be zero in the interval. False otherwise.
    """
    if test_coeff.ndim != 2:
        return False

    #Define the interval
    interval = [[-1, -1], [1, 1]]

    #Get the coefficients of the quadratic part
    #Need to account for when certain coefs are zero.
    #Padding is slow, so check the shape instead.
    c = [0]*6
    shape = test_coeff.shape
    c[0] = test_coeff[0,0]
    if shape[0] > 1:
        c[1] = test_coeff[1,0]
    if shape[1] > 1:
        c[2] = test_coeff[0,1]
    if shape[0] > 2:
        c[3] = test_coeff[2,0]
    if shape[0] > 1 and shape[1] > 1:
        c[4] = test_coeff[1,1]
    if shape[1] > 2:
        c[5] = test_coeff[0,2]

    # The sum of the absolute values of the other coefs
    # Note: Overhead for instantiating a NumPy array is too costly for
    #  small arrays, so the second sum here is faster than using numpy
    other_sum = np.sum(np.abs(test_coeff)) - sum([fabs(coeff) for coeff in c]) + tol

    # Function for evaluating c0 + c1 T_1(x) + c2 T_1(y) +c3 T_2(x) + c4 T_1(x)T_1(y) + c5 T_2(y)
    # Use the Horner form because it is much faster, also do any repeated computatons in advance
    k0 = c[0]-c[3]-c[5]
    k3 = 2*c[3]
    k5 = 2*c[5]
    def eval_func(x,y):
        return k0 + (c[1] + k3 * x + c[4] * y) * x  + (c[2] + k5 * y) * y

    #The interior min
    #Comes from solving dx, dy = 0
    #Dx: 4c3x +  c4y = -c1    Matrix inverse is  [4c5  -c4]
    #Dy:  c4x + 4c5y = -c2                       [-c4  4c3]
    # This computation is the same for all subintevals, so do it first
    det = 16 * c[3] * c[5] - c[4]**2
    if det != 0:
        int_x = (c[2] * c[4] - 4 * c[1] * c[5]) / det
        int_y = (c[1] * c[4] - 4 * c[2] * c[3]) / det
    else:                      # det is zero,
        int_x = np.inf
        int_y = np.inf
        
    min_satisfied, max_satisfied = False,False
    #Check all the corners
    eval = eval_func(interval[0][0], interval[0][1])
    min_satisfied = min_satisfied or eval < other_sum
    max_satisfied = max_satisfied or eval > -other_sum
    if min_satisfied and max_satisfied:
        return False

    eval = eval_func(interval[1][0], interval[0][1])
    min_satisfied = min_satisfied or eval < other_sum
    max_satisfied = max_satisfied or eval > -other_sum
    if min_satisfied and max_satisfied:
        return False

    eval = eval_func(interval[0][0], interval[1][1])
    min_satisfied = min_satisfied or eval < other_sum
    max_satisfied = max_satisfied or eval > -other_sum
    if min_satisfied and max_satisfied:
        return False

    eval = eval_func(interval[1][0], interval[1][1])
    min_satisfied = min_satisfied or eval < other_sum
    max_satisfied = max_satisfied or eval > -other_sum
    if min_satisfied and max_satisfied:
        return False

    #Check the x constant boundaries
    #The partial with respect to y is zero
    #Dy:  c4x + 4c5y = -c2 =>   y = (-c2-c4x)/(4c5)
    if c[5] != 0:
        cc5 = 4 * c[5]
        x = interval[0][0]
        y = -(c[2] + c[4]*x)/cc5
        if interval[0][1] < y < interval[1][1]:
            eval = eval_func(x,y)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                return False
        x = interval[1][0]
        y = -(c[2] + c[4]*x)/cc5
        if interval[0][1] < y < interval[1][1]:
            eval = eval_func(x,y)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                return False

    #Check the y constant boundaries
    #The partial with respect to x is zero
    #Dx: 4c3x +  c4y = -c1  =>  x = (-c1-c4y)/(4c3)
    if c[3] != 0:
        cc3 = 4*c[3]
        y = interval[0][1]
        x = -(c[1] + c[4]*y)/cc3
        if interval[0][0] < x < interval[1][0]:
            eval = eval_func(x,y)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                return False

        y = interval[1][1]
        x = -(c[1] + c[4]*y)/cc3
        if interval[0][0] < x < interval[1][0]:
            eval = eval_func(x,y)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                return False

    #Check the interior value
    if interval[0][0] < int_x < interval[1][0] and interval[0][1] < int_y < interval[1][1]:
        eval = eval_func(int_x,int_y)
        min_satisfied = min_satisfied or eval < other_sum
        max_satisfied = max_satisfied or eval > -other_sum
        if min_satisfied and max_satisfied:
            return False

    #No root possible
    return True

def quadratic_check_3D(test_coeff, tol):
    """One of subinterval_checks

    Finds the min of the absolute value of the quadratic part, and compares to the sum of the
    rest of the terms.  There can't be a root if min(extreme_values) > other_sum	or if
    max(extreme_values) < -other_sum. We can short circuit and finish
    faster as soon as we find one value that is < other_sum and one value that > -other_sum.

    Parameters
    ----------
    test_coeff : numpy array
        The coefficient matrix of the polynomial to check
    tol: float
        The bound of the sup norm error of the chebyshev approximation.

    Returns
    -------
    mask : list
        A list of the results of each interval. False if the function is guarenteed to never be zero
        in the unit box, True otherwise
    """
    if test_coeff.ndim != 3:
        return False

    #Padding is slow, so check the shape instead.
    c = [0]*10
    shape = test_coeff.shape
    c[0] = test_coeff[0,0,0]
    if shape[0] > 1:
        c[1] = test_coeff[1,0,0]
    if shape[1] > 1:
        c[2] = test_coeff[0,1,0]
    if shape[2] > 1:
        c[3] = test_coeff[0,0,1]
    if shape[0] > 1 and shape[1] > 1:
        c[4] = test_coeff[1,1,0]
    if shape[0] > 1 and shape[2] > 1:
        c[5] = test_coeff[1,0,1]
    if shape[1] > 1 and shape[2] > 1:
        c[6] = test_coeff[0,1,1]
    if shape[0] > 2:
        c[7] = test_coeff[2,0,0]
    if shape[1] > 2:
        c[8] = test_coeff[0,2,0]
    if shape[2] > 2:
        c[9] = test_coeff[0,0,2]

    #The sum of the absolute values of everything else
    other_sum = np.sum(np.abs(test_coeff)) - sum([fabs(coeff) for coeff in c]) + tol

    #function for evaluating c0 + c1x + c2y +c3z + c4xy + c5xz + c6yz + c7T_2(x) + c8T_2(y) + c9T_2(z)
    # Use the Horner form because it is much faster, also do any repeated computatons in advance
    k0 = c[0]-c[7]-c[8]-c[9]
    k7 = 2*c[7]
    k8 = 2*c[8]
    k9 = 2*c[9]
    def eval_func(x,y,z):
        return k0 + (c[1] + k7 * x + c[4] * y + c[5] * z) * x + \
                    (c[2] + k8 * y + c[6] * z) * y + \
                    (c[3] + k9 * z) * z

    #The interior min
    #Comes from solving dx, dy, dz = 0
    #Dx: 4c7x +  c4y +  c5z = -c1    Matrix inverse is  [(16c8c9-c6^2) -(4c4c9-c5c6)  (c4c6-4c5c8)]
    #Dy:  c4x + 4c8y +  c6z = -c2                       [-(4c4c9-c5c6) (16c7c9-c5^2) -(4c6c7-c4c5)]
    #Dz:  c5x +  c6y + 4c9z = -c3                       [(c4c6-4c5c8)  -(4c6c7-c4c5) (16c7c8-c4^2)]
    #These computations are the same for all subintevals, so do them first
    kk7 = 2*k7 #4c7
    kk8 = 2*k8 #4c8
    kk9 = 2*k9 #4c9
    fix_x_det = kk8*kk9-c[6]**2
    fix_y_det = kk7*kk9-c[5]**2
    fix_z_det = kk7*kk8-c[4]**2
    minor_1_2 = kk9*c[4]-c[5]*c[6]
    minor_1_3 = c[4]*c[6]-kk8*c[5]
    minor_2_3 = kk7*c[6]-c[4]*c[5]
    det = 4*c[7]*fix_x_det - c[4]*minor_1_2 + c[5]*minor_1_3
    if det != 0:
        int_x = (c[1]*-fix_x_det + c[2]*minor_1_2  + c[3]*-minor_1_3)/det
        int_y = (c[1]*minor_1_2  + c[2]*-fix_y_det + c[3]*minor_2_3)/det
        int_z = (c[1]*-minor_1_3  + c[2]*minor_2_3  + c[3]*-fix_z_det)/det
    else:
        int_x = np.inf
        int_y = np.inf
        int_z = np.inf

    interval = [[-1, -1, -1], [1, 1, 1]]
    #easier names for each value...
    x0 = interval[0][0]
    x1 = interval[1][0]
    y0 = interval[0][1]
    y1 = interval[1][1]
    z0 = interval[0][2]
    z1 = interval[1][2]

    min_satisfied, max_satisfied = False,False
    #Check all the corners
    eval = eval_func(x0, y0, z0)
    min_satisfied = min_satisfied or eval < other_sum
    max_satisfied = max_satisfied or eval > -other_sum
    if min_satisfied and max_satisfied:
        return False
    eval = eval_func(x1, y0, z0)
    min_satisfied = min_satisfied or eval < other_sum
    max_satisfied = max_satisfied or eval > -other_sum
    if min_satisfied and max_satisfied:
        return False
    eval = eval_func(x0, y1, z0)
    min_satisfied = min_satisfied or eval < other_sum
    max_satisfied = max_satisfied or eval > -other_sum
    if min_satisfied and max_satisfied:
        return False
    eval = eval_func(x0, y0, z1)
    min_satisfied = min_satisfied or eval < other_sum
    max_satisfied = max_satisfied or eval > -other_sum
    if min_satisfied and max_satisfied:
        return False
    eval = eval_func(x1, y1, z0)
    min_satisfied = min_satisfied or eval < other_sum
    max_satisfied = max_satisfied or eval > -other_sum
    if min_satisfied and max_satisfied:
        return False
    eval = eval_func(x1, y0, z1)
    min_satisfied = min_satisfied or eval < other_sum
    max_satisfied = max_satisfied or eval > -other_sum
    if min_satisfied and max_satisfied:
        return False
    eval = eval_func(x0, y1, z1)
    min_satisfied = min_satisfied or eval < other_sum
    max_satisfied = max_satisfied or eval > -other_sum
    if min_satisfied and max_satisfied:
        return False
    eval = eval_func(x1, y1, z1)
    min_satisfied = min_satisfied or eval < other_sum
    max_satisfied = max_satisfied or eval > -other_sum
    if min_satisfied and max_satisfied:
        return False
    #Adds the x and y constant boundaries
    #The partial with respect to z is zero
    #Dz:  c5x +  c6y + 4c9z = -c3   => z=(-c3-c5x-c6y)/(4c9)
    if c[9] != 0:
        c5x0_c3 = c[5]*x0 + c[3]
        c6y0 = c[6]*y0
        z = -(c5x0_c3+c6y0)/kk9
        if z0 < z < z1:
            eval = eval_func(x0,y0,z)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                return False
        c6y1 = c[6]*y1
        z = -(c5x0_c3+c6y1)/kk9
        if z0 < z < z1:
            eval = eval_func(x0,y1,z)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                return False
        c5x1_c3 = c[5]*x1 + c[3]
        z = -(c5x1_c3+c6y0)/kk9
        if z0 < z < z1:
            eval = eval_func(x1,y0,z)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                return False
        z = -(c5x1_c3+c6y1)/kk9
        if z0 < z < z1:
            eval = eval_func(x1,y1,z)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                return False

    #Adds the x and z constant boundaries
    #The partial with respect to y is zero
    #Dy:  c4x + 4c8y + c6z = -c2   => y=(-c2-c4x-c6z)/(4c8)
    if c[8] != 0:
        c6z0 = c[6]*z0
        c2_c4x0 = c[2]+c[4]*x0
        y = -(c2_c4x0+c6z0)/kk8
        if y0 < y < y1:
            eval = eval_func(x0,y,z0)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                return False
        c6z1 = c[6]*z1
        y = -(c2_c4x0+c6z1)/kk8
        if y0 < y < y1:
            eval = eval_func(x0,y,z1)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                return False
        c2_c4x1 = c[2]+c[4]*x1
        y = -(c2_c4x1+c6z0)/kk8
        if y0 < y < y1:
            eval = eval_func(x1,y,z0)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                return False
        y = -(c2_c4x1+c6z1)/kk8
        if y0 < y < y1:
            eval = eval_func(x1,y,z1)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                return False

    #Adds the y and z constant boundaries
    #The partial with respect to x is zero
    #Dx: 4c7x +  c4y +  c5z = -c1   => x=(-c1-c4y-c5z)/(4c7)
    if c[7] != 0:
        c1_c4y0 = c[1]+c[4]*y0
        c5z0 = c[5]*z0
        x = -(c1_c4y0+c5z0)/kk7
        if x0 < x < x1:
            eval = eval_func(x,y0,z0)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                return False
        c5z1 = c[5]*z1
        x = -(c1_c4y0+c5z1)/kk7
        if x0 < x < x1:
            eval = eval_func(x,y0,z1)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                return False
        c1_c4y1 = c[1]+c[4]*y1
        x = -(c1_c4y1+c5z0)/kk7
        if x0 < x < x1:
            eval = eval_func(x,y1,z0)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                return False
        x = -(c1_c4y1+c5z1)/kk7
        if x0 < x < x1:
            eval = eval_func(x,y1,z1)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                return False

    #Add the x constant boundaries
    #The partials with respect to y and z are zero
    #Dy:  4c8y +  c6z = -c2 - c4x    Matrix inverse is [4c9  -c6]
    #Dz:   c6y + 4c9z = -c3 - c5x                      [-c6  4c8]
    if fix_x_det != 0:
        c2_c4x0 = c[2]+c[4]*x0
        c3_c5x0 = c[3]+c[5]*x0
        y = (-kk9*c2_c4x0 +   c[6]*c3_c5x0)/fix_x_det
        z = (c[6]*c2_c4x0 -    kk8*c3_c5x0)/fix_x_det
        if y0 < y < y1 and z0 < z < z1:
            eval = eval_func(x0,y,z)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                return False
        c2_c4x1 = c[2]+c[4]*x1
        c3_c5x1 = c[3]+c[5]*x1
        y = (-kk9*c2_c4x1 +   c[6]*c3_c5x1)/fix_x_det
        z = (c[6]*c2_c4x1 -    kk8*c3_c5x1)/fix_x_det
        if y0 < y < y1 and z0 < z < z1:
            eval = eval_func(x1,y,z)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                return False

    #Add the y constant boundaries
    #The partials with respect to x and z are zero
    #Dx: 4c7x +  c5z = -c1 - c40    Matrix inverse is [4c9  -c5]
    #Dz:  c5x + 4c9z = -c3 - c6y                      [-c5  4c7]
    if fix_y_det != 0:
        c1_c4y0 = c[1]+c[4]*y0
        c3_c6y0 = c[3]+c[6]*y0
        x = (-kk9*c1_c4y0 +   c[5]*c3_c6y0)/fix_y_det
        z = (c[5]*c1_c4y0 -    kk7*c3_c6y0)/fix_y_det
        if x0 < x < x1 and z0 < z < z1:
            eval = eval_func(x,y0,z)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                return False
        c1_c4y1 = c[1]+c[4]*y1
        c3_c6y1 = c[3]+c[6]*y1
        x = (-kk9*c1_c4y1 +   c[5]*c3_c6y1)/fix_y_det
        z = (c[5]*c1_c4y1 -    kk7*c3_c6y1)/fix_y_det
        if x0 < x < x1 and z0 < z < z1:
            eval = eval_func(x,y1,z)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                return False

    #Add the z constant boundaries
    #The partials with respect to x and y are zero
    #Dx: 4c7x +  c4y  = -c1 - c5z    Matrix inverse is [4c8  -c4]
    #Dy:  c4x + 4c8y  = -c2 - c6z                      [-c4  4c7]
    if fix_z_det != 0:
        c1_c5z0 = c[1]+c[5]*z0
        c2_c6z0 = c[2]+c[6]*z0
        x = (-kk8*c1_c5z0 +   c[4]*c2_c6z0)/fix_z_det
        y = (c[4]*c1_c5z0 -    kk7*c2_c6z0)/fix_z_det
        if x0 < x < x1 and y0 < y < y1:
            eval = eval_func(x,y,z0)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                return False
        c1_c5z1 = c[1]+c[5]*z1
        c2_c6z1 = c[2]+c[6]*z1
        x = (-kk8*c1_c5z1 +   c[4]*c2_c6z1)/fix_z_det
        y = (c[4]*c1_c5z1 -    kk7*c2_c6z1)/fix_z_det
        if x0 < x < x1 and y0 < y < y1:
            eval = eval_func(x,y,z1)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                return False

    #Add the interior value
    if x0 < int_x < x1 and y0 < int_y < y1 and\
            z0 < int_z < z1:
        eval = eval_func(int_x,int_y,int_z)
        min_satisfied = min_satisfied or eval < other_sum
        max_satisfied = max_satisfied or eval > -other_sum
        if min_satisfied and max_satisfied:
            return False

    # No root possible
    return True

def quadratic_check_nd(test_coeff, tol):
    """One of subinterval_checks

    Finds the min of the absolute value of the quadratic part, and compares to the sum of the
    rest of the terms. There can't be a root if min(extreme_values) > other_sum	or if
    max(extreme_values) < -other_sum. We can short circuit and finish
    faster as soon as we find one value that is < other_sum and one value that > -other_sum.

    Parameters
    ----------
    test_coeff_in : numpy array
        The coefficient matrix of the polynomial to check
    tol: float
        The bound of the sup norm error of the chebyshev approximation.

    Returns
    -------
    True if there is guaranteed to be no root in the interval, False otherwise
    """
    #get the dimension and make sure the coeff tensor has all the right
    # quadratic coeff spots, set to zero if necessary
    dim = test_coeff.ndim
    padding = [(0,max(0,3-i)) for i in test_coeff.shape]
    test_coeff = np.pad(test_coeff.copy(), padding, mode='constant')
    interval = [-np.ones(dim), np.ones(dim)]

    #Possible extrema of qudaratic part are where D_xk = 0 for some subset of the variables xk
    # with the other variables are fixed to a boundary value
    #Dxk = c[0,...,0,1,0,...0] (k-spot is 1) + 4c[0,...,0,2,0,...0] xk (k-spot is 2)
    #       + \Sum_{j\neq k} xj c[0,...,0,1,0,...,0,1,0,...0] (k and j spot are 1)
    #This gives a symmetric system of equations AX+B = 0
    #We will fix different columns of X each time, resulting in slightly different
    #systems, but storing A and B now will be helpful later

    #pull out coefficients we care about
    quad_coeff = np.zeros([3]*dim)
    #A and B are arrays for slicing
    A = np.zeros([dim,dim])
    B = np.zeros(dim)
    pure_quad_coeff = [0]*dim
    for spot in itertools.product(range(3),repeat=dim):
        spot_deg = sum(spot)
        if spot_deg == 1:
            #coeff of linear terms
            i = [idx for idx in range(dim) if spot[idx]!= 0][0]
            B[i] = test_coeff[spot].copy()
            quad_coeff[spot] = test_coeff[spot]
            test_coeff[spot] = 0
        elif spot_deg == 0:
            #constant term
            const = test_coeff[spot].copy()
            quad_coeff[spot] = const
            test_coeff[spot] = 0
        elif spot_deg < 3:
            where_nonzero = [idx for idx in range(dim) if spot[idx]!= 0]
            if len(where_nonzero) == 2:
                #coeff of cross terms
                i,j = where_nonzero
                #with symmetric matrices, we only need to store the lower part
                A[j,i] = test_coeff[spot].copy()
                A[i,j] = A[j,i]
                #todo: see if we can store this in only one half of A

            else:
                #coeff of pure quadratic terms
                i = where_nonzero[0]
                pure_quad_coeff[i] = test_coeff[spot].copy()
            quad_coeff[spot] = test_coeff[spot]
            test_coeff[spot] = 0
    pure_quad_coeff_doubled = [p*2 for p in pure_quad_coeff]
    A[np.diag_indices(dim)] = [p*2 for p in pure_quad_coeff_doubled]

    #create a poly object for evals
    k0 = const - sum(pure_quad_coeff)
    def eval_func(point):
        "fast evaluation of quadratic chebyshev polynomials using horner's algorithm"
        _sum = k0
        for i,coord in enumerate(point):
            _sum += (B[i] + pure_quad_coeff_doubled[i]*coord + \
                     sum([A[i,j]*point[j] for j in range(i+1,dim)])) * coord
        return _sum

    #The sum of the absolute values of everything else
    other_sum = np.sum(np.abs(test_coeff)) + tol

    #iterator for sides
    fixed_vars = get_fixed_vars(dim)

    Done = False
    min_satisfied, max_satisfied = False,False
    #fix all variables--> corners
    for corner in itertools.product([0,1],repeat=dim):
        #j picks if upper/lower bound. i is which var
        eval = eval_func([interval[j][i] for i,j in enumerate(corner)])
        min_satisfied = min_satisfied or eval < other_sum
        max_satisfied = max_satisfied or eval > -other_sum
        if min_satisfied and max_satisfied:
            Done = True
            break
    #need to check sides/interior
    if not Done:
        X = np.zeros(dim)
        for fixed in fixed_vars:
            #fixed some variables --> "sides"
            #we only care about the equations from the unfixed variables
            fixed = np.array(fixed)
            unfixed = np.delete(np.arange(dim), fixed)
            A_ = A[unfixed][:,unfixed]
            #if diagonal entries change sign, can't be definite
            diag = np.diag(A_)
            for i,c in enumerate(diag[:-1]):
                #sign change?
                if c*diag[i+1]<0:
                    break
            #if no sign change, can find extrema
            else:
                #not full rank --> no soln
                if np.linalg.matrix_rank(A_,hermitian=True) == A_.shape[0]:
                    fixed_A = A[unfixed][:,fixed]
                    B_ = B[unfixed]
                    for side in itertools.product([0,1],repeat=len(fixed)):
                        X0 = np.array([interval[j][i] for i,j in enumerate(side)])
                        X_ = la.solve(A_, -B_-fixed_A@X0, assume_a='sym')
                        #make sure it's in the domain
                        for i,var in enumerate(unfixed):
                            if interval[0][var] <= X_[i] <= interval[1][var]:
                                continue
                            else:
                                break
                        else:
                            X[fixed] = X0
                            X[unfixed] = X_
                            eval = eval_func(X)
                            min_satisfied = min_satisfied or eval < other_sum
                            max_satisfied = max_satisfied or eval > -other_sum
                            if min_satisfied and max_satisfied:
                                Done = True
                                break
            if Done:
                break
        else:
            #fix no vars--> interior
            #if diagonal entries change sign, can't be definite
            for i,c in enumerate(pure_quad_coeff[:-1]):
                #sign change?
                if c*pure_quad_coeff[i+1]<0:
                    break
            #if no sign change, can find extrema
            else:
                #not full rank --> no soln
                if np.linalg.matrix_rank(A,hermitian=True) == A.shape[0]:
                    X = la.solve(A, -B, assume_a='sym')
                    #make sure it's in the domain
                    for i in range(dim):
                        if interval[0][i] <= X[i] <= interval[1][i]:
                            continue
                        else:
                            break
                    else:
                        eval = eval_func(X)
                        min_satisfied = min_satisfied or eval < other_sum
                        max_satisfied = max_satisfied or eval > -other_sum
                        if min_satisfied and max_satisfied:
                            Done = True
        #no root
    return not Done