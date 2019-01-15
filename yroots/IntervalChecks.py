"""
The check functions are all functions that take in a coefficent matrix and run a quick check
to determine if there can ever be zeros on the unit box there. They are then put into the list
all_bound_check_functions in the order we want to run them (probably fastest first). These are
then all run to throw out intervals as possible.
"""
import numpy as np
from itertools import product
import itertools
from yroots.polynomial import MultiCheb
from matplotlib import pyplot as plt
from yroots.polynomial import MultiCheb, Polynomial
from matplotlib import patches

class IntervalData:
    '''
    Class to handle all the things realted to intervals. It holds and runs the interval checks
    and also tracks what happened to each interval, and how much progress has been made.

    Attributes
    ----------
    interval_checks: list
        A list of functions. Each function accepts a coefficient matrix and a tolerance,
        and returns whether the Chebyshev Polynomial represented by that matrix, and
        accurate to within that tolerance, can ever be zero on the n dimensional interval [-1,1].
    subinterval_checks:list
        A list of functions. Each function accepts a coefficient matrix, a list of intervals, a list of
        sign changes,and a tolerance. It then returns a list of booleans whether the Chebyshev Polynomial
        represented by that matrix, and accurate to within that tolerance, can ever be zero on the given intervals.
        The list of sign changes represents if we already know the function changes sign on a given interval.
    a: numpy array
        The lower bounds of the overall interval to solve on.
    b: numpy array
        The upper bounds of the overall interval to solve on.
    interval_results: dictionary
        A dictionary of funciton names to lists of intervals that were solved by that function.
    total_area: float
        The total n dimensional volume of the overall interval being solved on.
    current_area: float
        How much of the n dimensional volume has been checked.
    polishing: bool
        If true this class is just being used as a shell to pass into the polish code.
    tick: int
        Keeps track of how many intervals have been solved. Every 100 it resets and prints the progress.

    Methods
    -------
    __init__
        Initializes everything.
    check_intervals
        Checks if a polynomial can be zero on an interval.
    check_subintervals
        Checks if a polynomial can be zero on an list of intervals.
    track_interval
        Tracks what happened to a given interval.
    print_progress
        Prints what percentage of the domain has been searched
    print_results
        Prints the results of how much each method contributed to the overall search
    plot_results
        Plots the results of subdivision solve
    '''
    def __init__(self,a,b):
        self.interval_checks = [constant_term_check]
        self.subinterval_checks = [quadratic_check]
        self.a = a
        self.b = b
        self.interval_results = dict()
        for check in self.interval_checks:
            self.interval_results[check.__name__] = []
        for check in self.subinterval_checks:
            self.interval_results[check.__name__] = []
        self.interval_results["Base Case"] = []
        self.interval_results["Division"] = []
        self.total_area = np.prod(self.b-self.a)
        self.current_area = 0.
        self.polishing = False
        self.tick = 0

    def check_interval(self, coeff, approx_tol, a, b):
        ''' Runs the interval checks on the interval [a,b]

        Parameters
        ----------
        coeff : numpy array.
            The coefficient matrix of the Chebyshev approximation to check.
        approx_tol: float
            The sup norm bound on the approximation error.
        a: numpy array
            The lower bounds of the interval to check.
        b: numpy array
            The upper bounds of the interval to check.
        Returns
        -------
        check_interval : bool
            True if we can throw out the interval. Otherwise False.
        '''
        for check in self.interval_checks:
            if not check(coeff, approx_tol):
                self.track_interval(check.__name__, [a,b])
                return True
        return False

    def check_subintervals(self, subintervals, scaled_subintervals, polys, change_sign, approx_tol):
        ''' Runs the subinterval checks on the given intervals

        Parameters
        ----------
        subintervals : list
            A list of the intervals to check.
        scaled_subintervals: list
            A list of the intervals to check, scaled to the unit box that the approxiations are valid on.
        polys: list
            The MultiCheb polynomials that approximate the functions on these intervals.
        change_sign: list
            A list of bools of whether we know the functions can change sign on the subintervals.
        approx_tol: float
            The sup norm bound on the approximation error.
        Returns
        -------
        check_interval : bool
            True if we can throw out the interval. Otherwise False.
        '''
        for check in self.subinterval_checks:
            for poly in polys:
                mask = check(poly, scaled_subintervals, change_sign, approx_tol)
                new_scaled_subintervals = []
                new_subintervals = []
                for i, result in enumerate(mask):
                    if result:
                        new_scaled_subintervals.append(scaled_subintervals[i])
                        new_subintervals.append(subintervals[i])
                    else:
                        self.track_interval(check.__name__, subintervals[i])
                scaled_subintervals = new_scaled_subintervals
                subintervals = new_subintervals
        return subintervals

    def track_interval(self, name, interval):
        ''' Stores what happened to a given interval

        Parameters
        ----------
        name : string
            The name of the check or process (Division, Base Case) that solved this interval
        interval: list
            [a,b] where a and b are the lower and upper bound of the interval to track.
        '''
        if not self.polishing:
            self.interval_results[name].append(interval)
            self.current_area += np.prod(interval[1] - interval[0])

    def print_progress(self):
        ''' Prints the progress of subdivision solve. Only prints every 100th time this function is
            called to save time.
        '''
        if not self.polishing:
            if self.tick == 100:
                self.tick = 0
                print("\rPercent Finished: {}%       ".format(round(100*self.current_area/self.total_area,2)), end='')
            self.tick += 1

    def print_results(self):
        ''' Prints the results of subdivision solve, how many intervals there were and what percent were
            solve by each check/method.
        '''
        results_numbers = np.array([len(self.interval_results[name]) for name in self.interval_results])
        total_intervals = sum(results_numbers)
        checkers = [name for name in self.interval_results]
        print("Total intervals checked was {}".format(total_intervals))
        print("Methods used were {}".format(checkers))
        print("The percent solved by each was {}".format((100*results_numbers / total_intervals).round(2)))

    def plot_results(self, funcs, zeros, plot_intervals):
        ''' Prints the results of subdivision solve. Only works if the funcitons are two dimensional.

        Parameters
        ----------
        funcs : list
            A list of the functions the were solved
        zeros: numpy array
            Each row is a zero of the funcitons
        plot_intervals: bool
            If true, shows on the plot which areas were solved by which check/method.
        '''
        #colors: use alpha = .5, dark green, black, orange roots. Change colors of check info plots
        #3D plot with small alpha, matplotlib interactive, animation
        #make logo
        #make easier to input lower/upper bounds as a list
        plt.figure(dpi=1200)
        fig,ax = plt.subplots(1)
        fig.set_size_inches(10, 10)
        plt.xlim(self.a[0],self.b[0])
        plt.xlabel('$x$')
        plt.ylim(self.a[1],self.b[1])
        plt.ylabel('$y$')
        plt.title('Zero-Loci and Roots')

        dim = 2

        #print the contours
        contour_colors = ['#00cc00','#cc33ff']
        x = np.linspace(self.a[0],self.b[0],1000)
        y = np.linspace(self.a[1],self.b[1],1000)
        X,Y = np.meshgrid(x,y)
        for i in range(dim):
            if isinstance(funcs[i], Polynomial):
                Z = np.zeros_like(X)
                for spot,num in np.ndenumerate(X):
                    Z[spot] = funcs[i]([X[spot],Y[spot]])
                plt.contour(X,Y,Z,levels=[0],colors=contour_colors[i])
            else:
                plt.contour(X,Y,funcs[i](X,Y),levels=[0],colors=contour_colors[i])

        #Plot the zeros
        plt.plot(np.real(zeros[:,0]), np.real(zeros[:,1]),'.',color='k',markersize=10,label='Roots')
        colors = ['w','#d3d3d3', '#708090', '#c5af7d', '#897A57', '#D6C7A4','#73e600','#ccff99']

        if plot_intervals:
            plt.title('What happened to the intervals')
            #plot results
            i = -1
            for check in self.interval_results:
                i += 1
                results = self.interval_results[check]
                first = True
                for data in results:
                    a0,b0 = data
                    if first:
                        first = False
                        rect = patches.Rectangle((a0[0],a0[1]),b0[0]-a0[0],b0[1]-a0[1],linewidth=.05,\
                                                 edgecolor='k',facecolor=colors[i], label=check)
                    else:
                        rect = patches.Rectangle((a0[0],a0[1]),b0[0]-a0[0],b0[1]-a0[1],linewidth=.05,\
                                                 edgecolor='k',facecolor=colors[i])
                    ax.add_patch(rect)
            plt.legend()
        plt.show()

def extreme_val3(test_coeff, maxx = True):
    ''' Finds the extreme value of test_coeff on -1 to 1, used by quad_check

    test_coeff is [a,b,c] and represents the funciton a + bx + c(2x^2 - 1).
    Basic calculus can be used to find the extreme values.

    Parameters
    ----------
    test_coeff : numpy array
        Array representing [a,b,c]
    maxx: bool
        If true returns the absolute value of the max of the funciton, otherwise returns
        the absolute value of the min of the function.
    Returns
    -------
    extreme_val3 : float
        The extreme value (max or min) of the absolute value of a + bx + c(2x^2 - 1).
    '''
    a,b,c = test_coeff
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

def constant_term_check(test_coeff, tol):
    """One of interval_checks

    Checks if the constant term is bigger than all the other terms combined, using the fact that
    each Chebyshev monomial is bounded by 1.

    Parameters
    ----------
    test_coeff : numpy array
        The coefficient matrix of the polynomial to check
    tol: float
        The bound of the sup norm error of the chebyshev approximation.

    Returns
    -------
    constant_term_check : bool
        False if the function is guarenteed to never be zero in the unit box, True otherwise
    """
    test_sum = np.sum(np.abs(test_coeff))
    if np.abs(test_coeff[tuple([0]*test_coeff.ndim)]) * 2 > test_sum + tol:
        return False
    else:
        return True

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

    #Get the min of the quadratic including the constant term
    start = extreme_val3(test_coeff[tuple(slices)], maxx = False)
    rest = 0

    #Get the max's of the other quadratics
    shape = list(test_coeff.shape)
    shape[slice_direc] = 1
    for spots in itertools.product(*[np.arange(i) for i in shape]):
        if sum(spots) > 0:
            for i in range(1, dim):
                slices[i] = spots[i]
            rest += extreme_val3(test_coeff[tuple(slices)])

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

    if start > rest + tol:
        return False
    else:
        return True

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

def linear_check(test_coeff_in, intervals, change_sign, tol):
    """One of subinterval_checks

    Checks the max of the linear part of the approximation and compares to the sum of the other terms.

    Parameters
    ----------
    test_coeff_in : numpy array
        The coefficient matrix of the polynomial to check
    intervals : list
        A list of the intervals to check.
    change_sign: list
        A list of bools of whether we know the functions can change sign on the subintervals.
    tol: float
        The bound of the sup norm error of the chebyshev approximation.

    Returns
    -------
    mask : list
        A list of the results of each interval. False if the function is guarenteed to never be zero
        in the unit box, True otherwise
    """
    dim = test_coeff_in.ndim
    coeff_abs_sum = np.sum(np.abs(test_coeff_in))
    mask = []
    for i, interval in enumerate(intervals):
        if change_sign[i]:
            mask.append(True)
            continue

        test_coeff = test_coeff_in.copy()

        a,b = interval

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
        if 2*abs_smallest_corner > coeff_abs_sum + tol:
            # case: corner is far enough from 0
            mask.append(False)
        else:
            mask.append(True)

    return mask

def quadratic_check(test_coeff, intervals,change_sign,tol):
    """One of subinterval_checks

    Finds the min of the absolute value of the quadratic part, and compares to the sum of the
    rest of the terms. quadratic_check_2D and quadratic_check_3D are faster so runs those if it can,
    otherwise it runs the genereic n-dimensional version.

    Parameters
    ----------
    test_coeff_in : numpy array
        The coefficient matrix of the polynomial to check
    intervals : list
        A list of the intervals to check.
    change_sign: list
        A list of bools of whether we know the functions can change sign on the subintervals.
    tol: float
        The bound of the sup norm error of the chebyshev approximation.

    Returns
    -------
    mask : list
        A list of the results of each interval. False if the function is guarenteed to never be zero
        in the unit box, True otherwise
    """
    if test_coeff.ndim == 2:
        return quadratic_check_2D(test_coeff, intervals,change_sign,tol)
    elif test_coeff.ndim == 3:
        return quadratic_check_3D(test_coeff, intervals,change_sign,tol)
    else:
        return quadratic_check_nd(test_coeff, intervals,change_sign,tol)

def quadratic_check_2D(test_coeff, intervals,change_sign,tol):
    """One of subinterval_checks

    Finds the min of the absolute value of the quadratic part, and compares to the sum of the
    rest of the terms.

    Parameters
    ----------
    test_coeff_in : numpy array
        The coefficient matrix of the polynomial to check
    intervals : list
        A list of the intervals to check.
    change_sign: list
        A list of bools of whether we know the functions can change sign on the subintervals.
    tol: float
        The bound of the sup norm error of the chebyshev approximation.

    Returns
    -------
    mask : list
        A list of the results of each interval. False if the function is guarenteed to never be zero
        in the unit box, True otherwise
    """
    if test_coeff.ndim != 2:
        return [True]*len(intervals)

    #Padding is slow, so check the shape instead.
    shape = test_coeff.shape
    c0 = test_coeff[0,0]
    if shape[0] > 1:
        c1 = test_coeff[1,0]
    else:
        c1 = 0
    if shape[1] > 1:
        c2 = test_coeff[0,1]
    else:
        c2 = 0
    if shape[0] > 2:
        c3 = test_coeff[2,0]
    else:
        c3 = 0
    if shape[0] > 1 and shape[1] > 1:
        c4 = test_coeff[1,1]
    else:
        c4 = 0
    if shape[1] > 2:
        c5 = test_coeff[0,2]
    else:
        c5 = 0

    #The sum of the absolute values of everything else
    other_sum = np.sum(np.abs(test_coeff)) - np.sum(np.abs([c0,c1,c2,c3,c4,c5]))

    #function for evaluating c0 + c1x + c2y +c3x^2 + c4xy + c5y^2 (note these are chebyshev monomials)
    eval_func = lambda x,y: c0 + c1*x + c2*y + c3*(2*x**2-1) + c4*x*y + c5*(2*y**2-1)

    #The interior min
    det = 16*c3*c5 - c4**2
    if det != 0:
        int_x = (c2*c4 - 4*c1*c5)/det
        int_y = (c1*c4 - 4*c2*c3)/det
    else:#Something outside the unit box
        int_x = 100
        int_y = 100

    mask = []
    for i, interval in enumerate(intervals):
        if change_sign[i]:
            mask.append(True)
            continue

        extreme_points = []
        #Add all the corners
        extreme_points.append(eval_func(interval[0][0], interval[0][1]))
        extreme_points.append(eval_func(interval[1][0], interval[0][1]))
        extreme_points.append(eval_func(interval[0][0], interval[1][1]))
        extreme_points.append(eval_func(interval[1][0], interval[1][1]))

        #Add the x constant boundaries
        if c5 != 0:
            x = interval[0][0]
            y = -(c2 + c4*x)/(4*c5)
            if interval[0][1] < y < interval[1][1]:
                extreme_points.append(eval_func(x,y))
            x = interval[1][0]
            y = -(c2 + c4*x)/(4*c5)
            if interval[0][1] < y < interval[1][1]:
                extreme_points.append(eval_func(x,y))

        #Add the y constant boundaries
        if c3 != 0:
            y = interval[0][1]
            x = -(c1 + c4*y)/(4*c3)
            if interval[0][0] < x < interval[1][0]:
                extreme_points.append(eval_func(x,y))
            y = interval[1][1]
            x = -(c1 + c4*y)/(4*c3)
            if interval[0][0] < x < interval[1][0]:
                extreme_points.append(eval_func(x,y))

        #Add the interior value
        if interval[0][0] < int_x < interval[1][0] and interval[0][1] < int_y < interval[1][1]:
            extreme_points.append(eval_func(int_x,int_y))

        extreme_points = np.array(extreme_points)

        #If sign change, True
        if not np.all(extreme_points > 0) and not np.all(extreme_points < 0):
            mask.append(True)
            continue

        if np.min(np.abs(extreme_points)) - tol > other_sum:
            mask.append(False)
        else:
            mask.append(True)

    return mask

def quadratic_check_3D(test_coeff, intervals,change_sign,tol):
    """One of subinterval_checks

    Finds the min of the absolute value of the quadratic part, and compares to the sum of the
    rest of the terms.

    Parameters
    ----------
    test_coeff_in : numpy array
        The coefficient matrix of the polynomial to check
    intervals : list
        A list of the intervals to check.
    change_sign: list
        A list of bools of whether we know the functions can change sign on the subintervals.
    tol: float
        The bound of the sup norm error of the chebyshev approximation.

    Returns
    -------
    mask : list
        A list of the results of each interval. False if the function is guarenteed to never be zero
        in the unit box, True otherwise
    """
    if test_coeff.ndim != 3:
        return [True]*len(intervals)

    #Padding is slow, so check the shape instead.
    shape = test_coeff.shape
    c0 = test_coeff[0,0,0]
    if shape[0] > 1:
        c1 = test_coeff[1,0,0]
    else:
        c1 = 0
    if shape[1] > 1:
        c2 = test_coeff[0,1,0]
    else:
        c2 = 0
    if shape[2] > 1:
        c3 = test_coeff[0,0,1]
    else:
        c3 = 0
    if shape[0] > 1 and shape[1] > 1:
        c4 = test_coeff[1,1,0]
    else:
        c4 = 0
    if shape[0] > 1 and shape[2] > 1:
        c5 = test_coeff[1,0,1]
    else:
        c5 = 0
    if shape[1] > 1 and shape[2] > 1:
        c6 = test_coeff[0,1,1]
    else:
        c6 = 0
    if shape[0] > 2:
        c7 = test_coeff[2,0,0]
    else:
        c7 = 0
    if shape[1] > 2:
        c8 = test_coeff[0,2,0]
    else:
        c8 = 0
    if shape[2] > 2:
        c9 = test_coeff[0,0,2]
    else:
        c9 = 0

    #function for evaluating c0 + c1x + c2y +c3z + c4xy + c5xz + c6yz + c7x^2 + c8y^2 + c9z^2
    #(note these are chebyshev monomials)
    eval_func = lambda x,y,z: c0 + c1*x + c2*y + c3*z + c4*x*y + c5*x*z + c6*y*z +\
                                    c7*(2*x**2-1) + c8*(2*y**2-1) + c9*(2*z**2-1)

    #The sum of the absolute values of everything else
    other_sum = np.sum(np.abs(test_coeff)) - np.sum(np.abs([c0,c1,c2,c3,c4,c5,c6,c7,c8,c9]))

    #The interior min
    det = 4*c7*(16*c8*c9-c6**2) - c4*(4*c9*c4-c6*c5) + c5*(c6*c4-4*c8*c5)
    if det != 0:
        int_x = (c1*(c6**2-16*c9*c8) + c2*(4*c9*c4-c6*c5) + c3*(4*c8*c5-c6*c4))/det
        int_y = (c1*(4*c9*c4-c6*c5) + c2*(c5**2-16*c9*c7) + c3*(4*c6*c7-c4*c5))/det
        int_z = (c1*(4*c8*c5-c6*c4) + c2*(4*c6*c7-c4*c5) + c3*(c4**2-16*c7*c8))/det
    else:#Something outside the unit box
        int_x = 100
        int_y = 100
        int_z = 100

    mask = []
    for interval_num, interval in enumerate(intervals):
        if change_sign[interval_num]:
            mask.append(True)
            continue

        extreme_points = []
        #Add all the corners
        extreme_points.append(eval_func(interval[0][0], interval[0][1], interval[0][2]))
        extreme_points.append(eval_func(interval[1][0], interval[0][1], interval[0][2]))
        extreme_points.append(eval_func(interval[0][0], interval[1][1], interval[0][2]))
        extreme_points.append(eval_func(interval[0][0], interval[0][1], interval[1][2]))
        extreme_points.append(eval_func(interval[1][0], interval[1][1], interval[0][2]))
        extreme_points.append(eval_func(interval[1][0], interval[0][1], interval[1][2]))
        extreme_points.append(eval_func(interval[0][0], interval[1][1], interval[1][2]))
        extreme_points.append(eval_func(interval[1][0], interval[1][1], interval[1][2]))

        #Add the x constant boundaries
        det = 16*c8*c9-c6**2
        if det != 0:
            x = interval[0][0]
            y = (4*c8*(-c2-c4*x)+c6*(-c3-c5*x))/det
            z = (c6*(-c2-c4*x)+4*c9*(-c3-c5*x))/det
            if interval[0][1] < y < interval[1][1] and interval[0][2] < z < interval[1][2]:
                extreme_points.append(eval_func(x,y,z))
            x = interval[1][0]
            y = (4*c8*(-c2-c4*x)+c6*(-c3-c5*x))/det
            z = (c6*(-c2-c4*x)+4*c9*(-c3-c5*x))/det
            if interval[0][1] < y < interval[1][1] and interval[0][2] < z < interval[1][2]:
                extreme_points.append(eval_func(x,y,z))

        #Add the y constant boundaries
        det = 16*c7*c9-c5**2
        if det != 0:
            y = interval[0][1]
            x = (4*c7*(-c1-c4*y)+c5*(-c3-c6*y))/det
            z = (c5*(-c1-c4*y)+4*c9*(-c3-c6*y))/det
            if interval[0][0] < x < interval[1][0] and interval[0][2] < x < interval[1][2]:
                extreme_points.append(eval_func(x,y,z))
            y = interval[1][1]
            x = (4*c7*(-c1-c4*y)+c5*(-c3-c6*y))/det
            z = (c5*(-c1-c4*y)+4*c9*(-c3-c6*y))/det
            if interval[0][0] < x < interval[1][0] and interval[0][2] < x < interval[1][2]:
                extreme_points.append(eval_func(x,y,z))

        #Add the z constant boundaries
        A = np.array([[4*c7,c4],[c4,4*c8]])
        det = 16*c7*c8-c4**2
        if det != 0:
            z = interval[0][2]
            x = (4*c7*(-c1-c5*z)+c4*(-c2-c6*z))/det
            y = (c4*(-c1-c5*z)+4*c8*(-c2-c6*z))/det
            if interval[0][0] < x < interval[1][0] and interval[0][1] < y < interval[1][1]:
                extreme_points.append(eval_func(x,y,z))
            z = interval[1][2]
            x = (4*c7*(-c1-c5*z)+c4*(-c2-c6*z))/det
            y = (c4*(-c1-c5*z)+4*c8*(-c2-c6*z))/det
            if interval[0][0] < x < interval[1][0] and interval[0][1] < y < interval[1][1]:
                extreme_points.append(eval_func(x,y,z))

        #Add the interior value
        if interval[0][0] < int_x < interval[1][0] and interval[0][1] < int_y < interval[1][1] and\
                interval[0][2] < int_z < interval[1][2]:
            extreme_points.append(eval_func(int_x,int_y,int_z))

        extreme_points = np.array(extreme_points)

        #If sign change, True
        if not np.all(extreme_points > 0) and not np.all(extreme_points < 0):
            mask.append(True)
            continue

        if np.min(np.abs(extreme_points)) - tol > other_sum:
            mask.append(False)
        else:
            mask.append(True)

    return mask

def quadratic_check_nd(test_coeff, intervals,change_sign,tol):
    """One of subinterval_checks

    Finds the min of the absolute value of the quadratic part, and compares to the sum of the
    rest of the terms.

    Parameters
    ----------
    test_coeff_in : numpy array
        The coefficient matrix of the polynomial to check
    intervals : list
        A list of the intervals to check.
    change_sign: list
        A list of bools of whether we know the functions can change sign on the subintervals.
    tol: float
        The bound of the sup norm error of the chebyshev approximation.

    Returns
    -------
    mask : list
        A list of the results of each interval. False if the function is guarenteed to never be zero
        in the unit box, True otherwise
    """
    if change_sign is None:
        return -1

    dim = test_coeff.ndim
    padding = [(0,max(0,3-i)) for i in test_coeff.shape]
    test_coeff = np.pad(test_coeff.copy(), padding, mode='constant')

    quad_coeff = np.zeros([3]*dim)
    C = np.zeros([dim,dim])
    C1 = np.zeros(dim)
    C2 = np.zeros(dim)
    for spot in itertools.product(np.arange(3),repeat=dim):
        if np.sum(spot) < 3:
            spot_array = np.array(spot)
            if np.sum(spot_array != 0) == 2:
                C[tuple(np.where([spot_array != 0])[1])] = test_coeff[spot]
                C[tuple(np.where([spot_array != 0])[1][::-1])] = test_coeff[spot]
            elif np.any(spot_array == 2):
                C2[np.where([spot_array == 2])[1][0]] = test_coeff[spot]
            elif np.any(spot_array == 1):
                C1[np.where([spot_array == 1])[1][0]] = test_coeff[spot]

            quad_coeff[spot] = test_coeff[spot]
            test_coeff[spot] = 0

    quad_poly = MultiCheb(quad_coeff)

    #The sum of the absolute values of everything else
    other_sum = np.sum(np.abs(test_coeff))

    mask = []
    for interval_num, interval in enumerate(intervals):
        if change_sign[interval_num]:
            mask.append(True)
            continue
        ##If constant + c1x + c2y +c3xy = 0 in the region, useless

        extreme_points = []
        #Add all the corners
        for corner in itertools.product([0,1],repeat=dim):
            extreme_points.append(quad_poly([interval[j][i] for i,j in enumerate(corner)]))

        for var in range(dim):
            #Add the x_i constant boundaries
            A = np.vstack([np.delete(C[var],var,0)]*(dim-1))
            np.fill_diagonal(A, 0)
            A += 4*np.diag(np.delete(C2,var,0))
            if np.linalg.matrix_rank(A) < dim:
                continue
            #First Boundary
            x_var = interval[0][var]
            B = -np.delete(C[var],var,0)*x_var-np.delete(C1,var,0)
            ext_spot = np.linalg.solve(A,B)
            if np.all(np.delete(interval[0],var,0) < ext_spot) and\
                        np.all(np.delete(interval[1],var,0) > ext_spot):
                ext_spot = np.insert(ext_spot,var,x_var)
                extreme_points.append(quad_poly(ext_spot))
            #Second Boundary
            x_var = interval[1][var]
            B = -np.delete(C[var],var,0)*x_var-np.delete(C1,var,0)
            ext_spot = np.linalg.solve(A,B)
            if np.all(np.delete(interval[0],var,0) < ext_spot) and\
                        np.all(np.delete(interval[1],var,0) > ext_spot):
                ext_spot = np.insert(ext_spot,var,x_var)
                extreme_points.append(quad_poly(ext_spot))

        #Add the interior value
        A = C.copy()
        A += 4*np.diag(C2)
        B = -C1.copy()
        if np.linalg.matrix_rank(A) == dim:
            ext_spot = np.linalg.solve(A,B)
            if np.all(interval[0] < ext_spot) and np.all(interval[1] > ext_spot):
                extreme_points.append(quad_poly(ext_spot))

        extreme_points = np.array(extreme_points)

        #If sign change, True
        if not np.all(extreme_points > 0) and not np.all(extreme_points < 0):
            mask.append(True)
            continue

        if np.min(np.abs(extreme_points)) - tol > other_sum:
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
