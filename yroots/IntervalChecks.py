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
from scipy import linalg as la

class IntervalData:
    '''
    Class to handle all the things related to intervals. It holds and runs the interval checks
    and also tracks what happened to each interval, and how much progress has been made.

    Attributes
    ----------
    interval_checks: list
        A list of functions. Each function accepts a coefficient matrix and a tolerance,
        and returns whether the Chebyshev Polynomial represented by that matrix, and
        accurate to within that tolerance, can ever be zero on the n dimensional interval [-1,1].
    subinterval_checks: list
        A list of functions. Each function accepts a coefficient matrix, a list of subintervals, a list of
        sign changes, and a tolerance. It then returns a list of booleans whether the Chebyshev Polynomial
        represented by that matrix, and accurate to within that tolerance, can ever be zero on the given subintervals.
        Before the checks can be run the subintervals must be rescaled to subintervals of [-1,1]
        The list of sign changes represents if we already know the function changes sign on a given subinterval.
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
    polish_intervals: list
        The intervals polishing will be run on
    polish_num: int
        The number of time polishing has been run
    polish_interval_num: int
        The current interval being polished
    polish_a: numpy array
        The lower bounds of the interval being polished
    polish_b: numpy array
        The upper bounds of the interval being polished

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
        self.interval_results["Macaulay"] = []
        self.interval_results["Too Deep"] = []
        self.total_area = np.prod(self.b-self.a)
        self.current_area = 0.
        self.tick = 0

        #For polishing code
        self.polishing = False
        self.polish_intervals = []
        self.polish_num = 0
        self.polish_interval_num = -1
        self.polish_a = np.array([])
        self.polish_b = np.array([])

        #for keeping track of condition numbers
        self.cond = 0
        self.backcond = 0

    def add_polish_intervals(self, polish_intervals):
        ''' Add the intervals that polishing will be run on.

        Parameters
        ----------
        polish_intervals : list
            The intervals polishing will be run on.
        '''
        self.polishing = True
        self.polish_intervals = polish_intervals
        self.polish_num += 1
        self.polish_interval_num = -1

    def start_polish_interval(self):
        '''Get the tracking ready to track the next polished interval
        '''
        #self.tick = 99 #So it will print right away.
        self.polish_interval_num += 1
        self.polish_a, self.polish_b = self.polish_intervals[self.polish_interval_num]
        self.total_area = np.prod(self.polish_b-self.polish_a)
        self.current_area = 0.

    def check_interval(self, coeff, error, a, b):
        ''' Runs the interval checks on the interval [a,b]

        Parameters
        ----------
        coeff : numpy array.
            The coefficient matrix of the Chebyshev approximation to check.
        error: float
            The approximation error.
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
            if not check(coeff, error):
                if not self.polishing:
                    self.track_interval(check.__name__, [a,b])
                return True
        return False

    def check_subintervals(self, subintervals, scaled_subintervals, polys, change_sign, errors):
        ''' Runs the subinterval checks on the given subintervals of [-1,1]

        Parameters
        ----------
        subintervals : list
            A list of the intervals to check.
        scaled_subintervals: list
            A list of the subintervals to check, scaled to be within the unit box that the approxiations are valid on.
        polys: list
            The MultiCheb polynomials that approximate the functions on these intervals.
        change_sign: list
            A list of bools of whether we know the functions can change sign on the subintervals.
        errors: list
            The approximation errors of the polynomials.
        Returns
        -------
        check_interval : bool
            True if we can throw out the interval. Otherwise False.
        '''
        for check in self.subinterval_checks:
            for poly,error in zip(polys, errors):
                mask = check(poly, scaled_subintervals, change_sign, error)
                new_scaled_subintervals = []
                new_subintervals = []
                for i, result in enumerate(mask):
                    if result:
                        new_scaled_subintervals.append(scaled_subintervals[i])
                        new_subintervals.append(subintervals[i])
                    else:
                        if not self.polishing:
                            self.track_interval(check.__name__, subintervals[i])
                scaled_subintervals = new_scaled_subintervals
                subintervals = new_subintervals
        return subintervals

    def track_interval(self, name, interval):
        ''' Stores what happened to a given interval

        Parameters
        ----------
        name : string
            The name of the check or process (Macaulay, Base Case, Too Deep) that solved this interval
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
        self.tick += 1
        if self.tick >= 100:
            self.tick = 0
            if not self.polishing:
                print("\rPercent Finished: {}%       ".format(round(100*self.current_area/self.total_area,2)), end='')
            else:
                print_string =  '\rPolishing Round: {}'.format(self.polish_num)
                print_string += ' Interval: {}/{}:'.format(self.polish_interval_num, len(self.polish_intervals))
                print_string += " Percent Finished: {}%{}".format(round(100*self.current_area/self.total_area,2), ' '*20)
                print(print_string, end='')

    def print_results(self):
        ''' Prints the results of subdivision solve, how many intervals there were and what percent were
            solve by each check/method.
        '''
        results_numbers = np.array([len(self.interval_results[name]) for name in self.interval_results])
        total_intervals = sum(results_numbers)
        self.total_intervals = total_intervals
        checkers = [name for name in self.interval_results]
        print("Total intervals checked was {}".format(total_intervals))
        print("Methods used were {}".format(checkers))
        print("The percent solved by each was {}".format((100*results_numbers / total_intervals).round(4)))

    def plot_results(self, funcs, zeros, plot_intervals, print_plot=True):
        ''' Prints the results of subdivision solve. Only works if the functions are two dimensional.

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
        plt.figure(dpi=600)
        fig,ax = plt.subplots(1)
        fig.set_size_inches(6.5, 3)
        plt.xlim(self.a[0],self.b[0])
        plt.xlabel('$x$')
        plt.ylim(self.a[1],self.b[1])
        plt.ylabel('$y$')
        plt.title('Zero-Loci and Roots')

        dim = 2

        #print the contours
        contour_colors = ['#003cff','#50c878'] #royal blue and emerald green
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

        colors = ['w','#c3c3c3', 'C8', '#708090', '#897A57', '#D6C7A4','#73e600','#ccff99']
        #colors = ['w','#d3d3d3', '#708090', '#c5af7d', '#897A57', '#D6C7A4','#73e600','#ccff99']

        if plot_intervals:
            plt.title('')
            #plt.title('What happened to the intervals')
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
                        rect = patches.Rectangle((a0[0],a0[1]),b0[0]-a0[0],b0[1]-a0[1],linewidth=.1,\
                                                 edgecolor='red',facecolor=colors[i], label=check)
                    else:
                        rect = patches.Rectangle((a0[0],a0[1]),b0[0]-a0[0],b0[1]-a0[1],linewidth=.1,\
                                                 edgecolor='red',facecolor=colors[i])
                    ax.add_patch(rect)
            plt.legend()

        #Plot the zeros
        if len(zeros) > 0:
            plt.plot(np.real(zeros[:,0]), np.real(zeros[:,1]),'o',color='#ff0000',markeredgecolor='#ff0000',markersize=3,
                 zorder=22)

        if print_plot:
            plt.savefig('intervals.pdf', bbox_inches='tight')
        plt.show()

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
        return quadratic_check_2D(test_coeff, intervals, change_sign, tol)
    elif test_coeff.ndim == 3:
        return quadratic_check_3D(test_coeff, intervals, change_sign, tol)
    else:
        return quadratic_check_nd(test_coeff, intervals, change_sign, tol)

def quadratic_check_2D(test_coeff, intervals, change_sign, tol):
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

    #Get the coefficients of the quadratic part
    #Need to account for when certain coefs are zero.
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

    #The sum of the absolute values of the other coefs
    other_sum = np.sum(np.abs(test_coeff)) - np.sum(np.abs([c0,c1,c2,c3,c4,c5]))

    #function for evaluating c0 + c1x + c2y +c3x^2 + c4xy + c5y^2 (note these are chebyshev monomials)
    eval_func = lambda x,y: c0 + c1*x + c2*y + c3*(2*x**2-1) + c4*x*y + c5*(2*y**2-1)

    #The interior min
    #Comes from solving dx, dy = 0
    #Dx: 4c3x +  c4y = -c1    Matrix inverse is  [4c5  -c4]
    #Dy:  c4x + 4c5y = -c2                       [-c4  4c3]
    det = 16*c3*c5 - c4**2
    if det != 0:
        int_x = (c2*c4 - 4*c1*c5)/det
        int_y = (c1*c4 - 4*c2*c3)/det
    else:
        int_x = None

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
        #The partial with respect to y is zero
        #Dy:  c4x + 4c5y = -c2 =>   y = (-c2-c4x)/(4c5)
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
        #The partial with respect to x is zero
        #Dx: 4c3x +  c4y = -c1  =>  x = (-c1-c4y)/(4c3)
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
        if int_x is not None and interval[0][0] < int_x < interval[1][0] and interval[0][1] < int_y < interval[1][1]:
            extreme_points.append(eval_func(int_x,int_y))

        #No root if min(extreme_points) > (other_sum + tol)
        # OR max(extreme_points) < -(other_sum+tol)
        #Logical negation gives the boolean we want
        mask.append(np.min(extreme_points) < (other_sum + tol)
                and np.max(extreme_points) > -(other_sum+tol))

    return mask

def quadratic_check_3D(test_coeff, intervals, change_sign, tol):
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
    #Comes from solving dx, dy, dz = 0
    #Dx: 4c7x +  c4y +  c5z = -c1    Matrix inverse is  [(16c8c9-c6^2) -(4c4c9-c5c6)  (c4c6-4c5c8)]
    #Dy:  c4x + 4c8y +  c6z = -c2                       [-(4c4c9-c5c6) (16c7c9-c5^2) -(4c6c7-c4c5)]
    #Dz:  c5x +  c6y + 4c9z = -c3                       [(c4c6-4c5c8)  -(4c6c7-c4c5) (16c7c8-c4^2)]
    det = 4*c7*(16*c8*c9-c6**2) - c4*(4*c4*c9-c5*c6) + c5*(c4*c6-4*c5*c8)
    if det != 0:
        int_x = (c1*(c6**2-16*c8*c9) + c2*(4*c4*c9-c5*c6)  + c3*(4*c5*c8-c4*c6))/det
        int_y = (c1*(4*c4*c9-c5*c6)  + c2*(c5**2-16*c7*c9) + c3*(4*c6*c7-c4*c5))/det
        int_z = (c1*(4*c5*c8-c4*c6)  + c2*(4*c6*c7-c4*c5)  + c3*(c4**2-16*c7*c8))/det
    else:
        int_x = None

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

        #Adds the x and y constant boundaries
        #The partial with respect to z is zero
        #Dz:  c5x +  c6y + 4c9z = -c3   => z=(-c3-c5x-c6y)/(4c9)
        if c9 != 0:
            x = interval[0][0]
            y = interval[0][1]
            z = -(c3+c5*x+c6*y)/(4*c9)
            if interval[0][2] < z < interval[1][2]:
                extreme_points.append(eval_func(x,y,z))
            x = interval[1][0]
            y = interval[0][1]
            z = -(c3+c5*x+c6*y)/(4*c9)
            if interval[0][2] < z < interval[1][2]:
                extreme_points.append(eval_func(x,y,z))
            x = interval[0][0]
            y = interval[1][1]
            z = -(c3+c5*x+c6*y)/(4*c9)
            if interval[0][2] < z < interval[1][2]:
                extreme_points.append(eval_func(x,y,z))
            x = interval[1][0]
            y = interval[1][1]
            z = -(c3+c5*x+c6*y)/(4*c9)
            if interval[0][2] < z < interval[1][2]:
                extreme_points.append(eval_func(x,y,z))

        #Adds the x and z constant boundaries
        #The partial with respect to y is zero
        #Dy:  c4x + 4c8y + c6z = -c2   => y=(-c2-c4x-c6z)/(4c8)
        if c8 != 0:
            x = interval[0][0]
            z = interval[0][2]
            y = -(c2+c4*x+c6*z)/(4*c8)
            if interval[0][1] < y < interval[1][1]:
                extreme_points.append(eval_func(x,y,z))
            x = interval[1][0]
            z = interval[0][2]
            y = -(c2+c4*x+c6*z)/(4*c8)
            if interval[0][1] < y < interval[1][1]:
                extreme_points.append(eval_func(x,y,z))
            x = interval[0][0]
            z = interval[1][2]
            y = -(c2+c4*x+c6*z)/(4*c8)
            if interval[0][1] < y < interval[1][1]:
                extreme_points.append(eval_func(x,y,z))
            x = interval[1][0]
            z = interval[1][2]
            y = -(c2+c4*x+c6*z)/(4*c8)
            if interval[0][1] < y < interval[1][1]:
                extreme_points.append(eval_func(x,y,z))

        #Adds the y and z constant boundaries
        #The partial with respect to x is zero
        #Dx: 4c7x +  c4y +  c5z = -c1   => x=(-c1-c4y-c5z)/(4c7)
        if c7 != 0:
            y = interval[0][1]
            z = interval[0][2]
            x = -(c1+c4*y+c5*z)/(4*c7)
            if interval[0][0] < x < interval[1][0]:
                extreme_points.append(eval_func(x,y,z))
            y = interval[1][1]
            z = interval[0][2]
            x = -(c1+c4*y+c5*z)/(4*c7)
            if interval[0][0] < x < interval[1][0]:
                extreme_points.append(eval_func(x,y,z))
            y = interval[0][1]
            z = interval[1][2]
            x = -(c1+c4*y+c5*z)/(4*c7)
            if interval[0][0] < x < interval[1][0]:
                extreme_points.append(eval_func(x,y,z))
            y = interval[1][1]
            z = interval[1][2]
            x = -(c1+c4*y+c5*z)/(4*c7)
            if interval[0][0] < x < interval[1][0]:
                extreme_points.append(eval_func(x,y,z))

        #Add the x constant boundaries
        #The partials with respect to y and z are zero
        #Dy:  4c8y +  c6z = -c2 - c4x    Matrix inverse is [4c9  -c6]
        #Dz:   c6y + 4c9z = -c3 - c5x                      [-c6  4c8]
        det = 16*c8*c9-c6**2
        if det != 0:
            x = interval[0][0]
            y = (-4*c9*(c2+c4*x) +   c6*(c3+c5*x))/det
            z = (c6*(c2+c4*x)    - 4*c8*(c3+c5*x))/det
            if interval[0][1] < y < interval[1][1] and interval[0][2] < z < interval[1][2]:
                extreme_points.append(eval_func(x,y,z))
            x = interval[1][0]
            y = (-4*c9*(c2+c4*x) +   c6*(c3+c5*x))/det
            z = (c6*(c2+c4*x)    - 4*c8*(c3+c5*x))/det
            if interval[0][1] < y < interval[1][1] and interval[0][2] < z < interval[1][2]:
                extreme_points.append(eval_func(x,y,z))

        #Add the y constant boundaries
        #The partials with respect to x and z are zero
        #Dx: 4c7x +  c5z = -c1 - c4y    Matrix inverse is [4c9  -c5]
        #Dz:  c5x + 4c9z = -c3 - c6y                      [-c5  4c7]
        det = 16*c7*c9-c5**2
        if det != 0:
            y = interval[0][1]
            x = (-4*c9*(c1+c4*y) +   c5*(c3+c6*y))/det
            z = (c5*(c1+c4*y)    - 4*c7*(c3+c6*y))/det
            if interval[0][0] < x < interval[1][0] and interval[0][2] < z < interval[1][2]:
                extreme_points.append(eval_func(x,y,z))
            y = interval[1][1]
            x = (-4*c9*(c1+c4*y) +   c5*(c3+c6*y))/det
            z = (c5*(c1+c4*y)    - 4*c7*(c3+c6*y))/det
            if interval[0][0] < x < interval[1][0] and interval[0][2] < z < interval[1][2]:
                extreme_points.append(eval_func(x,y,z))

        #Add the z constant boundaries
        #The partials with respect to x and y are zero
        #Dx: 4c7x +  c4y  = -c1 - c5z    Matrix inverse is [4c8  -c4]
        #Dy:  c4x + 4c8y  = -c2 - c6z                      [-c4  4c7]
        det = 16*c7*c8-c4**2
        if det != 0:
            z = interval[0][2]
            x = (-4*c8*(c1+c5*z) +   c4*(c2+c6*z))/det
            y = (c4*(c1+c5*z)    - 4*c7*(c2+c6*z))/det
            if interval[0][0] < x < interval[1][0] and interval[0][1] < y < interval[1][1]:
                extreme_points.append(eval_func(x,y,z))
            z = interval[1][2]
            x = (-4*c8*(c1+c5*z) +   c4*(c2+c6*z))/det
            y = (c4*(c1+c5*z)    - 4*c7*(c2+c6*z))/det
            if interval[0][0] < x < interval[1][0] and interval[0][1] < y < interval[1][1]:
                extreme_points.append(eval_func(x,y,z))

        #Add the interior value
        if int_x is not None and interval[0][0] < int_x < interval[1][0] and interval[0][1] < int_y < interval[1][1] and\
                interval[0][2] < int_z < interval[1][2]:
            extreme_points.append(eval_func(int_x,int_y,int_z))

        #No root if min(extreme_points) > (other_sum + tol)
        # OR max(extreme_points) < -(other_sum+tol)
        #Logical negation gives the boolean we want
        mask.append(np.min(extreme_points) < (other_sum + tol)
                and np.max(extreme_points) > -(other_sum+tol))

    return mask

def quadratic_check_nd(test_coeff, intervals, change_sign, tol):
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
    #get the dimension and make sure the coeff tensor has all the right
    # quadratic coeff spots, set to zero if necessary
    dim = test_coeff.ndim
    padding = [(0,max(0,3-i)) for i in test_coeff.shape]
    test_coeff = np.pad(test_coeff.copy(), padding, mode='constant')

    #Possible extrema of qudaratic part are where D_xk = 0 for some subset of the variables xk
    # with the other variables are fixed to a boundary value
    #Dxk = c[0,...,0,1,0,...0] (k-spot is 1) + 4c[0,...,0,2,0,...0] xk (k-spot is 2)
    #       + \Sum_{j\neq k} xj c[0,...,0,1,0,...,0,1,0,...0] (k and j spot are 1)
    #This gives a symmetric system of equations AX+B = 0
    #We will fix different columns of X each time, resulting in slightly different
    #systems, but storing A and B now will be helpful later

    #pull out coefficients we care about
    quad_coeff = np.zeros([3]*dim)
    A = np.zeros([dim,dim])
    B = np.zeros(dim)
    for spot in itertools.product(np.arange(3),repeat=dim):
        if np.sum(spot) < 3:
            spot_array = np.array(spot)
            if np.sum(spot_array != 0) == 2:
                #coef of cross terms
                i,j = np.where(spot_array != 0)[0]
                A[i,j] = test_coeff[spot].copy()
                A[j,i] = test_coeff[spot].copy()
            elif np.any(spot_array == 2):
                #coef of pure quadratic terms
                i = np.where(spot_array != 0)[0][0]
                A[i,i] = 4*test_coeff[spot].copy()
            elif np.any(spot_array == 1):
                #coef of linear terms
                i = np.where(spot_array != 0)[0][0]
                B[i] = test_coeff[spot].copy()
            quad_coeff[spot] = test_coeff[spot]
            test_coeff[spot] = 0

    #create a poly object for evaluations
    quad_poly = MultiCheb(quad_coeff)

    #The sum of the absolute values of everything else
    other_sum = np.sum(np.abs(test_coeff))

    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r)\
                                                     for r in range(len(s)+1))
    mask = []
    for interval_num, interval in enumerate(intervals):
        if change_sign[interval_num]:
            mask.append(True)
            continue

        extreme_points = []
        for fixed in powerset(np.arange(dim)):
            fixed = np.array(fixed)
            if len(fixed) == 0:
                #fix no vars--> interior
                if np.linalg.matrix_rank(A) < A.shape[0]:
                    #no interior critical point
                    continue
                X = la.solve(A, -B, assume_a='sym')
                #make sure it's in the domain
                if np.all([interval[0][i] <= X[i] <= interval[1][i] for i in range(dim)]):
                    extreme_points.append(quad_poly(X))
            elif len(fixed) == dim:
                #fix all variables--> corners
                for corner in itertools.product([0,1],repeat=dim):
                    #j picks if upper/lower bound. i is which var
                    extreme_points.append(quad_poly([interval[j][i] for i,j in enumerate(corner)]))
            else:
                #fixed some variables --> "sides"
                #we only care about the equations from the unfixed variables
                unfixed = np.delete(np.arange(dim), fixed)
                A_ = A[unfixed][:,unfixed]
                if np.linalg.matrix_rank(A_) < A_.shape[0]:
                    #no solutions
                    continue
                fixed_A = A[unfixed][:,fixed]
                B_ = B[unfixed]

                for side in itertools.product([0,1],repeat=len(fixed)):
                    X0 = np.array([interval[j][i] for i,j in enumerate(side)])
                    X_ = la.solve(A_, -B_-fixed_A@X0, assume_a='sym')
                    X = np.zeros(dim)
                    X[fixed] = X0
                    X[unfixed] = X_
                    if np.all([interval[0][i] <= X[i] <= interval[1][i] for i in range(dim)]):
                        extreme_points.append(quad_poly(X))

        #No root if min(extreme_points) > (other_sum + tol)
        # OR max(extreme_points) < -(other_sum+tol)
        #Logical negation gives the boolean we want
        mask.append(np.min(extreme_points) < (other_sum + tol)
                and np.max(extreme_points) > -(other_sum+tol))

    return mask
