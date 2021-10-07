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
from math import fabs                      # faster than np.abs for small arrays
from yroots.utils import memoize, transform, get_var_list, isNumber
from copy import copy


INTERVAL_REDUCTION_FUNCS = ["improveBound", "getBoundingParallelogram"]

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
    intervalReductionMethodsToUse: list
        A list of indices to index into INTERVAL_REDUCTION_FUNCS_2D and INTERVAL_REDUCTION_FUNCS_ND
        to run interval reduction methods on each subinterval.

    Methods
    -------
    __init__
        Initializes everything.
    get_subintervals
        Returns the intervals needed for subdivision after running subinterval checks
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
    def __init__(self, a, b, intervalReductions):
        self.interval_checks = []
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
        self.interval_results["getBoundingInterval"] = []
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

        # Variables to store for Subintervals
        if isNumber(a):
            return
        dim = len(a)
        self.RAND = 0.5139303900908738
        self.mask = np.ones([2]*dim, dtype = bool)
        self.throwOutMask = np.zeros([2]*dim, dtype = bool)
        self.middleVal = 2*self.RAND - 1
        self.middleValChebSqrd = 2*self.middleVal**2 - 1
        self.middleValSqrd = self.middleVal**2
        self.subintervals = np.zeros([2]*dim + [2, dim])
        for spot in product([0,1], repeat=dim):
            for i,val in enumerate(spot):
                self.subintervals[spot][0][i] = -1 if val == 0 else self.middleVal
                self.subintervals[spot][1][i] = self.middleVal if val == 0 else 1
        self.__intervalReductionMethodsToUse = []
        for methodName in intervalReductions:
            if methodName in INTERVAL_REDUCTION_FUNCS:
                self.__intervalReductionMethodsToUse.append(INTERVAL_REDUCTION_FUNCS.index(methodName))

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

    def get_subintervals(self, a, b, polys, errors, runChecks):
            """Gets the subintervals to divide a search interval into.

            Parameters
            ----------
            a : numpy array
                The lower bound on the interval.
            b : numpy array
                The upper bound on the interval.
            dimensions : numpy array
                The dimensions we want to cut in half.
            polys : list
                A list of MultiCheb polynomials representing the function approximations on the
                interval to subdivide. Used in the subinterval checks.
            errors: list of floats
                The bound of the sup norm error of the chebyshev approximation.

            Returns
            -------
            subintervals : list
                Each element of the list is a tuple containing an a and b, the lower and upper bounds of the interval.
            """
            #Try to find a bounding interval
            boundingSize = np.inf
            if len(self.__intervalReductionMethodsToUse) != 0:
                boundingInterval = getBoundingInterval(polys, errors, self.__intervalReductionMethodsToUse)
            else:
                boundingInterval = None
            if boundingInterval is not None:
                boundingSize = np.product(boundingInterval[1] - boundingInterval[0])
                boundingInterval = transform(boundingInterval, a, b)
            #See we should use it
            if boundingSize == 0:
                self.track_interval_bounded(getBoundingInterval.__name__, [a,b], boundingInterval)
                return []
            elif boundingSize < 0.5: #Something to think about
                self.track_interval_bounded(getBoundingInterval.__name__, [a,b], boundingInterval)
                return [boundingInterval]

            #Default to keeping everything
            self.mask.fill(True)

            #For getting the subintervals
            temp1 = b - a
            temp2 = b + a

            #Create the new intervals based on the ones we are keeping
            newIntervals = self.subintervals.copy()
            newIntervals[:,:1,:] = (newIntervals[:,:1,:] * temp1 + temp2) / 2
            newIntervals[:,1:,:] = (newIntervals[:,1:,:] * temp1 + temp2) / 2

            thrownOuts = []
            if runChecks:
                #Run checks to set mask to False
                for check in self.subinterval_checks:
                    for poly,error in zip(polys, errors):
                        #The function returns things we should throw out
                        throwOutMask = check(poly, self.mask, error, self.RAND, self.subintervals)
                        #Throw stuff out
                        thrownOutIntervals = newIntervals[throwOutMask]
                        for old_a,old_b in thrownOutIntervals:
                            thrownOuts.append([check.__name__, [old_a,old_b]])
                        self.mask &= ~throwOutMask

            if boundingSize < np.sum(self.mask) and boundingSize < 3: #Something to think about
                self.track_interval_bounded(getBoundingInterval.__name__, [a,b], boundingInterval)
                return [boundingInterval]

            for params in thrownOuts:
                self.track_interval(*params)

            return newIntervals[self.mask]

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

    def track_interval_bounded(self, name, interval, bounding_interval):
            ''' Stores what happened to a given interval when we use a new bounding interval inside it
            Parameters
            ----------
            name : string
                The name of the check or process (Macaulay, Base Case, Too Deep) that solved this interval
            interval: list
                [a,b] where a and b are the lower and upper bound of the interval to track.
            bounding_interval: list
                [a,b] where a and b are the lower and upper bound of the bounding_interval to subdivide into.
            '''
            if not self.polishing:
                self.interval_results[name].append(interval)
            self.current_area += np.prod(interval[1] - interval[0]) - np.prod(bounding_interval[1] - bounding_interval[0])

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
        #plt.figure(dpi=300)
        fig,ax = plt.subplots(1)
        fig.set_size_inches(6.5, 6.5)
        fig.set_dpi(300)
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
                plt.contour(X,Y,Z,levels=[0],colors=contour_colors[i],zorder=20)
            else:
                plt.contour(X,Y,funcs[i](X,Y),levels=[0],colors=contour_colors[i],zorder=20)

                colors = ['w', '#101010', '#b3b3b3','#707070', '#E8E8E8', '#D3D3D3','#202020','#303030']
                #colors = ['w','#c3c3c3', 'C8', '#708090', '#897A57', '#D6C7A4','#73e600','#ccff99']
        #colors = ['w','#d3d3d3', '#708090', '#c5af7d', '#897A57', '#D6C7A4','#73e600','#ccff99']

        if plot_intervals:
            plt.title('Interval Tracking')
            #plt.title('What happened to the intervals')
            #plot results
            i = -1
            ordering = {i:i for i in range(len(self.interval_results))}
            ordering[1] = 9
            ordering[4] = 0
            ordering[0] = 2
            for check in self.interval_results:
                i += 1
                results = self.interval_results[check]
                first = True
                for data in results:
                    a0,b0 = data
                    if first:
                        first = False
                        rect = patches.Rectangle((a0[0],a0[1]),b0[0]-a0[0],b0[1]-a0[1],
                                                 linewidth=.1, edgecolor='k',
                                                 facecolor=colors[i],
                                                 zorder=ordering[i], label=check)
                    else:
                        rect = patches.Rectangle((a0[0],a0[1]),b0[0]-a0[0],b0[1]-a0[1],
                                                 linewidth=.1, edgecolor='k',
                                                 facecolor=colors[i], zorder=ordering[i])
                    ax.add_patch(rect)
            plt.legend()

        #Plot the zeros
        if len(zeros) > 0:
            ax.plot(np.real(zeros[:,0]), np.real(zeros[:,1]),'o',color='#ff0000',markeredgecolor='#ff0000',markersize=3,alpha=0.5,
                 zorder=22)

        if print_plot:
            plt.savefig('intervals.pdf', bbox_inches='tight')
        plt.show()

def getBoundingInterval(coeffs, errors, intervalReductionMethodsToUse):
    numPolys = len(coeffs)
    if numPolys == 0:
        return None
    dim = coeffs[0].ndim
    if numPolys != dim:
        return None
    elif numPolys == 2:
        return getBoundingInterval2D(coeffs, errors, intervalReductionMethodsToUse)
    else:
        return getBoundingIntervalND(coeffs, errors, intervalReductionMethodsToUse)

def mergeIntervals(intervals):
    if len(intervals) == 0:
        return [-1, 1]
    result = [max([interval[0] for interval in intervals]), min([interval[1] for interval in intervals])]
    if result[0] > result[1]:
        return [0,0]
    return result

def boundingIntervalWidthAndBoundCheck(interval):
    MIN_WIDTH = .01
    a,b = interval

    #Bound a,b by [-1,1]
    a = max(min(a,1),-1)
    b = max(min(b,1),-1)
    #If the interval is now empty, return
    if a == b:
        return [0,0]

    #Apply the minnimum width
    width = (b-a)
    if width < MIN_WIDTH:
        center = (a+b)/2
        a = center - MIN_WIDTH/2
        b = center + MIN_WIDTH/2
        #Bound a,b by [-1,1] again it case it is now outside it
        a = max(min(a,1),-1)
        b = max(min(b,1),-1)
    return [a,b]

def improveBound2D(intervals, x_terms, y_terms, consts, errors):
    """Get a basic bound on x from y being in [-1, 1], and on y from
    x being in [-1, 1].

    Parameters
    ----------
    intervals : list
        A list of bounds found for each variable x_i.
    x_terms : list
        A list of coefficients of the x terms of the polynomials in the 2D system.
    y_terms : list
        A list of coefficients of the y terms of the polynomials in the 2D system.
    consts : list
        An array of all the constant terms of the polynomials in our
        system of equations.
    errors : numpy array
        The total error with the x, y and constant terms subtracted off.

    Returns
    -------
    allIntervals : list
        A list of bounds found for each variable x_i, with each new
        bound found added.
    """
    allIntervals = copy(intervals)
    #Get a basic bound on X from y being in [-1, 1]
    if x_terms[0] != 0:
        width = (abs(errors[0]) + abs(y_terms[0])) / abs(x_terms[0])
        center = -consts[0]/x_terms[0]
        allIntervals[0].append([center - width, center + width])
    if x_terms[1] != 0:
        width = (abs(errors[1]) + abs(y_terms[1])) / abs(x_terms[1])
        center = -consts[1]/x_terms[1]
        allIntervals[0].append([center - width, center + width])
    #Get a basic bound on Y from x being in [-1, 1]
    if y_terms[0] != 0:
        width = (abs(errors[0]) + abs(x_terms[0])) / abs(y_terms[0])
        center = -consts[0]/y_terms[0]
        allIntervals[1].append([center - width, center + width])
    if y_terms[1] != 0:
        width = (abs(errors[1]) + abs(x_terms[1])) / abs(y_terms[1])
        center = -consts[1]/y_terms[1]
        allIntervals[1].append([center - width, center + width])

    return allIntervals

def improveBoundND(intervals, A, consts, errors):
    """Get a basic bound on x_i from x_j, i != j, being
    in [-1, 1].

    Parameters
    ----------
    intervals : numpy array
        A list of bounds found for each variable x_i.
    A : numpy array
        An array of all linear terms for the polynomials
        in our system of equations.
    consts : numpy array
        An array of all the constant terms of the polynomials in our
        system of equations.
    errors : numpy array
        The total error with the sum of linear terms and constant terms subtracted off.

    Returns
    -------
    allIntervals : numpy array
        A list of bounds found for each variable x_i, with each new
        bound found added.
    """
    allIntervals = copy(intervals)
    dim = len(allIntervals)
    #Get a basic bound on each variable from the others being in [-1, 1]
    for funcNum in range(dim):
        totalError = sum([abs(num) for num in A[funcNum]]) + abs(errors[funcNum])
        for var in range(dim):
            if abs(A[funcNum][var]) == 0:
                continue
            width = totalError / abs(A[funcNum][var]) - 1
            center = -consts[funcNum]/A[funcNum][var]
            allIntervals[var].append([center - width, center + width])

    return allIntervals

def getBoundingParallelogram2D(intervals, x_terms, y_terms, consts, errors):
    """Get the bounding parallelogram given the x, y and constant terms from
    the 2D system of polynomials.

    Parameters
    ----------
    intervals : list
        A list of bounds found for each variable x_i.
    x_terms : list
        A list of coefficients of the x terms of the polynomials in the 2D system.
    y_terms : list
        A list of coefficients of the y terms of the polynomials in the 2D system.
    consts : list
        An array of all the constant terms of the polynomials in our
        system of equations.
    errors : numpy array
        The total error with the x, y and constant terms subtracted off.

    Returns
    -------
    allIntervals : list
        A list of bounds found for each variable x_i, with each new
        bound found added.
    """
    allIntervals = copy(intervals)
    #Get a bound from the parallelogram
    denom = x_terms[0]*y_terms[1] - x_terms[1]*y_terms[0]
    if denom != 0:
        yCenter = (x_terms[1]*consts[0]-x_terms[0]*consts[1])/denom
        xCenter = (y_terms[0]*consts[1]-y_terms[1]*consts[0])/denom
        yWidth = (abs(x_terms[1]*errors[0]) + abs(x_terms[0]*errors[1]))/abs(denom)
        xWidth = (abs(y_terms[1]*errors[0]) + abs(y_terms[0]*errors[1]))/abs(denom)
        allIntervals[0].append([xCenter - xWidth, xCenter + xWidth])
        allIntervals[1].append([yCenter - yWidth, yCenter + yWidth])

    return allIntervals

def getBoundingParallelogramND(intervals, A, consts, errors):
    """
    Get the bounding parallelogram given the coefficient arrays of the polynomials
    in the system, the constant terms, and the required errors.

    Parameters
    ----------
    intervals : numpy array
        A list of bounds found for each variable x_i.
    A : numpy array
        An array of all linear terms for the polynomials
        in our system of equations.
    consts : numpy array
        An array of all the constant terms of the polynomials in our
        system of equations.
    errors : numpy array
        The total error with the sum of linear terms and constant terms subtracted off.

    Returns
    -------
    allIntervals : numpy array
        A list of bounds found for each variable x_i, with each new
        bound found added.
    """
    allIntervals = copy(intervals)
    dim = len(allIntervals)
    #right hand sides
    B = np.array([-consts+np.array(err_comb) for err_comb in product(*[(e,-e) for e in errors])]).T
    #solve for corners of parallelogram
    #We should probably check to make sure A is full rank first?
    X = la.solve(A,B)
    #find the bounding interval
    a = np.min(X,axis=1)
    b = np.max(X,axis=1)
    for i in range(dim):
        allIntervals[i].append([a[i], b[i]])

    return allIntervals

INTERVAL_REDUCTION_FUNCS_2D = [improveBound2D, getBoundingParallelogram2D]
INTERVAL_REDUCTION_FUNCS_ND = [improveBoundND, getBoundingParallelogramND]

def getBoundingInterval2D(coeffs, errors, intervalReductionMethodsToUse):
    P1 = coeffs[0]
    P2 = coeffs[1]
    xIntervals = []
    yIntervals = []

    #Get Variables for Calculations
    a1 = P1[1,0]
    b1 = P1[0,1]
    c1 = P1[0,0]
    e1 = np.sum(np.abs(P1)) - abs(a1) - abs(b1) - abs(c1) + errors[0]
    a2 = P2[1,0]
    b2 = P2[0,1]
    c2 = P2[0,0]
    e2 = np.sum(np.abs(P2)) - abs(a2) - abs(b2) - abs(c2) + errors[1]

    # Run through all of the interval reduction methods specified by the user.
    for idx in intervalReductionMethodsToUse:
        xIntervals, yIntervals = INTERVAL_REDUCTION_FUNCS_2D[idx]([xIntervals, yIntervals], [a1, a2], [b1, b2], [c1, c2], [e1, e2])

    #Merge the intervals and check the bounds and min width
    xInterval = boundingIntervalWidthAndBoundCheck(mergeIntervals(xIntervals))
    yInterval = boundingIntervalWidthAndBoundCheck(mergeIntervals(yIntervals))

    return np.array([xInterval, yInterval]).T

def getBoundingIntervalND(test_coeffs, tols, intervalReductionMethodsToUse):
    dim = len(test_coeffs)
    allIntervals = [[] for i in range(dim)]

    #Solve the bounding parallelogram
    #create the linear system
    dim = len(test_coeffs)
    A = np.array([coeff[tuple(get_var_list(dim))] for coeff in test_coeffs])
    #compute the error terms
    consts = np.array([coeff[tuple([0]*dim)] for coeff in test_coeffs])
    linear_sums = np.sum(np.abs(A),axis=1)
    err = np.array([np.sum(np.abs(coeff))+tol - fabs(c) - l for coeff,tol,c,l in zip(test_coeffs,tols,consts,linear_sums)])

    # Run through all the interval reduction methods specified by the user.
    for idx in intervalReductionMethodsToUse:
        allIntervals = INTERVAL_REDUCTION_FUNCS_ND[idx](allIntervals, A, consts, err)

    #Merge the intervals and check the bounds and min width
    for i in range(dim):
        allIntervals[i] = boundingIntervalWidthAndBoundCheck(mergeIntervals(allIntervals[i]))

    return np.array(allIntervals).T

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
    if abs(test_coeff[tuple([0]*test_coeff.ndim)]) * 2 > test_sum + tol:
        return False
    else:
        return True

def quadratic_check(test_coeff, mask, tol, RAND, subintervals):
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
    tol: float
        The bound of the sup norm error of the chebyshev approximation.

    Returns
    -------
    mask : list
        A list of the results of each interval. False if the function is guarenteed to never be zero
        in the unit box, True otherwise
    """
    if test_coeff.ndim == 2:
        return quadratic_check_2D(test_coeff, mask, tol, RAND, subintervals)
    elif test_coeff.ndim == 3:
        return quadratic_check_3D(test_coeff, mask, tol, RAND, subintervals)
    else:
        return quadratic_check_nd(test_coeff, mask, tol, RAND, subintervals)

def quadratic_check_2D(test_coeff, mask, tol, RAND, subintervals):
    """One of subinterval_checks

    Finds the min of the absolute value of the quadratic part, and compares to the sum of the
    rest of the terms. There can't be a root if min(extreme_values) > other_sum	or if
    max(extreme_values) < -other_sum. We can short circuit and finish
    faster as soon as we find one value that is < other_sum and one value that > -other_sum.

    Parameters
    ----------
    test_coeff_in : numpy array
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
    if test_coeff.ndim != 2:
        return mask

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

    throwOutMask = mask.copy().reshape(4)
    for i, interval in enumerate(subintervals.reshape(4, 2, 2)):
        if not throwOutMask[i]:
            continue
        throwOutMask[i] = False
        min_satisfied, max_satisfied = False,False
        #Check all the corners
        eval = eval_func(interval[0][0], interval[0][1])
        min_satisfied = min_satisfied or eval < other_sum
        max_satisfied = max_satisfied or eval > -other_sum
        if min_satisfied and max_satisfied:
            continue

        eval = eval_func(interval[1][0], interval[0][1])
        min_satisfied = min_satisfied or eval < other_sum
        max_satisfied = max_satisfied or eval > -other_sum
        if min_satisfied and max_satisfied:
            continue

        eval = eval_func(interval[0][0], interval[1][1])
        min_satisfied = min_satisfied or eval < other_sum
        max_satisfied = max_satisfied or eval > -other_sum
        if min_satisfied and max_satisfied:
            continue

        eval = eval_func(interval[1][0], interval[1][1])
        min_satisfied = min_satisfied or eval < other_sum
        max_satisfied = max_satisfied or eval > -other_sum
        if min_satisfied and max_satisfied:
            continue

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
                    continue
            x = interval[1][0]
            y = -(c[2] + c[4]*x)/cc5
            if interval[0][1] < y < interval[1][1]:
                eval = eval_func(x,y)
                min_satisfied = min_satisfied or eval < other_sum
                max_satisfied = max_satisfied or eval > -other_sum
                if min_satisfied and max_satisfied:
                    continue

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
                    continue

            y = interval[1][1]
            x = -(c[1] + c[4]*y)/cc3
            if interval[0][0] < x < interval[1][0]:
                eval = eval_func(x,y)
                min_satisfied = min_satisfied or eval < other_sum
                max_satisfied = max_satisfied or eval > -other_sum
                if min_satisfied and max_satisfied:
                    continue

        #Check the interior value
        if interval[0][0] < int_x < interval[1][0] and interval[0][1] < int_y < interval[1][1]:
            eval = eval_func(int_x,int_y)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                continue

        # No root possible
        throwOutMask[i] = True
    return throwOutMask.reshape(2, 2)

def quadratic_check_3D(test_coeff, mask, tol, RAND, subintervals):
    """One of subinterval_checks

    Finds the min of the absolute value of the quadratic part, and compares to the sum of the
    rest of the terms.  There can't be a root if min(extreme_values) > other_sum	or if
    max(extreme_values) < -other_sum. We can short circuit and finish
    faster as soon as we find one value that is < other_sum and one value that > -other_sum.

    Parameters
    ----------
    test_coeff_in : numpy array
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
    if test_coeff.ndim != 3:
        return mask

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

    throwOutMask = mask.copy().reshape(8)
    for i, interval in enumerate(subintervals.reshape(8, 2, 3)):
        if not throwOutMask[i]:
            continue
        throwOutMask[i] = False
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
            continue
        eval = eval_func(x1, y0, z0)
        min_satisfied = min_satisfied or eval < other_sum
        max_satisfied = max_satisfied or eval > -other_sum
        if min_satisfied and max_satisfied:
            continue
        eval = eval_func(x0, y1, z0)
        min_satisfied = min_satisfied or eval < other_sum
        max_satisfied = max_satisfied or eval > -other_sum
        if min_satisfied and max_satisfied:
            continue
        eval = eval_func(x0, y0, z1)
        min_satisfied = min_satisfied or eval < other_sum
        max_satisfied = max_satisfied or eval > -other_sum
        if min_satisfied and max_satisfied:
            continue
        eval = eval_func(x1, y1, z0)
        min_satisfied = min_satisfied or eval < other_sum
        max_satisfied = max_satisfied or eval > -other_sum
        if min_satisfied and max_satisfied:
            continue
        eval = eval_func(x1, y0, z1)
        min_satisfied = min_satisfied or eval < other_sum
        max_satisfied = max_satisfied or eval > -other_sum
        if min_satisfied and max_satisfied:
            continue
        eval = eval_func(x0, y1, z1)
        min_satisfied = min_satisfied or eval < other_sum
        max_satisfied = max_satisfied or eval > -other_sum
        if min_satisfied and max_satisfied:
            continue
        eval = eval_func(x1, y1, z1)
        min_satisfied = min_satisfied or eval < other_sum
        max_satisfied = max_satisfied or eval > -other_sum
        if min_satisfied and max_satisfied:
            continue
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
                    continue
            c6y1 = c[6]*y1
            z = -(c5x0_c3+c6y1)/kk9
            if z0 < z < z1:
                eval = eval_func(x0,y1,z)
                min_satisfied = min_satisfied or eval < other_sum
                max_satisfied = max_satisfied or eval > -other_sum
                if min_satisfied and max_satisfied:
                    continue
            c5x1_c3 = c[5]*x1 + c[3]
            z = -(c5x1_c3+c6y0)/kk9
            if z0 < z < z1:
                eval = eval_func(x1,y0,z)
                min_satisfied = min_satisfied or eval < other_sum
                max_satisfied = max_satisfied or eval > -other_sum
                if min_satisfied and max_satisfied:
                    continue
            z = -(c5x1_c3+c6y1)/kk9
            if z0 < z < z1:
                eval = eval_func(x1,y1,z)
                min_satisfied = min_satisfied or eval < other_sum
                max_satisfied = max_satisfied or eval > -other_sum
                if min_satisfied and max_satisfied:
                    continue

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
                    continue
            c6z1 = c[6]*z1
            y = -(c2_c4x0+c6z1)/kk8
            if y0 < y < y1:
                eval = eval_func(x0,y,z1)
                min_satisfied = min_satisfied or eval < other_sum
                max_satisfied = max_satisfied or eval > -other_sum
                if min_satisfied and max_satisfied:
                    continue
            c2_c4x1 = c[2]+c[4]*x1
            y = -(c2_c4x1+c6z0)/kk8
            if y0 < y < y1:
                eval = eval_func(x1,y,z0)
                min_satisfied = min_satisfied or eval < other_sum
                max_satisfied = max_satisfied or eval > -other_sum
                if min_satisfied and max_satisfied:
                    continue
            y = -(c2_c4x1+c6z1)/kk8
            if y0 < y < y1:
                eval = eval_func(x1,y,z1)
                min_satisfied = min_satisfied or eval < other_sum
                max_satisfied = max_satisfied or eval > -other_sum
                if min_satisfied and max_satisfied:
                    continue

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
                    continue
            c5z1 = c[5]*z1
            x = -(c1_c4y0+c5z1)/kk7
            if x0 < x < x1:
                eval = eval_func(x,y0,z1)
                min_satisfied = min_satisfied or eval < other_sum
                max_satisfied = max_satisfied or eval > -other_sum
                if min_satisfied and max_satisfied:
                    continue
            c1_c4y1 = c[1]+c[4]*y1
            x = -(c1_c4y1+c5z0)/kk7
            if x0 < x < x1:
                eval = eval_func(x,y1,z0)
                min_satisfied = min_satisfied or eval < other_sum
                max_satisfied = max_satisfied or eval > -other_sum
                if min_satisfied and max_satisfied:
                    continue
            x = -(c1_c4y1+c5z1)/kk7
            if x0 < x < x1:
                eval = eval_func(x,y1,z1)
                min_satisfied = min_satisfied or eval < other_sum
                max_satisfied = max_satisfied or eval > -other_sum
                if min_satisfied and max_satisfied:
                    continue

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
                    continue
            c2_c4x1 = c[2]+c[4]*x1
            c3_c5x1 = c[3]+c[5]*x1
            y = (-kk9*c2_c4x1 +   c[6]*c3_c5x1)/fix_x_det
            z = (c[6]*c2_c4x1 -    kk8*c3_c5x1)/fix_x_det
            if y0 < y < y1 and z0 < z < z1:
                eval = eval_func(x1,y,z)
                min_satisfied = min_satisfied or eval < other_sum
                max_satisfied = max_satisfied or eval > -other_sum
                if min_satisfied and max_satisfied:
                    continue

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
                    continue
            c1_c4y1 = c[1]+c[4]*y1
            c3_c6y1 = c[3]+c[6]*y1
            x = (-kk9*c1_c4y1 +   c[5]*c3_c6y1)/fix_y_det
            z = (c[5]*c1_c4y1 -    kk7*c3_c6y1)/fix_y_det
            if x0 < x < x1 and z0 < z < z1:
                eval = eval_func(x,y1,z)
                min_satisfied = min_satisfied or eval < other_sum
                max_satisfied = max_satisfied or eval > -other_sum
                if min_satisfied and max_satisfied:
                    continue

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
                    continue
            c1_c5z1 = c[1]+c[5]*z1
            c2_c6z1 = c[2]+c[6]*z1
            x = (-kk8*c1_c5z1 +   c[4]*c2_c6z1)/fix_z_det
            y = (c[4]*c1_c5z1 -    kk7*c2_c6z1)/fix_z_det
            if x0 < x < x1 and y0 < y < y1:
                eval = eval_func(x,y,z1)
                min_satisfied = min_satisfied or eval < other_sum
                max_satisfied = max_satisfied or eval > -other_sum
                if min_satisfied and max_satisfied:
                    continue

        #Add the interior value
        if x0 < int_x < x1 and y0 < int_y < y1 and\
                z0 < int_z < z1:
            eval = eval_func(int_x,int_y,int_z)
            min_satisfied = min_satisfied or eval < other_sum
            max_satisfied = max_satisfied or eval > -other_sum
            if min_satisfied and max_satisfied:
                continue

        # No root possible
        throwOutMask[i] = True
    return throwOutMask.reshape(2, 2, 2)

@memoize
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

def quadratic_check_nd(test_coeff, mask, tol, RAND, subintervals):
    """One of subinterval_checks

    Finds the min of the absolute value of the quadratic part, and compares to the sum of the
    rest of the terms. There can't be a root if min(extreme_values) > other_sum	or if
    max(extreme_values) < -other_sum. We can short circuit and finish
    faster as soon as we find one value that is < other_sum and one value that > -other_sum.

    Parameters
    ----------
    test_coeff_in : numpy array
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

    throwOutMask = mask.copy().reshape(2**dim)
    for k, interval in enumerate(subintervals.reshape(*[2**dim, 2, dim])):
        if not throwOutMask[k]:
            continue
        throwOutMask[k] = False

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
        if not Done:
            throwOutMask[k] = True

    return throwOutMask.reshape(*[2]*dim)

def slices_max_min_check(test_coeff, intervals, tol):
    dim = test_coeff.ndim
    #at first just implement WRT x
    mask = [True]*len(intervals)
    #pull out the slices
    # min_slice =

    for i, interval in enumerate(intervals):
        Done = False
        #check interval

        #no root
        if not Done:
            mask[i] = False

    return mask
