# A collection of functions used in the F4 Macaulay and TVB solvers
import numpy as np
import itertools
from scipy.linalg import qr, solve_triangular
from scipy.misc import comb
#from TVB_Method.root_finder import newton_polish

class TVBError(RuntimeError):
    pass

class Term(object):
    '''
    Terms are just tuples of exponents with the grevlex ordering
    '''
    def __init__(self,val):
        self.val = tuple(val)

    def __repr__(self):
        return str(self.val) + ' with grevlex order'

    def __lt__(self, other, order = 'grevlex'):
        '''
        Redfine less-than according to grevlex
        '''
        if order == 'grevlex': #Graded Reverse Lexographical Order
            if sum(self.val) < sum(other.val):
                return True
            elif sum(self.val) > sum(other.val):
                return False
            else:
                for i,j in zip(reversed(self.val),reversed(other.val)):
                    if i < j:
                        return False
                    if i > j:
                        return True
                return False
        elif order == 'lexographic': #Lexographical Order
            for i,j in zip(self.val,other.val):
                if i < j:
                    return True
                if i > j:
                    return False
            return False
        elif order == 'grlex': #Graded Lexographical Order
            if sum(self.val) < sum(other.val):
                return True
            elif sum(self.val) > sum(other.val):
                return False
            else:
                for i,j in zip(self.val,other.val):
                    if i < j:
                        return True
                    if i > j:
                        return False
                return False

def row_swap_matrix(matrix):
    '''Rearrange the rows of matrix so it is close to upper traingular.

    Parameters
    ----------
    matrix : 2D numpy array
        The matrix whose rows need to be switched

    Returns
    -------
    2D numpy array
        The same matrix but with the rows changed so it is close to upper
        triangular

    Examples
    --------
    >>> utils.row_swap_matrix(np.array([[0,2,0,2],[0,1,3,0],[1,2,3,4]]))
    array([[1, 2, 3, 4],
           [0, 2, 0, 2],
           [0, 1, 3, 0]])
    '''
    leading_mon_columns = list()
    for row in matrix:
        leading_mon_columns.append(np.where(row!=0)[0][0])

    return matrix[np.argsort(leading_mon_columns)]

def clean_zeros_from_matrix(array, accuracy=1.e-10):
    '''Sets all values in the array less than the given accuracy to 0.

    Parameters
    ----------
    array : numpy array
    accuracy : float, optional
        Values in the matrix less than this will be set to 0.

    Returns
    -------
    array : numpy array
        Same array, but with values less than the given accuracy set to 0.
    '''
    array[(array < accuracy) & (array > -accuracy)] = 0
    return array

def slice_top(matrix):
    '''Construct a list of slices needed to slice a matrix into the top 
       corner of another.  

    Parameters
    ----------
    coeff : numpy matrix.
        The matrix of interest.
    Returns
    -------
    slices : list
        Each value of the list is a slice of the matrix in some dimension. 
        It is exactly the size of the matrix.
    '''
    slices = list()
    for i in matrix.shape:
        slices.append(slice(0,i))
    return slices

def slice_bottom(matrix):
    ''' Gets the n-d slices needed to slice a matrix into the bottom corner of another.

    Parameters
    ----------
    coeff : numpy matrix.
        The matrix of interest.
    Returns
    -------
    slices : list
        Each value of the list is a slice of the matrix in some dimension. It is exactly the size of the matrix.
    '''
    slices = list()
    for i in matrix.shape:
        slices.append(slice(-i,None))
    return slices

def match_poly_dimensions(polys):
    '''Matches the dimensions of a list of polynomials.

    Parameters
    ----------
    polys : list
        Polynomials of possibly different dimensions.

    Returns
    -------
    new_polys : list
        The same polynomials but of the same dimensions.
    '''
    dim = max(poly.dim for poly in polys)
    new_polys = list()
    for poly in polys:
        if poly.dim != dim:
            coeff_shape = list(poly.shape)
            for i in range(dim - poly.dim):
                coeff_shape.insert(0,1)
            poly.__init__(poly.coeff.reshape(coeff_shape))
        new_polys.append(poly)
    return new_polys

def match_size(a,b):
    '''
    Matches the shape of two matrixes.

    Parameters
    ----------
    a, b : ndarray
        Matrixes whose size is to be matched.

    Returns
    -------
    a, b : ndarray
        Matrixes of equal size.
    '''
    new_shape = np.maximum(a.shape, b.shape)

    a_new = np.zeros(new_shape)
    a_new[slice_top(a)] = a
    b_new = np.zeros(new_shape)
    b_new[slice_top(b)] = b
    return a_new, b_new


def get_var_list(dim):
    '''Return a list of tuples corresponding to the variables [x_1, x_2, ..., x_n].
       The tuple for x_1 is (1,0,0,...,0), and for x_i the 1 is in the ith slot.
    '''
    _vars = []
    var = [0]*dim
    for i in range(dim):
        var[i] = 1
        _vars.append(tuple(var))
        var[i] = 0
    return _vars


def mon_combosHighest(mon, numLeft, spot = 0):
    '''Find all the monomials of a given degree and returns them. Works recursively.

    Very similar to mon_combos, but only returns the monomials of the desired degree.

    Parameters
    --------
    mon: list
        A list of zeros, the length of which is the dimension of the desired monomials. Will change
        as the function searches recursively.
    numLeft : int
        The degree of the monomials desired. Will decrease as the function searches recursively.
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
        mon[spot] = numLeft
        answers.append(mon.copy())
        return answers
    if numLeft == 0: #Nothing else can be added.
        answers.append(mon.copy())
        return answers
    temp = mon.copy() #Quicker than copying every time inside the loop.
    for i in range(numLeft+1): #Recursively add to mon further down.
        temp[spot] = i
        answers += mon_combosHighest(temp, numLeft-i, spot+1)
    return answers

def mon_combos(mon, numLeft, spot = 0):
    '''Finds all the monomials up to a given degree and returns them. Works recursively.

    Parameters
    --------
    mon: list
        A list of zeros, the length of which is the dimension of the desired monomials. Will change
        as the function searches recursively.
    numLeft : int
        The degree of the monomials desired. Will decrease as the function searches recursively.
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
        for i in range(numLeft+1):
            mon[spot] = i
            answers.append(mon.copy())
        return answers
    if numLeft == 0: #Nothing else can be added.
        answers.append(mon.copy())
        return answers
    temp = mon.copy() #Quicker than copying every time inside the loop.
    for i in range(numLeft+1): #Recursively add to mon further down.
        temp[spot] = i
        answers += mon_combos(temp, numLeft-i, spot+1)
    return answers

def sort_polys_by_degree(polys, ascending = True):
    '''Sorts the polynomials by their degree.

    Parameters
    ----------
    polys : list.
        A list of polynomials.
    ascending : bool
        Defaults to True. If True the polynomials are sorted in order of ascending degree. If False they
        are sorted in order of descending degree.
    Returns
    -------
    sorted_polys : list
        A list of the same polynomials, now sorted.
    '''
    degs = [poly.degree for poly in polys]
    argsort_list = np.argsort(degs)
    sorted_polys = list()
    for i in argsort_list:
        sorted_polys.append(polys[i])
    if ascending:
        return sorted_polys
    else:
        return sorted_polys[::-1]

def makePolyCoeffMatrix(inputString):
    '''
    Takes a string input of a polynomaial and returns the coefficient matrix for it. Usefull for making things of high
    degree of dimension so you don't have to make it by hand.

    All strings must be of the following syntax. Ex. '3x0^2+2.1x1^2*x2+-14.73x0*x2^3'

    1. There can be no spaces.
    2. All monomials must be seperated by a '+'. If the coefficient of the monomial is negative then the '-' sign
       should come after the '+'. This is not needed for the first monomial.
    3. All variables inside a monomial are seperated by a '*'.
    4. The power of a variable in a monomial is given folowing a '^' sign.
    '''
    matrixSpots = list()
    coefficients = list()
    for monomial in inputString.split('+'):
        coefficientString = monomial[:first_x(monomial)]
        if coefficientString == '-':
            coefficient = -1
        elif coefficientString == '':
            coefficient = 1
        else:
            coefficient = float(coefficientString)
        mons = monomial[first_x(monomial):].split('*')
        matrixSpot = [0]
        for mon in mons:
            stuff = mon.split('^')
            if len(stuff) == 1:
                power = 1
            else:
                power = int(stuff[1])
            if stuff[0] == '':
                varDegree = -1
            else:
                varDegree = int(stuff[0][1:])
            if varDegree != -1:
                if len(matrixSpot) <= varDegree:
                    matrixSpot = np.append(matrixSpot, [0]*(varDegree - len(matrixSpot)+1))
                matrixSpot[varDegree] = power
        matrixSpots.append(matrixSpot)
        coefficients.append(coefficient)
    #Pad the matrix spots so they are all the same length.
    length = max(len(matrixSpot) for matrixSpot in matrixSpots)
    for i in range(len(matrixSpots)):
        matrixSpot = matrixSpots[i]
        if len(matrixSpot) < length:
            matrixSpot = np.append(matrixSpot, [0]*(length - len(matrixSpot)))
            matrixSpots[i] = matrixSpot
    matrixSize = np.maximum.reduce([matrixSpot for matrixSpot in matrixSpots])
    matrixSize = matrixSize + np.ones_like(matrixSize)
    matrixSize = matrixSize[::-1] #So the variables are in the right order.
    matrix = np.zeros(matrixSize)
    for i in range(len(matrixSpots)):
        matrixSpot = matrixSpots[i][::-1] #So the variables are in the right order.
        coefficient = coefficients[i]
        matrix[tuple(matrixSpot)] = coefficient
    return matrix

def check_zeros(zeros, polys, real=True, tol=1e-5):
    """
    Check whether 'zeros' are, indeed, all the zeros of 'polys', and how many are 
          outside the sup-norm unit ball |z|_\infty < 1.

    Parameters
    ----------
    zeros : list
        Supposed roots (usually found using the root finder).
    polys : list
        Polynomials that 'zeros' should be roots of.
    real : bool
        Whether to check that the bad zeros are real (real=True) or 
         not to check (real=False)

    Prints
    ----------
    The number of correct zeroes found with the total number.
    The number of incorrect zeros that were out of range or nonreal.

    Returns
    ----------
    A list of bad zeros.


    """
    correct = 0
    outOfRange = 0
    bad = set()
    bad_inrange = []
    if zeros != -1:
        for zero in zeros:
            good = True
            for poly in polys:
                v = poly.evaluate_at(zero)
                if np.abs(v) > tol:
                    good = False
                    bad.add(tuple(zero))
            if good:
                correct += 1

    bad_list = [np.array(zero) for zero in bad]
    for zero in bad_list:
        if (np.abs(zero) > 1).any():
            outOfRange += 1
        elif real and np.any(np.abs(np.imag(zero))>tol):
            outOfRange += 1
            real_status = 'not real or'
        else:
            bad_inrange.append(zero)
            
    print("{} zeros are correct to {}, out of {} total zeros.".format(correct, tol,len(zeros)))
    print("{} are bad, but {} of these were {} out of range (expected to be bad).".format(len(bad), outOfRange, real_status))
    print("{} might be lost".format(len(bad_inrange)))

    better = []
    diff = []
    newt_bad = set()
    for zero in bad_inrange:
        newt_zero = newton_polish(polys,zero,tol=1e-10)
        better.append(newt_zero)
        diff.append(zero - newt_zero)
        for poly in polys:
            v = poly.evaluate_at(newt_zero)
            if np.abs(v) > tol:
                newt_bad.add(tuple(newt_zero))
        
    print('{} seem to be lost after newton polishing'.format(len(newt_bad)))
    #print("Differences between the bad zeros and the polished ones are {}".format(diff)) 
                    
    return bad_inrange, newt_bad

    

    
