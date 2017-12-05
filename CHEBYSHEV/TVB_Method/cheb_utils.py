# A collection of functions used in the F4 Macaulay and TVB solvers
import numpy as np
import itertools
from scipy.linalg import qr, solve_triangular
from scipy.misc import comb

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
        Redefine less-than according to grevlex, or chosen ordering
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
        The same matrix but with the row orer changed so it is close to upper
        triangular

    Examples
    --------
    >>> utils.row_swap_matrix(np.array([[0,2,0,2],
                                        [0,1,3,0],
                                        [1,2,3,4]]))
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
    ''' Gets the n-d slices needed to slice a matrix into the top corner of another.

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
        Each value of the list is a slice of the matrix in some dimension. 
        It is exactly the size of the matrix.
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
        The same polynomials but of homogenized dimensions.
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
    Matches the shape of two matrices.

    Parameters
    ----------
    a, b : ndarray
        Matrices whose size is to be matched.

    Returns
    -------
    a, b : ndarray
        Matrices of equal size.
    '''
    new_shape = np.maximum(a.shape, b.shape)

    a_new = np.zeros(new_shape)
    a_new[slice_top(a)] = a
    b_new = np.zeros(new_shape)
    b_new[slice_top(b)] = b
    return a_new, b_new

def get_var_list(dim):
    '''Returns a list of the variables [x_1, x_2, ..., x_n] as tuples.'''
    _vars = []
    var = [0]*dim
    for i in range(dim):
        var[i] = 1
        _vars.append(tuple(var))
        var[i] = 0
    return _vars

def mon_combos(mon, num_left, spot = 0):
    '''Finds all the monomials up to a given degree and returns them. Works recursively.

    Parameters
    --------
    mon: list
        A list of zeros, the length of which is the dimension of the desired monomials. Will change
        as the function searches recursively.
    num_left : int
        The degree of the monomials desired. Will decrease as the function searches recursively.
    spot : int
        The current position in the list the function is iterating through. Defaults to 0,
        but increases in each step of the recursion.

    Returns
    -----------
    answers : list
        A list of all the monomials.
    '''
    answers = list()
    if len(mon) == spot+1: #We are at the end of mon, no more recursion.
        for i in range(num_left+1):
            mon[spot] = i
            answers.append(mon.copy())
        return answers
    if num_left == 0: #Nothing else can be added.
        answers.append(mon.copy())
        return answers
    temp = mon.copy() #Quicker than copying every time inside the loop.
    for i in range(num_left+1): #Recursively add to mon further down.
        temp[spot] = i
        answers += mon_combos(temp, num_left-i, spot+1)
    return answers

def mon_combos_highest(mon, num_left, spot = 0):
    '''Finds all the monomials of a given degree and returns them. Works recursively.

    Very similar to mon_combos, but only returns the monomials of the desired degree.

    Parameters
    --------
    mon: list
        A list of zeros, the length of which is the dimension of the desired monomials. 
        Will change as the function searches recursively.
    num_left : int
        The degree of the monomials desired. Will decrease as the function searches recursively.
    spot : int
        The current position in the list the function is iterating through. Defaults to 0,
        but increases in each step of the recursion.

    Returns
    -----------
    answers : list
        A list of all the monomials of highest degree.
    '''
    answers = list()
    if len(mon) == spot+1: #We are at the end of mon, no more recursion.
        mon[spot] = num_left
        answers.append(mon.copy())
        return answers
    if num_left == 0: #Nothing else can be added.
        answers.append(mon.copy())
        return answers
    temp = mon.copy() #Quicker than copying every time inside the loop.
    for i in range(num_left+1): #Recursively add to mon further down.
        temp[spot] = i
        answers += mon_combos_highest(temp, num_left-i, spot+1)
    return answers

def sort_polys_by_degree(polys, ascending = True):
    '''Sorts the polynomials by their degree.

    Parameters
    ----------
    polys : list.
        A list of polynomials.
    ascending : bool
        Defaults to True. If True the polynomials are sorted in order of ascending degree.
        If False they are sorted in order of descending degree.
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

def make_poly_coeff_matrix(input_string):
    '''
    Takes a string input of a polynomaial and returns the coefficient matrix for it.
    Usefull for making things of high degree of dimension so you don't have to make it by hand.

    All strings must be of the following syntax. Ex. '3x0^2+2.1x1^2*x2+-14.73x0*x2^3'

    1. There can be no spaces.
    2. All monomials must be seperated by a '+'. If the coefficient of the monomial is negative
         then the '-' sign should come after the '+'. This is not needed for the first monomial.
    3. All variables inside a monomial are seperated by a '*'.
    4. The power of a variable in a monomial is given folowing a '^' sign.
    '''
    matrix_spots = list()
    coefficients = list()
    for monomial in input_string.split('+'):
        coefficient_string = monomial[:first_x(monomial)]
        if coefficient_string == '-':
            coefficient = -1
        elif coefficient_string == '':
            coefficient = 1
        else:
            coefficient = float(coefficient_string)
        mons = monomial[first_x(monomial):].split('*')
        matrix_spot = [0]
        for mon in mons:
            variables = mon.split('^')
            if len(variables) == 1:
                power = 1
            else:
                power = int(variables[1])
            if variables[0] == '':
                var_degree = -1
            else:
                var_degree = int(variables[0][1:])
            if var_degree != -1:
                if len(matrix_spot) <= var_degree:
                    matrix_spot = np.append(matrix_spot, [0]*(var_degree - len(matrix_spot)+1))
                matrix_spot[var_degree] = power
        matrix_spots.append(matrix_spot)
        coefficients.append(coefficient)
    #Pad the matrix spots so they are all the same length.
    length = max(len(matrix_spot) for matrix_spot in matrix_spots)
    for i in range(len(matrix_spots)):
        matrix_spot = matrix_spots[i]
        if len(matrix_spot) < length:
            matrix_spot = np.append(matrix_spot, [0]*(length - len(matrix_spot)))
            matrix_spots[i] = matrix_spot
    matrix_size = np.maximum.reduce([matrix_spot for matrix_spot in matrix_spots])
    matrix_size = matrix_size + np.ones_like(matrix_size)
    matrix_size = matrix_size[::-1] #So the variables are in the right order.
    matrix = np.zeros(matrix_size)
    for i in range(len(matrix_spots)):
        matrix_spot = matrix_spots[i][::-1] #So the variables are in the right order.
        coefficient = coefficients[i]
        matrix[tuple(matrix_spot)] = coefficient
    return matrix

def check_zeros(zeros, polys):
    """
    Function that checks the zeros.

    Parameters
    ----------
    zeros : list
        The list of roots found using the root finder.
    polys : list
        The polynomials that were used for root finding.

    Prints
    ----------
    The number of correct zeroes found with the total number.
    The number of zeros that were out of range.

    """
    correct = 0
    out_of_range = 0
    if zeros != -1:
        for zero in zeros:
            good = True
            for poly in polys:
                if not np.isclose(0, poly.evaluate_at(zero), atol = 1.e-3):
                    good = False
                    if (np.abs(zero) > 1).any():
                        out_of_range += 1
                    break
            if good:
                correct += 1
    print("{} ZEROS ARE CORRECT OUT OF {}".format(correct, len(zeros)))
<<<<<<< HEAD
    print("{} of them were out of range".format(out_of_range))
=======
    print("{} of them were out of range".format(outOfRange))
>>>>>>> 95a66c69ca1a260d01326921302cae7b8dd68a47
