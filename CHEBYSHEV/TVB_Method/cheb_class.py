import numpy as np
import itertools
from numpy.polynomial import chebyshev as cheb

"""
Module for defining the class of Chebyshev polynomials, as well as various related 
classes and methods, including:

   Classes:
   -------

        Polynomial: Superclass for MultiPower and MultiCheb. Contains methods and 
             attributes applicable to both subclasses

        MultiCheb: Chebyshev polynomials in arbitrary dimension.

        Term: Terms are just tuples of exponents with the degrevlex ordering

    Methods:
    --------

        match_poly_dimensions(polys): Matches the dimensions of a list of polynomials.

        mon_combos_highest(mon, numLeft): Find all the monomials of a given degree and 
             returns them. Works recursively.

        mon_combos(mon, numLeft): Finds all the monomials _up to_ a given degree and 
             returns them. Works recursively.

        sort_polys_by_degree(polys): Sorts the polynomials by their degree.

        get_var_list(dim): Return a list of tuples corresponding to the variables 
                [x_1, x_2, ..., x_n]. The tuple for x_1 is (1,0,0,...,0), and for x_i 
                the 1 is in the ith slot.

        slice_bottom(arr):Gets the nd slices needed to slice an array into the bottom 
                corner of another.  There is probably a better (vectorized) way to do this.

        slice_top(arr): Construct a list of slices needed to put an array into the upper 
                left corner of another.  There is probably a better way to do this.

        match_size(a,b): Reshape two coefficient ndarrays to have the same shape.
    
        makePolyCoeffMatrix(inputString): Take a string input of a polynomaial and 
              return the coefficient matrix for it. Usefull for making things of high 
              degree of dimension so you don't have to make it by hand.


"""

class Polynomial(object):
    '''
    Superclass for MultiPower and MultiCheb. Contains methods and attributes
    that are applicable to both subclasses.

    Attributes
    ----------
    coeff
        The coefficient matrix represented in the object.
    dim
        The number of dimensions of the coefficient matrix
    order
        Ordering type given as a string
    shape
        The shape of the coefficient matrix
    lead_term
        The polynomial term with the largest total degree
    degree
        The total degree of the lead_term
    lead_coeff
        The coeff of the lead_term

    Parameters
    ----------
    coeff : ndarray
        Coefficients of the polynomial
    order : string
    lead_term : Tuple
        Default is None. Accepts tuple or tuple-like inputs
    clean_zeros : bool
        Default is True. If True, all extra rows, columns, etc of all zeroes are
        removed from matrix of coefficients.

    Methods
    -------
    clean_coeff
        Removes extra rows, columns, etc of zeroes from end of matrix of coefficients
    match_size
        Matches the shape of two matrices.
    monomialList
        Creates a list of monomials that make up the polynomial in degrevlex order.
    monSort
        Calls monomial list.
    update_lead_term
        Finds the lead_term of a polynomial
    evaluate_at
        Evaluates a polynomial at a certain point.
    __eq__
        Checks if two polynomials are equal.
    __ne__
        Checks if two polynomials are not equal.

    '''
    def __init__(self, coeff, order='degrevlex', lead_term=None, clean_zeros=True):
        '''
        order : string
              Term order to use for the polynomial.  degrevlex is default.  
              Currently no other order is implemented.
        '''
        if isinstance(coeff,np.ndarray):
            self.coeff = coeff
        elif isinstance(coeff,str):
            self.coeff = makePolyCoeffMatrix(coeff)
        else:
            raise ValueError('coeff must be an np.array or a string!')
        if clean_zeros:
            self.clean_coeff()
        self.dim = self.coeff.ndim
        self.order = order
        self.jac = None
        self.shape = self.coeff.shape
        if lead_term is None:
            self.update_lead_term()
        else:
            self.lead_term = tuple(lead_term)
            self.degree = sum(self.lead_term)
            self.lead_coeff = self.coeff[self.lead_term]

    def clean_coeff(self):
        """
        Remove 0s on the outside of the coeff matrix. Acts in place.
        """

        for axis in range(self.coeff.ndim):
            change = True
            while change:
                change = False
                if self.coeff.shape[axis] == 1:
                    continue
                axisCount = 0
                slices = list()
                for i in self.coeff.shape:
                    if axisCount == axis:
                        s = slice(i-1,i)
                    else:
                        s = slice(0,i)
                    slices.append(s)
                    axisCount += 1
                if np.sum(abs(self.coeff[slices])) == 0:
                    self.coeff = np.delete(self.coeff,-1,axis=axis)
                    change = True

    def update_lead_term(self):
        """
        Update the lead term of the polynomial.
        """
        
        non_zeros = list()
        for i in zip(*np.where(self.coeff != 0)):
            non_zeros.append(Term(i))
        if len(non_zeros) != 0:
            self.lead_term = max(non_zeros).val
            self.degree = sum(self.lead_term)
            self.lead_coeff = self.coeff[self.lead_term]
        else:
            self.lead_term = None
            self.lead_coeff = 0
            self.degree = -1

    def evaluate_at(self, point):
        '''
        Evaluate the polynomial at 'point'. This method is overridden
        by the MultiPower and MultiCheb classes, so this definition only
        checks if the polynomial can be evaluated at the given point.

        Parameters
        ----------
        point : array-like
            the point at which to evaluate the polynomial

        Returns
        -------
        evaluate_at : complex
            value of the polynomial at the given point
        '''
        if len(point) != len(self.coeff.shape):
            raise ValueError('Cannot evaluate polynomial in {} variables at point {}'\
            .format(self.dim, point))

    def grad(self, point):
        '''
        Evaluates the gradient of the polynomial at 'point'. This method is 
        overridden by the MultiPower and MultiCheb classes, so this definition only
        checks if the polynomial can be evaluated at the given point.

        Parameters
        ----------
        point : array-like
            the point at which to evaluate the polynomial

        Returns
        -------
        grad : ndarray
            Gradient of the polynomial at the given point.
        '''
        if len(point) != len(self.coeff.shape):
            raise ValueError('Cannot evaluate polynomial in {} variables at point {}'\
            .format(self.dim, point))

    def __eq__(self,other):
        '''
        Check if coeff matrices of 'self' and 'other' are the same.
        '''
        
        if self.shape != other.shape:
            return False
        return np.allclose(self.coeff, other.coeff)

    def __ne__(self,other):
        '''
        Check if coeff matrices of 'self' and 'other' are not the same.
        '''
        return not (self == other)


###############################################################################

#### MULTI_CHEB ###############################################################

class MultiCheb(Polynomial):
    """
    A Chebyshev polynomial.

    Attributes
    ----------
    coeff: ndarray
        A tensor of coefficients whose i_1,...,i_{dim} entry 
        corresponds to the coefficient of the term 
        T_{i_1}(x_1)...T_{i_{dim}}(x_{dim}) 
    dim:
        The number of variables, dimension of polynomial.
    order: string
        Term order 
    shape: tuple of ints
        The shape of the coefficient array.
    lead_term:
        The term with the largest total degree. 
    degree: int
        The total degree of lead_term.
    lead_coeff
        The coefficient of the lead_term.
    terms : int
        Highest term of single-variable polynomials. The polynomial has  
               degree at most terms+1 in each variable.



    Parameters
    ----------
    coeff : list(terms**dim) or np.array ([terms,] * dim)
        coefficents in given ordering.
    order : string
        Term order for Groebner calculations. Default = 'degrevlex'
    lead_term : list
        The index of the current leading coefficent.  If None, this is computed at initialization.
    clean_zeros: boolean
        If True, strip off any rows or columns of zeros on the outside of the coefficient array.

    Methods
    -------

    __add__
        Add two MultiCheb polynomials.
    __sub__
        Subtract two MultiCheb polynomials.
    mon_mult
        Multiply a MultiCheb monomial by a MultiCheb polynomial.
    evaluate_at
        Evaluate a MultiCheb polynomial at a point.

    """
    def __init__(self, coeff, order='degrevlex', lead_term=None, clean_zeros = True):
        super(MultiCheb, self).__init__(coeff, order, lead_term, clean_zeros)

    def __add__(self,other):
        '''
        Addition of two MultiCheb polynomials.

        Parameters
        ----------
        other : MultiCheb

        Returns
        -------
        MultiCheb
            The sum of the coeff of self and coeff of other.

        '''
        if self.shape != other.shape:
            new_self, new_other = match_size(self.coeff,other.coeff)
        else:
            new_self, new_other = self.coeff, other.coeff

        return MultiCheb(new_self + new_other,clean_zeros = False)

    def __sub__(self,other):
        '''
        Subtraction of two MultiCheb polynomials.

        Parameters
        ----------
        other : MultiCheb

        Returns
        -------
        MultiCheb
            The coeff values are the result of self.coeff - other.coeff.
        '''
        if self.shape != other.shape:
            new_self, new_other = match_size(self.coeff,other.coeff)
        else:
            new_self, new_other = self.coeff, other.coeff
        return MultiCheb((new_self - (new_other)), clean_zeros = False)

    
    
    def _fold_in_i_dir(coeff_array, dim, fdim, size_in_fdim, fold_idx):
        """Find coeffs corresponding to T_|m-n| (referred to as 'folding' in
        some of this documentation) when multiplying a monomial times
        a Chebyshev polynomial.

        Multiplying the monomial T_m(x_i) times T_n(x_i) gives 
        (T_{m+n}(x_i) + T_{|n-m|}(x_i))/2
        So multipying T_m(x_i) times polynomial P with coefficients 
        in coeff_array results in a new coefficient array sol that has 
        coeff_array in the bottom right corner plus a 'folded' copy of 
        coeff_array in locations corresponding to |n-m|.  This method 
        returns the folded part (not dividing by 2)

        Parameters
        ----------
        coeff_array : ndarray
            coefficients of the polynomial.
        dim : int
            The number of dimensions in coeff_array.
        fdim : int
            The dimension being folded ('i' in the explanation above)
        size_in_fdim : int
            The size of the solution matrix in the dimension being folded.
        fold_idx : int
            The index to fold around ('m' in the explanation above)

        Returns
        -------
        sol : ndarray

        """
        if fold_idx == 0:
            return coeff_array

        target = np.zeros_like(coeff_array) # Array of zeroes in which to insert
                                         # the new values.

        ## Compute the n-m part for n >= m

        # slice source and target in the dimension of interest (i = fdim)
        target_slice = slice(0,size_in_fdim-fold_idx,None) # n-m for n>=m
        source_slice = slice(fold_idx,size_in_fdim,None)   # n for n>=m

        # indexers have a slice index for every dimension.
        source_indexer = [slice(None)]*dim
        source_indexer[fdim] = source_slice
        target_indexer = [slice(None)]*dim
        target_indexer[fdim] = target_slice

        # Put the appropriately indexed source into the target
        target[target_indexer] = coeff_array[source_indexer]

        ## Compute the m-n part for n < m

        # slice source and target in the dimension of interest (i = fdim)
        target_slice = slice(fold_idx, 0 , -1)   # m-n for n < m
        source_slice = slice(None, fold_idx, None) # n for n < m

        # indexers have a slice index for every dimension.
        source_indexer = [slice(None)]*dim
        source_indexer[fdim] = source_slice
        target_indexer = [slice(None)]*dim
        target_indexer[fdim] = target_slice

        # Add the appropriately indexed source to the target
        target[target_indexer] += coeff_array[source_indexer]

        return target


    def _mon_mult1(coeff_array, monom, mult_idx):
        """
        Monomial multiply in one dimension, that is, T_m(x_i) * P(x_1,...,x_n), 
        where P is a Chebyshev polynomial and T_m(x_i) is a Chebyshev monomial 
        in the lone variable x_i.

        Parameters
        ----------
        coeff_array : array_like
            Coefficients of a Chebyshev polynomial (denoted P above).

        monom : tuple of ints
            Index  of the form (0,0,...,0,m,0...,0)  of a 
            monomial of one variable

        mult_idx : int
            The location (denoted i above) of the non-zero value in monom. 

        Returns
        -------
        ndarray
            Coeffs of the new polynomial T_m(x_i)*P. 

        """
                                                
        p1 = np.zeros(coeff_array.shape + monom)
        p1[slice_bottom(coeff_array)] = coeff_array  # terms corresp to T_{m+n}

        largest_idx = [i-1 for i in coeff_array.shape]
        new_shape = [max(i,j) for i,j in
                     itertools.zip_longest(largest_idx, monom, fillvalue = 0)]
        if coeff_array.shape[mult_idx] <= monom[mult_idx]:
            add_a = [i-j for i,j in itertools.zip_longest(new_shape, largest_idx, fillvalue = 0)]
            add_a_list = np.zeros((len(new_shape),2))
            #change the second column to the values of add_a and add_b.
            add_a_list[:,1] = add_a
            #use add_a_list and add_b_list to pad each polynomial appropriately.
            coeff_array = np.pad(coeff_array,add_a_list.astype(int),'constant')

        number_of_dim = coeff_array.ndim
        shape_of_self = coeff_array.shape

        if monom[mult_idx] != 0:
            coeff_array = MultiCheb._fold_in_i_dir(coeff_array,number_of_dim,
                                                   mult_idx, shape_of_self[mult_idx],
                                                   monom[mult_idx])
        if p1.shape != coeff_array.shape:
            monom = [i-j for i,j in zip(p1.shape,coeff_array.shape)]

            result = np.zeros(np.array(coeff_array.shape) + monom)
            result[slice_top(coeff_array)] = coeff_array
            coeff_array = result
        Pf = p1 + coeff_array
        return .5*Pf

    def mon_mult(self, monom, return_type = 'Poly'):
        """
        Multiply a Chebyshev polynomial by a monomial

        Parameters
        ----------
        monom : tuple of ints
            The index of the monomial to multiply self by.
        return_type : str
            If 'Poly' then returns a polynomial object.

        Returns
        -------
        MultiCheb object if return_type is 'Poly'.
        ndarray if return_type is "Matrix".

        """
        coeff_array = self.coeff
        monom_zeros = np.zeros(len(monom),dtype = int)
        for i in range(len(monom)):
            monom_zeros[i] = monom[i]
            coeff_array = MultiCheb._mon_mult1(coeff_array, monom_zeros, i)
            monom_zeros[i] = 0
        if return_type == 'Poly':
            return MultiCheb(coeff_array, lead_term = self.lead_term + np.array(monom), clean_zeros = False)
        elif return_type == 'Matrix':
            return coeff_array

    def evaluate_at(self, point):
        '''
        Evaluate the polynomial at 'point'.

        Parameters
        ----------
        point : array-like
                point at which to evaluate the polynomial

        Returns
        -------
        c : complex
            value of the polynomial at the given point
        '''
        super(MultiCheb, self).evaluate_at(point)

        c = self.coeff
        n = len(c.shape)
        c = cheb.chebval(point[0],c)
        for i in range(1,n):
            c = cheb.chebval(point[i],c,tensor=False)
        return c

    def grad(self, point):
        '''
        Evaluates the gradient of the polynomial at the given point.

        Parameters
        ----------
        point : array-like
            the point at which to evaluate the polynomial

        Returns
        -------
        out : ndarray
            Gradient of the polynomial at the given point.
        '''
        super(MultiCheb, self).evaluate_at(point)

        out = np.empty(self.dim,dtype="complex_")
        if self.jac is None:
            jac = list()
            for i in range(self.dim):
                jac.append(cheb.chebder(self.coeff,axis=i))
            self.jac = jac
        spot = 0
        for i in self.jac:
            out[spot] = chebvalnd(point,i)
            spot+=1
        return out

###############################################################################

def chebvalnd(x,c):
    """
    Evaluate a MultiCheb object at a point x

    Parameters
    ----------
    x : ndarray
        Point to evaluate at
    c : ndarray
        Tensor of Chebyshev coefficients

    Returns
    -------
    c : float
        Value of the MultiCheb polynomial at x
    """
    x = np.array(x)
    n = len(c.shape)
    c = cheb.chebval(x[0],c)
    for i in range(1,n):
        c = cheb.chebval(x[i],c,tensor=False)
    return c

def polyList(deg,dim,Type = 'random'):
    """
    Creates random polynomials for root finding.

    Parameters
    ----------
    deg : int
        Desired degree of the polynomials.
    dim : int
        Desired number of dimensions for the polynomials
    Type : str
        Either 'random' or 'int.

    Returns
    ----------
    polys : list
        polynomial objects that are used to test the root finding.

    """
    deg += 1
    polys = []
    if Type == 'random':
        for i in range(dim):
            polys.append(np.random.random_sample(deg*np.ones(dim, dtype = int)))
    elif Type == 'int':
        Range = 10
        for i in range(dim):
            polys.append(np.random.randint(-Range,Range,deg*np.ones(dim, dtype = int)))
    for i,j in np.ndenumerate(polys[0]):
        if np.sum(i) >= deg:
            for h in range(len(polys)):
                polys[h][i] = 0
    for i in range(len(polys)):
        polys[i] = MultiCheb(polys[i])
    return polys




#############   Cheb Utils  ####################3


class TVBError(RuntimeError):
    pass

class Term(object):
    '''
    Terms are just tuples of exponents with the degrevlex ordering
    '''
    def __init__(self,val):
        self.val = tuple(val)

    def __repr__(self):
        return str(self.val) + ' with degrevlex order'

    def __lt__(self, other, order = 'degrevlex'):
        '''
        Redfine less-than according to term order
        '''
        if order == 'degrevlex': #Graded Reverse Lexographical Order
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




def mon_combos_highest(mon, numLeft, spot = 0):
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
        answers += mon_combos_highest(temp, numLeft-i, spot+1)
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


def match_size(a,b):
    '''
    Matches the shape of two ndarrays.

    Parameters
    ----------
    a, b : ndarray
        Arrays whose size is to be matched.

    Returns
    -------
    a, b : ndarray
        Arrays of equal size.
    '''
    new_shape = np.maximum(a.shape, b.shape)

    a_new = np.zeros(new_shape)
    a_new[slice_top(a)] = a
    b_new = np.zeros(new_shape)
    b_new[slice_top(b)] = b
    return a_new, b_new


def slice_top(arr):
    '''Construct a list of slices needed to put an array into the upper left 
       corner of another.  

    Parameters
    ----------
    arr : ndarray
        The array of interest.
    Returns
    -------
    slices : list
        Each value of the list is a slice of the array in some dimension. 
        It is exactly the size of the array.
    '''
    slices = list()
    for i in arr.shape:
        slices.append(slice(0,i))
    return slices

def slice_bottom(arr):
    ''' Gets the n-d slices needed to slice an array into the bottom 
        corner of another.

    Parameters
    ----------
    arr : ndarray
        The array of interest.

    Returns
    -------
    slices : list
        Each value of the list is a slice of the array in some dimension. 
        It is exactly the size of the array.
    '''
    slices = list()
    for i in arr.shape:
        slices.append(slice(-i,None))
    return slices


def get_var_list(dim):
    '''Return a list of tuples corresponding to the 
       variables [x_1, x_2, ..., x_n]. The tuple for x_1 
       is (1,0,0,...,0), and for x_i the 1 is in the ith slot.
    '''
    
    _vars = []
    var = [0]*dim
    for i in range(dim):
        var[i] = 1
        _vars.append(tuple(var))
        var[i] = 0
    return _vars


