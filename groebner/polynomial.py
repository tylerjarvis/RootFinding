import numpy as np
from scipy.signal import fftconvolve, convolve
import itertools
from groebner.utils import Term, makePolyCoeffMatrix
from numpy.polynomial import chebyshev as cheb
from numpy.polynomial import polynomial as poly
import math
from groebner.utils import match_size

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
    def __init__(self, coeff, order='degrevlex', lead_term=None, clean_zeros = True):
        '''
        terms, int- number of Chebyshev polynomials each variable can have. Each dimension will have term terms
        dim, int- number of different variables, how many dim our tensor will be
        order, string- how you want to order your polynomials. Grevlex is default.
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
        self.shape = self.coeff.shape
        if lead_term is None:
            self.update_lead_term()
        else:
            self.lead_term = tuple(lead_term)
            self.degree = sum(self.lead_term)
            self.lead_coeff = self.coeff[self.lead_term]

    def clean_coeff(self):
        """
        Gets rid of any 0's on the outside of the coeff matrix, not giving any info.
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

    def evaluate_at(self, point):
        '''
        Evaluates the polynomial at the given point. This method is overridden
        by the MultiPower and MultiCheb classes, so this definition only
        checks if the polynomial can be evaluated at the given point.

        Parameters
        ----------
        point : tuple or list
            the point at which to evaluate the polynomial

        Returns
        -------
        complex : complex
            value of the polynomial at the given point
        '''
        if len(point) != len(self.coeff.shape):
            raise ValueError('Cannot evaluate polynomial in {} variables at point {}'\
            .format(self.dim, point))

    def __eq__(self,other):
        '''
        check if coeff matrix is the same.
        '''
        if self.shape != other.shape:
            return False
        return np.allclose(self.coeff, other.coeff)

    def __ne__(self,other):
        '''
        check if coeff matrix is not the same.
        '''
        return not (self == other)


###############################################################################

#### MULTI_CHEB ###############################################################

class MultiCheb(Polynomial):
    """
    Used to represent a Chebyshev polynomial.

    Attributes
    ----------
    coeff
        The coefficient matrix represented in the object.
    dim
        The number of dimensions of the coefficient matrix
    order
        Ordering type given as a string.
    shape
        The shape of the coefficient matrix.
    lead_term
        The polynomial term with the largest total degree.
    degree
        The total degree of the lead_term.
    lead_coeff
        The coeff of the lead_term.
        
    Parameters
    ----------
    dim : int
        number of variables, dimension of polynomial system.
    terms : int
        highest term of single variable power polynomials.
    coeff : list(terms**dim) or np.array ([terms,] * dim)
        coefficents in given ordering.
    order : string
        monomial ordering desired for Grobner calculations.
    lead_term : list
        the index of the current leading coefficent.

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

        return MultiCheb(new_self + new_other)

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

    def _fold_in_i_dir(solution_matrix, dim, fdim, size_in_fdim, fold_idx):
        """
        Finds T_|m-n| (Referred to as folding in proceeding documentation)
        for a given dimension of a matrix.

        Parameters
        ----------
        solution_matrix : ndarray
            Polynomial to by folded.
        dim : int
            The number of dimensions in solution_matrix.
        fdim : int
            The dimension being folded.
        size_in_fdim : int
            The size of the solution matrix in the dimension being folded.
        fold_idx : int
            The index to fold around.

        Returns
        -------
        sol : ndarray

        """
        if fold_idx == 0:
            return solution_matrix

        sol = np.zeros_like(solution_matrix) #Matrix of zeroes used to insert the new values..
        slice_0 = slice(None, 1, None) # index to take first slice
        slice_1 = slice(fold_idx, fold_idx+1, None) # index to take slice that contains the axis folding around.

        #indexers are made with a slice index for every dimension.
        indexer1 = [slice(None)]*dim
        indexer2 = [slice(None)]*dim
        indexer3 = [slice(None)]*dim

        #Changes the index in each indexer for the correct dimension
        indexer1[fdim] = slice_0
        indexer2[fdim] = slice_1

        #makes first slice in sol equal to the slice we fold around in solution_matrix
        sol[indexer1] = solution_matrix[indexer2]

        #Loop adds the slices above and below the slice we rotate around and inserts solutions in sol.
        for n in range(size_in_fdim):

            slice_2 = slice(n+1, n+2, None) #Used to imput new values in sol.
            slice_3 = slice(fold_idx+n+1, fold_idx+n+2, None) #Used to find slices that are n above fold_idx
            slice_4 = slice(fold_idx-n-1, fold_idx-n, None) #Used to find slices that are n below fold_idx

            indexer1[fdim] = slice_2
            indexer2[fdim] = slice_3
            indexer3[fdim] = slice_4

            #if statement checks to ensure that slices to be added are contained in the matrix.
            if fold_idx-n-1 < 0:
                if fold_idx+n+2 > size_in_fdim:
                    break
                else:
                    sol[indexer1] = solution_matrix[indexer2]
            else:
                if fold_idx+n+2 > size_in_fdim:
                    sol[indexer1] = solution_matrix[indexer3]
                else:
                    sol[indexer1] = solution_matrix[indexer3] + solution_matrix[indexer2]

        return sol

    def _mon_mult1(initial_matrix, idx, dim_mult):
        """
        Executes monomial multiplication in one dimension.
        
        Parameters
        ----------
        initial_matrix : array_like
            Matrix of coefficients that represent a Chebyshev polynomial.
        idx : tuple of ints
            The index of a monomial of one variable to multiply by initial_matrix.
        dim_mult : int
            The location of the non-zero value in idx.

        Returns
        -------
        ndarray
            Coeff that are the result of the one dimensial monomial multiplication.

        """
        pad_values = list()
        for i in idx: #iterates through monomial and creates a tuple of pad values for each dimension
            pad_dim_i = (i,0)
            #In np.pad each dimension is a tuple of (i,j) where i is how many to pad in front and j is how many to pad after.
            pad_values.append(pad_dim_i)
        p1 = np.pad(initial_matrix, (pad_values), 'constant')

        largest_idx = [i-1 for i in initial_matrix.shape]
        new_shape = [max(i,j) for i,j in itertools.zip_longest(largest_idx, idx, fillvalue = 0)] #finds the largest length in each dimmension
        if initial_matrix.shape[dim_mult] <= idx[dim_mult]:
            add_a = [i-j for i,j in itertools.zip_longest(new_shape, largest_idx, fillvalue = 0)]
            add_a_list = np.zeros((len(new_shape),2))
            #changes the second column to the values of add_a and add_b.
            add_a_list[:,1] = add_a
            #uses add_a_list and add_b_list to pad each polynomial appropriately.
            initial_matrix = np.pad(initial_matrix,add_a_list.astype(int),'constant')

        number_of_dim = initial_matrix.ndim
        shape_of_self = initial_matrix.shape

        #Loop iterates through each dimension of the polynomial and folds in that dimension
        for i in range(number_of_dim):
            if idx[i] != 0:
                initial_matrix = MultiCheb._fold_in_i_dir(initial_matrix, number_of_dim, i, shape_of_self[i], idx[i])
        if p1.shape != initial_matrix.shape:
            idx = [i-j for i,j in zip(p1.shape,initial_matrix.shape)]
            pad_values = list()
            for i in idx:
                pad_dim_i = (0,i)
                pad_values.append(pad_dim_i)
            initial_matrix = np.pad(initial_matrix, (pad_values), 'constant')
        Pf = p1 + initial_matrix
        return .5*Pf

    def mon_mult(self, idx, returnType = 'Poly'):
        """
        Multiplies a Chebyshev polynomial by a monomial

        Parameters
        ----------
        idx : tuple of ints
            The index of the monomial to multiply self by.
        returnType : str
            If 'Poly' then returns a polynomial object.

        Returns
        -------
        MultiCheb object if returnType is 'Poly'.
        ndarray if returnType is "Matrix".

        """
        initial_matrix = self.coeff
        for i in range(len(idx)):
            idx_zeros = np.zeros(len(idx),dtype = int)
            idx_zeros[i] = idx[i]
            initial_matrix = MultiCheb._mon_mult1(initial_matrix, idx_zeros, i)
        if returnType == 'Poly':
            return MultiCheb(initial_matrix, lead_term = self.lead_term + np.array(idx), clean_zeros = False)
        elif returnType == 'Matrix':
            return initial_matrix

    def evaluate_at(self, point):
        """
        Evaluates a Chebyshev polynomial at a given point.

        Parameters
        ----------
        point : array_like
            The point to be evaluated at in a polynomial.

        Returns
        -------
        float
            The result of plugging a point into a polynomial.
        """
        super(MultiCheb, self).evaluate_at(point)
        
        c = self.coeff
        n = len(c.shape)
        c = cheb.chebval(point[0],c)
        for i in range(1,n):
            c = cheb.chebval(point[i],c,tensor=False)
        return c

###############################################################################

#### MULTI_POWER ##############################################################

class MultiPower(Polynomial):
    """
    Used to represent a power basis polynomial.

    Attributes
    ----------
    coeff
        The coefficient matrix represented in the object.
    dim
        The number of dimensions of the coefficient matrix.
    order
        Ordering type given as a string.
    shape
        The shape of the coefficient matrix.
    lead_term
        The polynomial term with the largest total degree.
    degree
        The total degree of the lead_term.
    lead_coeff
        The coeff of the lead_term.

    Parameters
    ----------
    dim : int
        number of variables, dimension of polynomial system.
    terms : int
        highest term of single variable power polynomials
    coeff : list(terms**dim) or np.array ([terms,] * dim)
        coefficents in given ordering.
    order : string
        monomial ordering desired for Grobner calculations.
    lead_term : list
        the index of the current leading coefficent.
            
    Methods
    -------
    __add__
        Add two power polynomials.
    __sub__
        Subtract two power polynomials.
    __mul__
        Multiply two power polynomials.
    __eq__
        Check if two power polynomials are equal.
    __ne__
        Check if two power polynomials are not equal.
    mon_mult
        Multiplies a power monomial by a power polynomial.
    evaluate_at
        Evaluate a power polynomial at a point.
        
    """
    def __init__(self, coeff, order='degrevlex', lead_term=None, clean_zeros = True):
        super(MultiPower, self).__init__(coeff, order, lead_term, clean_zeros)

    def __add__(self,other):
        '''
        Addition of two MultiPower polynomials.

        Parameters
        ----------
        other : MultiPower

        Returns
        -------
        MultiPower object
            The sum of the coeff of self and coeff of other.

        '''
        if self.shape != other.shape:
            new_self, new_other = match_size(self.coeff,other.coeff)
        else:
            new_self, new_other = self.coeff, other.coeff
        return MultiPower((new_self + new_other), clean_zeros = False)

    def __sub__(self,other):
        '''
        Subtraction of two MultiPower polynomials.

        Parameters
        ----------
        other : MultiPower

        Returns
        -------
        MultiPower
            The coeff values are the result of self.coeff - other.coeff.

        '''
        if self.shape != other.shape:
            new_self, new_other = match_size(self.coeff,other.coeff)
        else:
            new_self, new_other = self.coeff, other.coeff
        return MultiPower((new_self - (new_other)), clean_zeros = False)

    def __mul__(self,other):
        '''
        Multiplication of two MultiPower polynomials.

        Parameters
        ----------
        other : MultiPower object

        Returns
        -------
        MultiPower object
            The result of self*other.

        '''
        if self.shape != other.shape:
            new_self, new_other = match_size(self.coeff,other.coeff)
        else:
            new_self, new_other = self.coeff, other.coeff

        return MultiPower(convolve(new_self, new_other))

    def __eq__(self,other):
        '''
        Check if the coeff matrix is the same.

        Parameters
        ----------
        other : MultiPower object

        Returns
        -------
        bool
            True if the coeff of self and other are the same for all entries.

        '''
        if self.shape != other.shape:
            return False
        else:
            return np.allclose(self.coeff, other.coeff)

    def __ne__(self,other):
        '''
        check if coeff matrix is not the same

        Parameters
        ----------
        other : MultiPower object

        Returns
        -------
        bool
            True if any corresponding entries in self and other are not the same.
        '''
        return not (self == other)

    def mon_mult(self, mon, returnType = 'Poly'):
        '''
        Multiplies a polynomial by a monomial.

        Parameters
        ----------
        mon : tuple
            The powers in the monomial.
            Ex: x^3*y^4*z^2 would be input as (3,4,2)
        returnType : str
            Which type of object to return.

        Returns
        -------
        MultiPower object if returnType is 'Poly'
        ndarray if returnType is 'Matrix'
        '''
        mon = np.array(mon)
        result = np.zeros(self.shape + mon)
        slices = list()
        for i in self.shape:
            slices.append(slice(-i,None))
        result[slices] = self.coeff
        if returnType == 'Poly':
            return MultiPower(result, clean_zeros = False, lead_term = self.lead_term + mon)
        elif returnType == 'Matrix':
            return result

    def evaluate_at(self, point):
        """
        Evaluates a polynomial at a given point.

        Parameters
        ----------
        point : array_like
            The point to be evaluated at in a polynomial.

        Returns
        -------
        float
            The result of plugging a point into a polynomial.
        """
        super(MultiPower, self).evaluate_at(point)
        
        c = self.coeff
        n = len(c.shape)
        c = poly.polyval(point[0],c)
        for i in range(1,n):
            c = poly.polyval(point[i],c,tensor=False)
        return c

###############################################################################

#### CONVERT_POLY #############################################################



def conv_cheb(T):
    """
    Convert a Chebyshev polynomial to the power basis representation in one dimension.

    Parameters
    ----------
    T : array_like
        A one dimensional array_like object that represents the coeff of a
        Chebyshev polynomial.

    Returns
    -------
    ndarray
        A one dimensional array that represents the coeff of a power basis polynomial.

    """
    conv = cheb.cheb2poly(T)
    if conv.size == T.size:
        return conv
    else:
        pad = T.size - conv.size
        new_conv = np.pad(conv, ((0,pad)), 'constant')
        return new_conv

def conv_poly(P):
    """
    Convert a standard polynomial to a Chebyshev polynomial in one dimension.

    Parameters
    ----------
    P : array_like
        A one dimensional array_like object that represents the coeff of a
        power basis polynomial.

    Returns
    -------
    ndarray
        A one dimensional array that represents the coeff of a Chebyshev polynomial.

    """
    conv = cheb.poly2cheb(P)
    if conv.size == P.size:
        return conv
    else:
        pad = P.size - conv.size
        new_conv = np.pad(conv, ((0,pad)), 'constant')
        return new_conv

def cheb2poly(T):
    """
    Convert a Chebyshev polynomial to a standard polynomial in multiple dimensions.

    Parameters
    ----------
    T : MultiCheb

    Returns
    -------
    MultiPower
    """
    dim = len(T.shape)
    A = T.coeff
    for i in range(dim):
        A = np.apply_along_axis(conv_cheb, i, A)
    return MultiPower(A)

def poly2cheb(P):
    """
    Convert a standard polynomial to a Chebyshev polynomial in multiple dimensions.

    Parameters
    ----------
    P : MultiPower

    Returns
    -------
    MultiCheb
        The multi-dimensional Chebyshev polynomial.

    """
    dim = len(P.shape)
    A = P.coeff
    for i in range(dim):
        A = np.apply_along_axis(conv_poly, i, A)
    return MultiCheb(A)
