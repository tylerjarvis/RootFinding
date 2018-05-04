import numpy as np
import itertools
from numpy.polynomial import chebyshev as cheb
from numpy.polynomial import polynomial as poly
from scipy.signal import fftconvolve, convolve
from numalgsolve.utils import Term, makePolyCoeffMatrix, match_size, slice_top, slice_bottom
import time

from numba import jit

@jit(cache=True)
def polyval(x, cc): #pragma: no cover
    c0 = cc[-1]
    for i in range(2, len(cc) + 1):
        c0 = cc[-i] + c0*x
    return c0

@jit(cache=True)
def polyval2(x, cc): #pragma: no cover
    cc = cc.reshape(cc.shape + (1,)*x.ndim)
    c0 = cc[-1]
    for i in range(2, len(cc) + 1):
        c0 = cc[-i] + c0*x
    return c0

@jit(cache=True)
def chebval(x, cc): #pragma: no cover
    if len(cc) == 1:
        c0 = cc[0]
        c1 = 0
    elif len(cc) == 2:
        c0 = cc[0]
        c1 = cc[1]
    else:
        x2 = 2*x
        c0 = cc[-2]
        c1 = cc[-1]
        for i in range(3, len(cc) + 1):
            tmp = c0
            c0 = cc[-i] - c1
            c1 = tmp + c1*x2
    return c0 + c1*x

@jit(cache=True)
def chebval2(x, cc): #pragma: no cover
    cc = cc.reshape(cc.shape + (1,)*x.ndim)
    if len(cc) == 1:
        c0 = cc[0]
        c1 = 0
    elif len(cc) == 2:
        c0 = cc[0]
        c1 = cc[1]
    else:
        x2 = 2*x
        c0 = cc[-2]
        c1 = cc[-1]
        for i in range(3, len(cc) + 1):
            tmp = c0
            c0 = cc[-i] - c1
            c1 = tmp + c1*x2
    return c0 + c1*x


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
    __call__
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
        elif isinstance(coeff, tuple):
            dim = len(coeff[0])
            deg = coeff[1]
            self.coeff = np.zeros([deg+1 for i in range(dim)])
            self.coeff[coeff[0]] = 1
        else:
            raise ValueError('coeff must be an np.array or a string!')
        if clean_zeros:
            self.clean_coeff()
        self.dim = self.coeff.ndim
        self.order = order
        self.shape = self.coeff.shape
        self.jac = None
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
        for cur_axis in range(self.coeff.ndim):
            change = True
            while change:
                change = False
                if self.coeff.shape[cur_axis] == 1:
                    continue
                slices = list()
                for i,degree in enumerate(self.coeff.shape):
                    if cur_axis == i:
                        s = slice(degree-1,degree)
                    else:
                        s = slice(0,degree)
                    slices.append(s)
                if np.sum(abs(self.coeff[slices])) == 0:
                    self.coeff = np.delete(self.coeff,-1,axis=cur_axis)
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
            self.degree = -1

    def __call__(self, points):
        '''
        Evaluates the polynomial at the given point. This method is overridden
        by the MultiPower and MultiCheb classes, so this definition only
        checks if the polynomial can be evaluated at the given point.

        Parameters
        ----------
        points : array-like
            the points at which to evaluate the polynomial

        Returns
        -------
         : numpy array
            valued of the polynomial at the given points
        '''
        points = np.array(points)
        if points.ndim == 1:
            points = points.reshape(1,points.shape[0])

        if points.shape[1] != self.dim:
            raise ValueError('Dimension of points does not match dimension of polynomial!')

        return points

    def grad(self, point):
        '''
        Evaluates the gradient of the polynomial at the given point. This method is overridden
        by the MultiPower and MultiCheb classes, so this definition only
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
        if len(point) != self.dim:
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
    __call__
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

        p1 = np.zeros(initial_matrix.shape + idx)
        p1[slice_bottom(initial_matrix)] = initial_matrix

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

            result = np.zeros(np.array(initial_matrix.shape) + idx)
            result[slice_top(initial_matrix)] = initial_matrix
            initial_matrix = result
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
        idx_zeros = np.zeros(len(idx),dtype = int)
        for i in range(len(idx)):
            idx_zeros[i] = idx[i]
            initial_matrix = MultiCheb._mon_mult1(initial_matrix, idx_zeros, i)
            idx_zeros[i] = 0
        if returnType == 'Poly':
            return MultiCheb(initial_matrix, lead_term = self.lead_term + np.array(idx), clean_zeros = False)
        elif returnType == 'Matrix':
            return initial_matrix

    def __call__(self, points):
        '''
        Evaluates the polynomial at the given point.

        Parameters
        ----------
        points : array-like
            the points at which to evaluate the polynomial

        Returns
        -------
        c : numpy array
            values of the polynomial at the given points
        '''
        points = super(MultiCheb, self).__call__(points)

        c = self.coeff
        n = c.ndim
        c = chebval(points[:,0],c)
        for i in range(1,n):
            c = chebval2(points[:,i],c)
        if len(c) == 1:
            return c[0]
        else:
            return c

    def evaluate_grid(self, xyz):
        '''
        Evaluates the Chebyshev polynomial on a grid of points, very efficiently.

        Parameters
        ----------
        xyz : array-like
            Each column contains the values for an axis. The direct product of these columns
            produces the points of the desired grid.

        Returns
        -------
        values: complex
            The polynomial evaluated at all of the points in the grid determined by
            the axis values
        '''

        xyz = super(MultiCheb, self).__call__(xyz)

        c = self.coeff
        n = c.ndim
        for i in range(xyz.shape[1]):
            c = chebval2(xyz[:,i] ,c)

        if np.product(c.shape)==1:
            return c[0]
        else:
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
        super(MultiCheb, self).__call__(point)

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
    __call__
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
        result[slice_bottom(self.coeff)] = self.coeff
        if returnType == 'Poly':
            return MultiPower(result, clean_zeros = False, lead_term = self.lead_term + mon)
        elif returnType == 'Matrix':
            return result

    def __call__(self, points):
        '''
        Evaluates the polynomial at the given point.

        Parameters
        ----------
        points : array-like
            the points at which to evaluate the polynomial

        Returns
        -------
        __call__: complex
            value of the polynomial at the given point
        '''
        points = super(MultiPower, self).__call__(points)

        c = self.coeff
        n = c.ndim
        c = polyval2(points[:,0],c)
        for i in range(1,n):
            c = polyval(points[:,i],c)
        if len(c) == 1:
            return c[0]
        else:
            return c

    def evaluate_grid(self, xyz):
        '''
        Evaluates the Power polynomial on a grid of points, very efficiently.

        Parameters
        ----------
        xyz : array-like
            Each column contains the values for an axis. The direct product of these columns
            produces the points of the desired grid.

        Returns
        -------
        values: complex
            The polynomial evaluated at all of the points in the grid determined by
            the axis values
        '''

        xyz = super(MultiPower, self).__call__(xyz)

        c = self.coeff
        n = c.ndim
        for i in range(xyz.shape[1]):
            c = polyval2(xyz[:,i] ,c)

        if np.product(c.shape)==1:
            return c[0]
        else:
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
        super(MultiPower, self).__call__(point)

        out = np.empty(self.dim,dtype="complex_")
        if self.jac is None:
            jac = list()
            for i in range(self.dim):
                jac.append(poly.polyder(self.coeff,axis=i))
            self.jac = jac
        spot = 0
        for i in self.jac:
            out[spot] = polyvalnd(point,i)
            spot+=1

        return out

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
    dim = T.dim
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
    dim = P.dim
    A = P.coeff
    for i in range(dim):
        A = np.apply_along_axis(conv_poly, i, A)
    return MultiCheb(A)

###############################################################################

#### is_power function ########################################################

def is_power(poly_list, return_string = False):
    """
    Determines the type of a list of polynomials.

    Parameters
    ----------
    poly_list : list of polynomial objects

    Returns
    ----------
    is_power : bool
        If the list is all power objects then returns True, if all obects are
        chebyshev then returns False, and if there is a mix then an error is
        raised

    """
    if all([type(p) == MultiPower for p in poly_list]):
        if return_string == False:
            return True
        else:
            return 'MultiPower'
    elif all([type(p) == MultiCheb for p in poly_list]):
        if return_string == False:
            return False
        else:
            return 'MultiCheb'
    else:
        print([type(p) == MultiPower for p in initial_poly_list])
        raise ValueError('Bad polynomials in list')

############################################################################

#### CHEBVALND, POLYVALND #############################################################

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
    n = c.ndim
    c = cheb.chebval(x[0],c)
    for i in range(1,n):
        c = cheb.chebval(x[i],c,tensor=False)
    return c

def polyvalnd(x,c):
    """
    Evaluate a MultiPower object at a point x

    Parameters
    ----------
    x : ndarray
        Point to evaluate at
    c : ndarray
        Tensor of Polynomial coefficients

    Returns
    -------
    c : float
        Value of the MultiPower polynomial at x
    """
    x = np.array(x)
    n = c.ndim
    c = poly.polyval(x[0],c)
    for i in range(1,n):
        c = poly.polyval(x[i],c,tensor=False)
    return c

############################################################################

#### CHEBYSHEV APPROXIMATOR ################################################
def cheb_approx(data):
    pass

#polynomial generator
def solve(poly1, poly2):
    """
    multiplies two polynomials given only their coefficients

    parameters
    ----------
    poly1,poly2 (tuple)
        tuples of coefficients of polynomials, in descending order of degree

    returns
    -------
    a tuple of coefficents for the resultant polynomial

    """
    v1 = np.array(poly1)
    v2 = np.array(poly2)

    #multiply coefficients
    M = np.outer(v2,v1.T)

    poly_coeffs = []
    #sum reverse diagonals
    #reverse matrix
    M2 = M[:,np.arange(M.shape[1])[::-1]]

    #sum diagonals
    rows,cols = M2.shape
    for i in range(-(rows-1),cols):
        poly_coeffs.append(np.trace(M2, i))

    #print('\n', poly1, poly2, poly_coeffs[::-1])
    return tuple(poly_coeffs[::-1])


def solve_poly(mylist):
    """give it the list of tuples to solve

        returns:
        tuple of solved polynomial
    """
    if len(mylist) == 1:
        return mylist[0]

    tuples = []
    size = len(mylist)
    if size % 2 == 0: #even number of roots
        for i in range(size//2):
            tuples.append(solve(mylist[i], mylist[-(i+1)]))
    else: #odd number of roots
        size -= 1
        extra = mylist[size//2]
        for i in range(size//2):
            tuples.append(solve(mylist[i], mylist[-(i+1)]))
        tuples.append(extra)
    return solve_poly(tuples)


def gen_poly(degree, variables=1):
    """
    generate degree number of random numbers in [-1,1]
    p=(x-n)(x-n)... for n in random numbers
    return n for n in numbers (roots), and the polynomial p

    (x-2)(x-3)(x-4)(x-5)

    |      1  -2 |
    |  1   1  -2 | = 1 -5 6 => x^2 -5x +6
    | -3  -3   6 |

    T_m*T_n=.5(T_(m+n)+T_(m-n))

    returns roots - list of roots
            solve_poly[::-1] - tuple with coefficients of resultant polynomial in ascending degree order
    """
    #generate <degree> random numbers
    deg = []
    for i in range(degree):
        #append tuples of form (1,-x) where x is a root
        deg.append((1,np.random.uniform(-1,1)))

    roots = []
    for i in range(degree):
        roots.append(-1*list(deg[i])[1])
    return roots, np.array(solve_poly(deg))[::-1]

def gen_poly2(rootList = [], variables=1):
    """
    generate degree number of random numbers from a list of roots

    returns roots - list of roots
            solve_poly - tuple with coefficients of resultant polynomial in ascending degree order
    """
    if rootList is None:
        return [], []

    deg = np.zeros((len(rootList),2))
    for i,v in enumerate(rootList):
        #append tuples of form (1,-x) where x is a root
        deg[i] = (1,-1*v)
    return rootList, np.array(solve_poly(deg))[::-1]
