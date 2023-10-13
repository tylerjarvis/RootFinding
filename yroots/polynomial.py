import numpy as np
from scipy.signal import convolve
from numpy.polynomial import chebyshev as cheb
from numpy.polynomial import polynomial as poly

def slice_top(matrix_shape):
    ''' Gets the n-d slices needed to slice a matrix into the top corner of another.

    Parameters
    ----------
    matrix_shape : tuple.
        The matrix shape of interest.
    Returns
    -------
    slices : list
        Each value of the list is a slice of the matrix in some dimension. It is exactly the size of matrix_shape.
    '''
    slices = list()
    for i in matrix_shape:
        slices.append(slice(0,i))
    return tuple(slices)

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
    a_new[slice_top(a.shape)] = a
    b_new = np.zeros(new_shape)
    b_new[slice_top(b.shape)] = b
    return a_new, b_new

############ Fast polynomial evaluation functions ############

def polyval(x, cc):
    c0 = cc[-1]
    for i in range(2, len(cc) + 1):
        c0 = cc[-i] + c0*x
    return c0

def polyval2(x, cc):
    c0 = cc[-1]
    for i in range(2, len(cc) + 1):
        c0 = cc[-i] + c0*x
    return c0

def chebval(x, cc):
    if len(cc) == 1:
        c0 = cc[0]
        c1 = np.zeros_like(c0)
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

def chebval2(x, cc):
    if len(cc) == 1:
        c0 = cc[0]
        c1 = np.zeros_like(c0)
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

################################################

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
    clean_zeros : bool
        Default is True. If True, all extra rows, columns, etc of all zeroes are
        removed from matrix of coefficients.

    Methods
    -------
    clean_coeff
        Removes extra rows, columns, etc of zeroes from end of matrix of coefficients
    match_size
        Matches the shape of two matrices.
    __call__
        Evaluates a polynomial at a certain point.
    __eq__
        Checks if two polynomials are equal.
    __ne__
        Checks if two polynomials are not equal.

    '''
    def __init__(self, coeff, clean_zeros = True):

        if isinstance(coeff,list):
            coeff = np.array(coeff)
        if isinstance(coeff,np.ndarray):
            self.coeff = coeff
            # If coeff has integer coefficients,
            # cast as numpy floats for jit compilation
            if coeff.dtype == np.int32 or coeff.dtype == np.int64:
                coeff = coeff.astype(np.float64)
        else:
            raise ValueError('Invalid input for Polynomial class object')
        if clean_zeros:
            self.clean_coeff()
        self.dim = self.coeff.ndim
        self.shape = self.coeff.shape
        self.jac = None

    def clean_coeff(self):
        """
        Get rid of any zeros on the outside of the coefficient matrix.
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
                if np.sum(abs(self.coeff[tuple(slices)])) == 0:
                    self.coeff = np.delete(self.coeff,-1,axis=cur_axis)
                    change = True

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
        if points.ndim == 0:
            points = np.array([points])

        if points.ndim == 1:
            if self.dim > 1:
                points = points.reshape(1,points.shape[0])
            else:
                points = points.reshape(points.shape[0],1)

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

    def __repr__(self):
        return str(self.coeff)
    
    def __str__(self):
        return str(self.coeff)

###############################################################################

#### MULTI_CHEB ###############################################################
class MultiCheb(Polynomial):
    """Coefficient tensor representation of a Chebyshev basis polynomial.

    Using this class instead of a Python callable function to represent a Chebyshev polynomial
    can lead to faster function evaluations during approximation.

    Examples
    --------

    To represent 4*T_2(x) + 1T_3(x) (using Chebyshev polynomials of the first kind):

    >>> f = yroots.MultiCheb([0,0,4,1])
    >>> print(f)
    [-4.   0.   5.5  0.   0.   0.   3. ]


    Parameters
    ----------
    coeff : list or numpy array
        An array containing the coefficients of the polynomial. If the polynomial is n-dimensional,
        the (i,j,...,n) index represents the term having T_i(x)*T_j(y)*....
    clean_zeros : bool
        Whether or not to remove all extra rows or columns containing only zeros. Defaults to True.

    """
    def __init__(self, coeff, clean_zeros = True):
        super(MultiCheb, self).__init__(coeff, clean_zeros)

    def __add__(self,other):
        '''Addition of two MultiCheb polynomials.

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
        cc = c.reshape(c.shape + (1,)*points.ndim)
        c = chebval2(points[:,0],cc)
        for i in range(1,n):
            c = chebval(points[:,i],c)
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
        for i in range(xyz.shape[1]):
            cc = c.reshape(c.shape + (1,)*xyz[:,i].ndim)
            c = chebval2(xyz[:,i] ,cc)

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
    """Coefficient tensor representation of a power basis polynomial.

    Using this class instead of a Python callable function to represent a power basis polynomial
    can lead to faster function evaluations during approximation.

    Examples
    --------

    To represent 3x^6 + 5.5x^2 -4:

    >>> f = yroots.MultiPower([-4,0,5.5,0,0,0,3])
    >>> print(f)
    [-4.   0.   5.5  0.   0.   0.   3. ]

    To represent 0.62x^3*y - 0.11x*y + 1.03y^2 - 0.58:

    >>> f = yroots.MultiPower(np.array([[-0.58,0,1.03],[0,-0.11,0],[0,0,0],[0,0.62,0]]))
    >>> print(f)
    [[-0.58  0.    1.03]
     [ 0.   -0.11  0.  ]
     [ 0.    0.    0.  ]
     [ 0.    0.62  0.  ]]


    Parameters
    ----------
    coeff : list or numpy array
        An array containing the coefficients of the polynomial. If the polynomial is n-dimensional,
        the (i,j,...,n) index represents the term of degree i in dimension 0, degree j in dimension 1,
        and so forth.
    clean_zeros : bool
        Whether or not to remove all extra rows or columns containing only zeros. Defaults to True.

    """
    def __init__(self, coeff, clean_zeros = True):
        super(MultiPower, self).__init__(coeff, clean_zeros)

    def __add__(self,other):
        '''Addition of two MultiPower polynomials.

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
        cc = c.reshape(c.shape + (1,)*points.ndim)
        c = polyval2(points[:,0],cc)
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
        for i in range(xyz.shape[1]):
            cc = c.reshape(c.shape + (1,)*xyz[:,i].ndim)
            c = polyval2(xyz[:,i] ,cc)

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