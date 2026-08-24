"""Coefficient-tensor polynomial types used throughout yroots.

Defines :class:`MultiCheb` and :class:`MultiPower` (both subclasses of the
:class:`Polynomial` base) for representing multivariate polynomials by their
coefficient tensor, plus the small helpers used to evaluate them efficiently.
"""
import numpy as np
from scipy.signal import convolve
from numpy.polynomial import chebyshev as cheb
from numpy.polynomial import polynomial as poly

def slice_top(matrix_shape):
    """Gets the n-d slices needed to slice a matrix into the top corner of another.

    Parameters
    ----------
    matrix_shape : tuple.
        The matrix shape of interest.
    Returns
    -------
    slices : list
        Each value of the list is a slice of the matrix in some dimension. It is exactly the size of matrix_shape.
    """
    slices = list()
    for i in matrix_shape:
        slices.append(slice(0,i))
    return tuple(slices)

def match_size(a,b):
    """
    Matches the shape of two matrices.

    Parameters
    ----------
    a, b : ndarray
        Matrices whose size is to be matched.

    Returns
    -------
    a, b : ndarray
        Matrices of equal size.
    """
    new_shape = np.maximum(a.shape, b.shape)

    a_new = np.zeros(new_shape)
    a_new[slice_top(a.shape)] = a
    b_new = np.zeros(new_shape)
    b_new[slice_top(b.shape)] = b
    return a_new, b_new

############ Fast polynomial evaluation functions ############

def polyval(x, cc):
    """Horner evaluation of a power-basis polynomial along the leading axis of ``cc``.

    Parameters
    ----------
    x : numpy array
        Points at which to evaluate.
    cc : numpy array
        Coefficient array whose first axis indexes the polynomial degree.

    Returns
    -------
    numpy array
        Polynomial values, with the leading degree axis consumed.
    """
    c0 = cc[-1]
    for i in range(2, len(cc) + 1):
        c0 = cc[-i] + c0*x
    return c0

def chebval(x, cc):
    """Clenshaw evaluation of a Chebyshev-basis polynomial along the leading axis of ``cc``.

    Parameters
    ----------
    x : numpy array
        Points at which to evaluate.
    cc : numpy array
        Coefficient array whose first axis indexes the Chebyshev degree.

    Returns
    -------
    numpy array
        Polynomial values, with the leading degree axis consumed.
    """
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
    """Superclass for :class:`MultiPower` and :class:`MultiCheb`.

    Attributes
    ----------
    coeff : numpy array
        The coefficient tensor of the polynomial.
    dim : int
        The number of dimensions of the coefficient tensor.
    shape : tuple of int
        The shape of the coefficient tensor.
    jac : list of numpy array or None
        Cached partial derivatives, populated on the first call to :meth:`grad`.

    Parameters
    ----------
    coeff : numpy array
        Coefficients of the polynomial.
    clean_zeros : bool
        Defaults to True. If True, trailing rows/columns/etc. of all-zero
        coefficients are trimmed from ``coeff``.
    """
    def __init__(self, coeff, clean_zeros = True):

        if isinstance(coeff,list):
            coeff = np.array(coeff)
        if not isinstance(coeff,np.ndarray):
            raise ValueError("Invalid input for Polynomial class object: the coefficients "
                            "must be a list or numpy array of real numbers, but a "
                            f"{type(coeff).__name__} was given")
        if not np.issubdtype(coeff.dtype, np.number) or np.issubdtype(coeff.dtype, np.complexfloating):
            raise ValueError("Invalid input for Polynomial class object: the coefficients must "
                             f"be real numbers, but an array of dtype '{coeff.dtype}' was given")
        if coeff.ndim == 0:
            raise ValueError("Invalid input for Polynomial class object: the coefficients must "
                             "have at least one dimension, but a 0-dimensional array was given. "
                             "Use np.array([c]) for the constant polynomial c")
        if coeff.size == 0:
            raise ValueError("Invalid input for Polynomial class object: the coefficients must "
                             f"contain at least one entry, but an empty array of shape {coeff.shape} "
                             "was given")
        if coeff.dtype != np.float64:
            coeff = coeff.astype(np.float64)
        self.coeff = coeff
        if clean_zeros:
            self.clean_coeff()
        self.dim = self.coeff.ndim
        self.shape = self.coeff.shape
        self.jac = None

    def clean_coeff(self):
        """Get rid of any zeros on the outside of the coefficient matrix."""
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
        """
        Evaluates the polynomial at the given point. This method is overridden
        by the MultiPower and MultiCheb classes, so this definition only
        checks if the polynomial can be evaluated at the given point.

        Parameters
        ----------
        points : array-like
            the points at which to evaluate the polynomial

        Returns
        -------
        points : numpy array
            The validated/reshaped input points. Subclasses (:class:`MultiCheb`,
            :class:`MultiPower`) override :meth:`__call__` to return the polynomial values
            themselves; the base implementation only normalizes the input.
        """
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
        """
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
        """
        if len(point) != self.dim:
            raise ValueError('Cannot evaluate polynomial in {} variables at point {}'\
            .format(self.dim, point))

    def matched_coeffs(self, other):
        """Line up this polynomial's coefficients with another's for a binary operation.

        Two polynomials can be combined coefficient by coefficient only when they are in
        the same basis and have the same number of variables. Shapes may still differ --
        one polynomial may carry higher degree terms than the other -- so the smaller
        tensor is zero padded up to the larger.

        Parameters
        ----------
        other : object
            The right hand operand. Anything that is not a polynomial of this same class
            and dimension is rejected.

        Returns
        -------
        tuple of numpy array, or None
            The two coefficient tensors, padded to a common shape, or None when ``other``
            is not a compatible operand. Callers return ``NotImplemented`` on None so
            python raises its own TypeError for the operator.
        """
        # A different class means a different basis: adding the tensors of a MultiCheb and
        # a MultiPower gives a polynomial that is neither of them.
        if type(other) is not type(self):
            return None
        # match_size broadcasts a lower dimensional tensor across the missing axes, which
        # silently turns a polynomial in one variable into a different one in two.
        if self.dim != other.dim:
            return None
        if self.shape != other.shape:
            return match_size(self.coeff, other.coeff)
        return self.coeff, other.coeff

    def __eq__(self,other):
        """Check if the polynomials are the same, in basis, dimension and coefficients."""
        if not isinstance(other, Polynomial):
            return NotImplemented
        matched = self.matched_coeffs(other)
        if matched is None:      # different basis or different number of variables
            return False
        return np.allclose(*matched)

    def __ne__(self,other):
        """Check if coeff matrix is not the same."""
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

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

    To represent 4*T_2(x) + 1*T_3(x) (using Chebyshev polynomials of the first kind):

    >>> f = yroots.MultiCheb([0,0,4,1])
    >>> print(f)
    [0. 0. 4. 1.]


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
        """Addition of two MultiCheb polynomials.

        Parameters
        ----------
        other : MultiCheb

        Returns
        -------
        MultiCheb
            The sum of the coeff of self and coeff of other.

        """
        matched = self.matched_coeffs(other)
        if matched is None:
            return NotImplemented
        new_self, new_other = matched

        return MultiCheb(new_self + new_other)

    def __sub__(self,other):
        """
        Subtraction of two MultiCheb polynomials.

        Parameters
        ----------
        other : MultiCheb

        Returns
        -------
        MultiCheb
            The coeff values are the result of self.coeff - other.coeff.
        """
        matched = self.matched_coeffs(other)
        if matched is None:
            return NotImplemented
        new_self, new_other = matched
        return MultiCheb((new_self - (new_other)))
    
    def __call__(self, points):
        """
        Evaluates the polynomial at the given point.

        Parameters
        ----------
        points : array-like
            the points at which to evaluate the polynomial

        Returns
        -------
        c : numpy array
            values of the polynomial at the given points
        """
        points = super(MultiCheb, self).__call__(points)

        c = self.coeff
        n = c.ndim
        cc = c.reshape(c.shape + (1,)*points.ndim)
        c = chebval(points[:,0],cc)
        for i in range(1,n):
            c = chebval(points[:,i],c)
        if len(c) == 1:
            return c[0]
        else:
            return c
        
    def evaluate_grid(self, xyz):
        """
        Evaluates the Chebyshev polynomial on a grid of points, very efficiently.

        Parameters
        ----------
        xyz : array-like
            Each column contains the values for an axis. The direct product of these columns
            produces the points of the desired grid.

        Returns
        -------
        values : numpy array
            The polynomial evaluated at all of the points in the grid determined by
            the axis values. Returns a scalar when the grid contains a single point.
        """

        xyz = super(MultiCheb, self).__call__(xyz)

        c = self.coeff
        for i in range(xyz.shape[1]):
            cc = c.reshape(c.shape + (1,)*xyz[:, i].ndim)
            c = chebval(xyz[:, i], cc)

        if np.prod(c.shape)==1:
            return c[0]
        else:
            return c

    def grad(self, point):
        """
        Evaluates the gradient of the polynomial at the given point.

        Parameters
        ----------
        point : array-like
            the point at which to evaluate the polynomial

        Returns
        -------
        out : ndarray
            Gradient of the polynomial at the given point.
        """
        super(MultiCheb, self).__call__(point)

        out = np.empty(self.dim,dtype=np.float64)
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
        """Addition of two MultiPower polynomials.

        Parameters
        ----------
        other : MultiPower

        Returns
        -------
        MultiPower object
            The sum of the coeff of self and coeff of other.

        """
        matched = self.matched_coeffs(other)
        if matched is None:
            return NotImplemented
        new_self, new_other = matched
        return MultiPower((new_self + new_other))

    def __sub__(self,other):
        """
        Subtraction of two MultiPower polynomials.

        Parameters
        ----------
        other : MultiPower

        Returns
        -------
        MultiPower
            The coeff values are the result of self.coeff - other.coeff.

        """
        matched = self.matched_coeffs(other)
        if matched is None:
            return NotImplemented
        new_self, new_other = matched
        return MultiPower((new_self - (new_other)))

    def __mul__(self,other):
        """
        Multiplication of two MultiPower polynomials.

        Parameters
        ----------
        other : MultiPower object

        Returns
        -------
        MultiPower object
            The result of self*other.

        """
        matched = self.matched_coeffs(other)
        if matched is None:
            return NotImplemented
        new_self, new_other = matched

        return MultiPower(convolve(new_self, new_other))
    
    def __call__(self, points):
        """
        Evaluates the polynomial at the given point.

        Parameters
        ----------
        points : array-like
            the points at which to evaluate the polynomial

        Returns
        -------
        c : numpy array
            values of the polynomial at the given points
        """
        points = super(MultiPower, self).__call__(points)

        c = self.coeff
        n = c.ndim
        cc = c.reshape(c.shape + (1,)*points.ndim)
        c = polyval(points[:,0],cc)
        for i in range(1,n):
            c = polyval(points[:,i],c)
        if len(c) == 1:
            return c[0]
        else:
            return c
    
    def evaluate_grid(self, xyz):
        """
        Evaluates the Power polynomial on a grid of points, very efficiently.

        Parameters
        ----------
        xyz : array-like
            Each column contains the values for an axis. The direct product of these columns
            produces the points of the desired grid.

        Returns
        -------
        values : numpy array
            The polynomial evaluated at all of the points in the grid determined by
            the axis values. Returns a scalar when the grid contains a single point.
        """

        xyz = super(MultiPower, self).__call__(xyz)

        c = self.coeff
        for i in range(xyz.shape[1]):
            cc = c.reshape(c.shape + (1,)*xyz[:,i].ndim)
            c = polyval(xyz[:,i] ,cc)

        if np.prod(c.shape)==1:
            return c[0]
        else:
            return c

    def grad(self, point):
        """
        Evaluates the gradient of the polynomial at the given point.

        Parameters
        ----------
        point : array-like
            the point at which to evaluate the polynomial

        Returns
        -------
        out : ndarray
            Gradient of the polynomial at the given point.
        """
        super(MultiPower, self).__call__(point)

        out = np.empty(self.dim,dtype=np.float64)
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
    def to_cheb(self):
        """Convert this power-basis polynomial's coefficients to the Chebyshev basis.

        Returns
        -------
        cheb_coeffs : numpy array
            The Chebyshev coefficient tensor equivalent to ``self.coeff``.
        """
        def get_new_As(As):
            """ Finds the next transformation coefficients from the previous ones.
                So if x^n = sum(As[i]*T_i(x)), x^(n+1) = sum(Bs[i]*T_i(x)).
            """
            n = len(As)
            if n == 0:
                return np.array([1.])
            Bs = np.zeros(n+1)
            # Edge case if As has length 1
            if n == 1:
                Bs[1] = As[0]
                return Bs
            # Put in the first and last coeffs
            if n%2 == 0:
                Bs[0] = As[1]/2
            Bs[-1] = As[-1]/2
            # Put in the second coeff
            if n == 2:
                Bs[1] = As[0]
                return Bs
            if n%2 == 1:
                Bs[1] = As[0] + As[2]/2
            # Do all the middle coefficients, only editing the ones that shouldn't be 0.
            if n > 3:
                Bs[2+n%2:-2:2] = (As[1+n%2:-2:2] + As[3+n%2::2])/2 
            return Bs
        def to_cheb1D(coeffs):
            """Transforms to chebyshev coeficcients along the first dimension of coeffs matrix"""
            cheb_coeffs = np.zeros_like(coeffs, dtype=np.float64)
            As = []
            # Update As, then take each slice of the coefficient matrix and matrix multiply 
            # by As, and add to the cheb_coeffs matrix.
            for i,coeff in enumerate(coeffs):
                As = get_new_As(As)
                # Invoke einsum to do the right matrix multiplication in n dimensions
                cheb_coeffs[:i+1] += np.einsum("i,...->i...",As,coeff) 
                #np.expand_dims(As,axis=1)@np.array([coeff])
            return cheb_coeffs
        def to_chebND(coeffs,dim):
            """Transforms to chebyshev coefficients along the dim axis of the coeffs matrix"""
            # Get the transopse order to make the desired dim first
            order = np.array([dim] + [i for i in range(dim)] + [i for i in range(dim+1, coeffs.ndim)])
            # Then transpose with the inverted order after the transformation occurs.
            backOrder = np.zeros(coeffs.ndim, dtype = int)
            backOrder[order] = np.arange(coeffs.ndim)
            # Transpose coeffs, transform them along the first dimension, then transpose them back.
            return to_cheb1D(coeffs.transpose(order),).transpose(backOrder)
        cheb_coeffs = self.coeff
        for dim in range(self.coeff.ndim):
            # Go through each dimension and transform
            cheb_coeffs = to_chebND(cheb_coeffs,dim)
        return cheb_coeffs

        
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