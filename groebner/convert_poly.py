import numpy as np
from groebner.polynomial import Polynomial
from numpy.polynomial import chebyshev as C
from groebner.multi_cheb import MultiCheb
from groebner.multi_power import MultiPower

def conv_cheb(T):
    """
    Convert a chebyshev polynomial to the power basis representation.
    Args:
        T (): The chebyshev polynomial to convert.
    Returns:
        new_conv (): The chebyshev polynomial converted to the power basis representation.

    """
    conv = C.cheb2poly(T)
    if conv.size == T.size:
        return conv
    else:
        pad = T.size - conv.size
        new_conv = np.pad(conv, ((0,pad)), 'constant')
        return new_conv

def conv_poly(P):
    """
    Convert a standard polynomial to a chebyshev polynomial in one dimension.

    Args:
        P (): The standard polynomial to be converted.

    Returns:
        new_conv (): The chebyshev polynomial.

    """
    conv = C.poly2cheb(P)
    if conv.size == P.size:
        return conv
    else:
        pad = P.size - conv.size
        new_conv = np.pad(conv, ((0,pad)), 'constant')
        return new_conv

def cheb2poly(T):
    """
    Convert a chebyshev polynomial to a standard polynomial in multiple dimensions.

    """
    dim = len(T.shape)
    A = T.coeff
    for i in range(dim):
        A = np.apply_along_axis(conv_cheb, i, A)
    return MultiPower(A)

def poly2cheb(P):
    """
    Convert a standard polynomial to a chebyshev polynomial in multiple dimensions.
    
    Args:
        P (): The multi-dimensional standard polynomial. (tensor?)

    Returns:
        (MultiCheb): The multi-dimensional chebyshev polynomial.

    """
    dim = len(P.shape)
    A = P.coeff
    for i in range(dim):
        A = np.apply_along_axis(conv_poly, i, A)
    return MultiCheb(A)
