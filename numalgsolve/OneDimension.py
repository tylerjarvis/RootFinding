import numpy as np
from scipy.linalg import eig, eigvals
from numpy import linalg as la
from numalgsolve.polynomial import MultiCheb, MultiPower

def solve(poly, method = 'mult', eigvals=True):
    """Finds the zeros of a 1-D polynomial.

    Parameters
    ----------
    poly : Polynomial
        The polynomial to find the roots of.

    method : str
        'mult' will use the multiplicaiton matrix technique.
        'div' will use the division matrix technique.
        Defaults to 'mult'

    Returns
    -------
    one_dimensional_solve : numpy array
        An array of the zeros.
    """
    if method != 'mult' and method != 'div':
        raise ValueError('method must be mult or div!')

    if type(poly) == MultiPower:
        size = len(poly.coeff)
        coeff = np.trim_zeros(poly.coeff)
        zeros = np.zeros(size - len(coeff), dtype = 'complex')
        if method == 'mult':
            return np.hstack((zeros,multPower(coeff, eigvals)))
        else:
            return np.hstack((zeros,divPower(coeff, eigvals)))
    else:
        if method == 'mult':
            return multCheb(poly.coeff, eigvals)
        else:
            return divCheb(poly.coeff, eigvals)

def multPower(coeffs, eigvals=True):
    """Finds the zeros of a 1-D power polynomial using a multiplication matrix.

    Parameters
    ----------
    coeffs : numpy array
        The coefficients of the polynomial.

    Returns
    -------
    zero : numpy array
        An array of the zeros.
    """
    n = len(coeffs) - 1
    matrix = np.zeros((n, n), dtype=coeffs.dtype)
    bot = matrix.reshape(-1)[n::n+1]
    bot[...] = 1
    matrix[:, -1] -= coeffs[:-1]/coeffs[-1]
    if eigvals:
        zeros = la.eigvals(matrix)
        return zeros
    else:
        vals,vecs = eig(matrix.T)
        return vecs[1,:]/vecs[0,:]

def multPowerR(coeffs, eigvals=True):
    """Finds the zeros of a 1-D power polynomial using a rotated multiplication matrix.

    Parameters
    ----------
    coeffs : numpy array
        The coefficients of the polynomial.

    Returns
    -------
    zero : numpy array
        An array of the zeros.
    """
    n = len(coeffs) - 1
    matrix = np.zeros((n, n), dtype=coeffs.dtype)
    bot = matrix.reshape(-1)[n::n+1]
    bot[...] = 1
    matrix[:, -1] -= coeffs[:-1]/coeffs[-1]
    matrix = np.rot90(matrix,2)
    if eigvals:
        zeros = la.eigvals(matrix)
        return zeros
    else:
        vals,vecs= eig(matrix, left=True, right=False)
        return np.conjugate(vecs[-2,:]/vecs[-1,:])

def divPower(coeffs, eigvals=True):
    """Finds the zeros of a 1-D power polynomial using a division matrix.

    Parameters
    ----------
    coeffs : numpy array
        The coefficients of the polynomial.

    Returns
    -------
    zero : numpy array
        An array of the zeros.
    """
    n = len(coeffs) - 1
    matrix = np.zeros((n, n), dtype=coeffs.dtype)
    bot = matrix.reshape(-1)[1::n+1]
    bot[...] = 1
    matrix[:, 0] -= coeffs[1:]/coeffs[0]
    if eigvals:
        zeros = 1/la.eigvals(matrix)
        return zeros
    else:
        vals,vecs = eig(matrix, left=True, right=False)
        return np.conjugate(vecs[1,:]/vecs[0,:])

def multCheb(coeffs, eigvals=True):
    """Finds the zeros of a 1-D chebyshev polynomial using a multiplication matrix.

    Parameters
    ----------
    coeffs : numpy array
        The coefficients of the polynomial.

    Returns
    -------
    zero : numpy array
        An array of the zeros.
    """
    n = len(coeffs) - 1
    matrix = np.zeros((n,n), dtype=coeffs.dtype)
    matrix[1][0] = 1
    bot = matrix.reshape(-1)[1::n+1]
    bot[...] = 1/2
    bot = matrix.reshape(-1)[2*n+1::n+1]
    bot[...] = 1/2
    matrix[:,-1] -= .5*coeffs[:-1]/coeffs[-1]
    if eigvals:
        zeros = la.eigvals(matrix)
        return zeros
    else:
        vals,vecs = eig(matrix, left=True, right=False)
        return vecs[1,:]/vecs[0,:]

def multChebR(coeffs, eigvals=True):
    """Finds the zeros of a 1-D chebyshev polynomial using a multiplication matrix.

    Parameters
    ----------
    coeffs : numpy array
        The coefficients of the polynomial.

    Returns
    -------
    zero : numpy array
        An array of the zeros.
    """
    n = len(coeffs) - 1
    matrix = np.zeros((n,n), dtype=coeffs.dtype)
    matrix[1][0] = 1
    bot = matrix.reshape(-1)[1::n+1]
    bot[...] = 1/2
    bot = matrix.reshape(-1)[2*n+1::n+1]
    bot[...] = 1/2
    matrix[:,-1] -= .5*coeffs[:-1]/coeffs[-1]
    matrix = np.rot90(matrix,2)
    if eigvals:
        zeros = la.eigvals(matrix)
        return zeros
    else:
        vals,vecs = eig(matrix, left=True, right=False)
        return np.conjugate(vecs[-2,:]/vecs[-1,:])


def getXinv(coeff):
    n = len(coeff)-1
    curr = coeff.copy()
    xinv = np.zeros(n, dtype=coeff.dtype)
    for i in range(1,n)[::-1]:
        val = -curr[i+1]
        curr[i-1] += val
        xinv[i]+=2*val
    temp = -curr[1]
    xinv[0]+=temp
    return xinv,curr[0]


def divCheb(coeffs, eigvals=True):
    """Finds the zeros of a 1-D chebyshev polynomial using a division matrix.

    Parameters
    ----------
    coeffs : numpy array
        The coefficients of the polynomial.

    Returns
    -------
    zero : numpy array
        An array of the zeros.
    """
    xinv,divisor = getXinv(coeffs)
    n = len(coeffs)-1

    matrix = np.zeros((n,n), dtype=coeffs.dtype)

    sign = 1
    for col in range(1,n,2):
        bot = matrix.reshape(-1)[col:(n-col)*n:n+1]
        bot[...] = 2*sign
        sign *= -1
    matrix[0]/=2

    if abs(divisor) > 1:
        xinv/=divisor
    else:
        matrix*=divisor

    sign = 1
    for col in range(0,n,2):
        matrix[:,col]+=xinv*sign
        sign*=-1

    if eigvals:
        zerosD = 1/la.eigvals(matrix)
        if abs(divisor) > 1:
            return zerosD
        else:
            return zerosD*divisor
    else:
        vals,vecs = eig(matrix, left=True,right=False)
        zerosD = np.conjugate(vecs[1,:]/vecs[0,:])
        return zerosD
