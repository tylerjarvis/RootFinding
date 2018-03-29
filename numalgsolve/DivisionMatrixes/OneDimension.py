import numpy as np
from scipy.linalg import eig, norm, eigvals
from numpy import linalg as la
from numalgsolve.polynomial import MultiCheb, MultiPower

def one_dimensional_solve(poly, method = 'M'):
    """Finds the zeros of a 1-D polynomial.
    
    Parameters
    ----------
    poly : Polynomial
        The polynomial to find the roots of.
    
    method : str
        'M' will use the multiplicaiton matrix technique.
        'D' will use the division matrix technique.
        Defaults to 'M'

    Returns
    -------
    one_dimensional_solve : numpy array
        An array of the zeros.
    """
    if type(poly) == MultiPower:
        size = len(poly.coeff)
        coeff = np.trim_zeros(poly.coeff)
        zeros = np.zeros(size - len(coeff), dtype = 'complex')
        if method == 'M':
            return np.hstack((zeros,multPower(coeff)))
        else:
            return np.hstack((zeros,divPower(coeff)))
    else:
        if method == 'M':
            return multCheb(poly.coeff)
        else:
            return divCheb(poly.coeff)

def multPower(coeffs):
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
    zeros = la.eigvals(matrix)
    return zeros

def divPower(coeffs):
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
    zeros = 1/la.eigvals(matrix)
    return zeros

def multCheb(coeffs):
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
    mMatrix = np.zeros((n,n), dtype=coeffs.dtype)
    mMatrix[1][0] = 1
    bot = mMatrix.reshape(-1)[1::n+1]
    bot[...] = 1/2
    bot = mMatrix.reshape(-1)[2*n+1::n+1]
    bot[...] = 1/2
    #print(coeffs[-1])
    mMatrix[:,-1] -= .5*coeffs[:-1]/coeffs[-1]
    zeros = la.eigvals(mMatrix)
    return zeros

def getXinv(coeff):
    n = len(coeff)-1
    curr = coeff.copy()
    xinv = np.zeros(n, dtype=coeff.dtype)
    for i in range(1,n)[::-1]:
        val = -curr[i+1]
        curr[i+1] += val
        curr[i-1] += val
        xinv[i]+=2*val
    temp = -curr[1]
    curr[1]+=temp
    xinv[0]+=temp
    #xinv/=curr[0]
    return xinv,curr[0]


def divCheb(coeffs):
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
    
    dMatrix = np.zeros((n,n), dtype=coeffs.dtype)

    sign = 1
    for col in range(1,n,2):
        bot = dMatrix.reshape(-1)[col:(n-col)*n:n+1]
        bot[...] = 2*sign
        sign *= -1
    dMatrix[0]/=2
    
    if abs(divisor) > 1:
        xinv/=divisor
    else:
        dMatrix*=divisor
    
    sign = 1
    for col in range(0,n,2):
        dMatrix[:,col]+=xinv*sign
        sign*=-1
    
    zerosD = 1/la.eigvals(dMatrix)
    #print(divisor)
    
    if abs(divisor) > 1:
        return zerosD
    else:
        return zerosD*divisor