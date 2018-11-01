import numpy as np
from scipy.linalg import eig, eigvals
from numpy import linalg as la
from numalgsolve.polynomial import MultiCheb, MultiPower

def solve(poly, MSmatrix=0, eigvals=True, verbose=False):
    """Finds the zeros of a 1-D polynomial.

    Parameters
    ----------
    poly : Polynomial
        The polynomial to find the roots of.

    MSmatrix : int
        Controls which Moller-Stetter matrix is constructed
        For a univariate polynomial, the options are:
            0 (default) -- The companion or colleague matrix, rotated 180 degrees
            1 -- The unrotated companion or colleague matrix
            -1 -- The inverse of the companion or colleague matrix

    Returns
    -------
    one_dimensional_solve : numpy array
        An array of the zeros.
    """
    if MSmatrix not in [-1, 0, 1]:
        raise ValueError('MSmatrix must be -1 (inverse companion), 0 (rotated companion), or 1 (standard companion)')

    if type(poly) == MultiPower:
        size = len(poly.coeff)
        coeff = np.trim_zeros(poly.coeff)
        zeros = np.zeros(size - len(coeff), dtype = 'complex')
        if MSmatrix == 1:
            return np.hstack((zeros,multPower(coeff, eigvals, verbose=verbose)))
        elif MSmatrix == 0:
            return np.hstack((zeros,multPowerR(coeff, eigvals, verbose=verbose)))
        else:
            return np.hstack((zeros,divPower(coeff, eigvals, verbose=verbose)))
    else:
        if MSmatrix == 1:
            return multCheb(poly.coeff, eigvals, verbose=verbose)
        elif MSmatrix == 0:
            return multChebR(poly.coeff, eigvals, verbose=verbose)
        else:
            return divCheb(poly.coeff, eigvals, verbose=verbose)

def multPower(coeffs, eigvals=True, verbose=False):
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
    #constant
    if len(coeffs) < 2:
        return np.zeros([0,1])
    #linear polynomial
    elif len(coeffs) < 3:
        return -coeffs[0]/coeffs[1]

    n = len(coeffs) - 1
    matrix = np.zeros((n, n), dtype=coeffs.dtype)
    bot = matrix.reshape(-1)[n::n+1]
    bot[...] = 1
    matrix[:, -1] -= coeffs[:-1]/coeffs[-1]
    if verbose:
        print('Companion Matrix\n', matrix)
    if eigvals:
        zeros = la.eigvals(matrix)
        if verbose:
            print('Eigenvalues\n',zeros)
        return zeros
    else:
        vals,vecs = eig(matrix.T)
        if verbose:
            print('Eigenvalues\n',vals)
            print('Left Eigenvectors\n',vecs)
        return vecs[1,:]/vecs[0,:]

def multPowerR(coeffs, eigvals=True, verbose=False):
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
    #constant
    if len(coeffs) < 2:
        return np.zeros([0,1])
    #linear polynomial
    elif len(coeffs) < 3:
        return -coeffs[0]/coeffs[1]

    n = len(coeffs) - 1
    matrix = np.zeros((n, n), dtype=coeffs.dtype)
    bot = matrix.reshape(-1)[n::n+1]
    bot[...] = 1
    matrix[:, -1] -= coeffs[:-1]/coeffs[-1]
    matrix = np.rot90(matrix,2)
    if verbose:
        print('180 Rotated Companion Matrix\n', matrix)
    if eigvals:
        if verbose:
            print('Eigenvalues\n',la.eigvals(matrix))
        zeros = la.eigvals(matrix)
        return zeros
    else:
        vals,vecs= eig(matrix, left=True, right=False)
        if verbose:
            print('Eigenvalues\n',vals)
            print('Left Eigenvectors\n',vecs)
        return np.conjugate(vecs[-2,:]/vecs[-1,:])

def divPower(coeffs, eigvals=True, verbose=False):
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
    #constant
    if len(coeffs) < 2:
        return np.zeros([0,1])
    #linear polynomial
    elif len(coeffs) < 3:
        return -coeffs[0]/coeffs[1]

    n = len(coeffs) - 1
    matrix = np.zeros((n, n), dtype=coeffs.dtype)
    bot = matrix.reshape(-1)[1::n+1]
    bot[...] = 1
    matrix[:, 0] -= coeffs[1:]/coeffs[0]
    if verbose:
        print('Division Matrix\n', matrix)
    if eigvals:
        zeros = 1/la.eigvals(matrix)
        if verbose:
            print('Eigenvalues\n',la.eigvals(matrix))
        return zeros
    else:
        vals,vecs = eig(matrix, left=True, right=False)
        if verbose:
            print('Eigenvalues\n',vals)
            print('Left Eigenvectors\n',vecs)
        return np.conjugate(vecs[1,:]/vecs[0,:])

def multCheb(coeffs, eigvals=True, verbose=False):
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
    #constant
    if len(coeffs) < 2:
        return np.zeros([0,1])
    #linear polynomial
    elif len(coeffs) < 3:
        return -coeffs[0]/coeffs[1]
    #higher degree
    n = len(coeffs) - 1
    matrix = np.zeros((n,n), dtype=coeffs.dtype)
    matrix[1][0] = 1
    bot = matrix.reshape(-1)[1::n+1]
    bot[...] = 1/2
    bot = matrix.reshape(-1)[2*n+1::n+1]
    bot[...] = 1/2
    matrix[:,-1] -= .5*coeffs[:-1]/coeffs[-1]
    if verbose:
        print('Colleaugue Matrix\n', matrix)
    if eigvals:
        zeros = la.eigvals(matrix)
        if verbose:
            print('Eigenvalues\n',zeros)
        return zeros
    else:
        vals,vecs = eig(matrix, left=True, right=False)
        if verbose:
            print('Eigenvalues\n',vals)
            print('Left Eigenvectors\n',vecs)
        return vecs[1,:]/vecs[0,:]

def multChebR(coeffs, eigvals=True, verbose=False):
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
    #constant
    if len(coeffs) < 2:
        return np.zeros([0,1])
    #linear polynomial
    elif len(coeffs) < 3:
        return -coeffs[0]/coeffs[1]

    n = len(coeffs) - 1
    matrix = np.zeros((n,n), dtype=coeffs.dtype)
    matrix[1][0] = 1
    bot = matrix.reshape(-1)[1::n+1]
    bot[...] = 1/2
    bot = matrix.reshape(-1)[2*n+1::n+1]
    bot[...] = 1/2
    matrix[:,-1] -= .5*coeffs[:-1]/coeffs[-1]
    matrix = np.rot90(matrix,2)
    if verbose:
        print('Rotated Colleague Matrix\n', matrix)
    if eigvals:
        zeros = la.eigvals(matrix)
        if verbose:
            print('Eigenvalues\n',zeros)
        return zeros
    else:
        vals,vecs = eig(matrix, left=True, right=False)
        if verbose:
            print('Eigenvalues\n',vals)
            print('Left Eigenvectors\n',vecs)
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

def divCheb(coeffs, eigvals=True, verbose=False):
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
    #constant
    if len(coeffs) < 2:
        return np.zeros([0,1])
    #linear polynomial
    elif len(coeffs) < 3:
        return -coeffs[0]/coeffs[1]

    #higher degree
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

    if verbose:
        print('Division Companion Matrix\n', matrix)
    if eigvals:
        zerosD = 1/la.eigvals(matrix)
        if verbose:
            print('Eigenvalues\n',la.eigvals(matrix))
        if abs(divisor) > 1:
            return zerosD
        else:
            return zerosD*divisor
    else:
        vals,vecs = eig(matrix, left=True,right=False)
        if verbose:
            print('Eigenvalues\n',vals)
            print('Left Eigenvectors\n',vecs)
        zerosD = np.conjugate(vecs[1,:]/vecs[0,:])
        return zerosD
