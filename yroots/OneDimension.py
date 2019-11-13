import numpy as np
from scipy.linalg import eig, eigvals
from numpy import linalg as la
from yroots.polynomial import MultiCheb, MultiPower

def solve(poly, MSmatrix=0, eigvals=True, verbose=False):
    """Finds the zeros of a 1-D polynomial.

    Parameters
    ----------
    poly : Polynomial
        The polynomial to find the roots of.

    MSmatrix : int
        Controls which Moller-Stetter matrix is constructed
        For a univariate polynomial, the options are:
            0 (default) -- The companion or colleague matrix
            -1 -- The inverse of the companion or colleague matrix

    Returns
    -------
    one_dimensional_solve : numpy array
        An array of the zeros.
    """
    if MSmatrix not in [-1, 0]:
        raise ValueError('MSmatrix must be -1 (inverse companion), or 0 (rotated companion)')

    if type(poly) == MultiPower:
        size = len(poly.coeff)
        coeff = np.trim_zeros(poly.coeff)
        zeros = np.zeros(size - len(coeff), dtype = 'complex')
        if MSmatrix == 0:
            #multPower is rotated 180 so it plays nice with hessenberg properties
            return np.hstack((zeros,multPower(coeff, eigvals, verbose=verbose)))
        else:
            return np.hstack((zeros,divPower(coeff, eigvals, verbose=verbose)))
    else:
        if MSmatrix == 0:
            return multCheb(poly.coeff, eigvals, verbose=verbose)
        else:
            return divCheb(poly.coeff, eigvals, verbose=verbose)

def multPower(coeff, eigvals=True, verbose=False):
    """Finds the zeros of a 1-D power polynomial using a rotated multiplication matrix.

    Parameters
    ----------
    coeff : numpy array
        The coefficients of the polynomial.

    Returns
    -------
    zero : numpy array
        An array of the zeros.
    """
    n = len(coeff) - 1

    # linear/constant cases
    if n < 1:
        return np.array([], dtype=coeff.dtype)
    if n == 1:
        return np.array([-coeff[0]/coeff[1]])

    matrix = np.zeros((n, n), dtype=coeff.dtype)
    bot = matrix.reshape(-1)[n::n+1]
    bot[...] = 1
    matrix[:, -1] -= coeff[:-1]/coeff[-1]
    matrix = np.rot90(matrix,2)
    matrix = np.array(matrix, dtype=float)
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

def divPower(coeff, eigvals=True, verbose=False):
    """Finds the zeros of a 1-D power polynomial using a division matrix.

    Parameters
    ----------
    coeff : numpy array
        The coefficients of the polynomial.

    Returns
    -------
    zero : numpy array
        An array of the zeros.
    """
    n = len(coeff) - 1

    # linear/constant cases
    if n < 1:
        return np.array([], dtype=coeff.dtype)
    if n == 1:
        return np.array([-coeff[0]/coeff[1]])


    matrix = np.zeros((n, n), dtype=coeff.dtype)
    bot = matrix.reshape(-1)[1::n+1]
    bot[...] = 1
    matrix[:, 0] -= coeff[1:]/coeff[0]
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

def multCheb(coeff, eigvals=True, verbose=False):
    """Finds the zeros of a 1-D chebyshev polynomial using a multiplication matrix.

    Parameters
    ----------
    coeff : numpy array
        The coefficients of the polynomial.

    Returns
    -------
    zero : numpy array
        An array of the zeros.
    """
    n = len(coeff) - 1

    # linear/constant cases
    if n < 1:
        return np.array([], dtype=coeff.dtype)
    if n == 1:
        return np.array([-coeff[0]/coeff[1]])

    matrix = np.zeros((n,n), dtype=coeff.dtype)
    matrix[1][0] = 1
    bot = matrix.reshape(-1)[1::n+1]
    bot[...] = 1/2
    bot = matrix.reshape(-1)[2*n+1::n+1]
    bot[...] = 1/2
    matrix[:,-1] -= .5*coeff[:-1]/coeff[-1]
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

def getXinv(coeff):
    """Helper function for division matrix"""
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

def divCheb(coeff, eigvals=True, verbose=False):
    """Finds the zeros of a 1-D chebyshev polynomial using a division matrix.

    Parameters
    ----------
    coeff : numpy array
        The coefficients of the polynomial.

    Returns
    -------
    zero : numpy array
        An array of the zeros.
    """
    xinv,divisor = getXinv(coeff)
    n = len(coeff)-1

    # linear/constant cases
    if n < 1:
        return np.array([], dtype=coeff.dtype)
    if n == 1:
        return np.array([-coeff[0]/coeff[1]])

    matrix = np.zeros((n,n), dtype=coeff.dtype)

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
