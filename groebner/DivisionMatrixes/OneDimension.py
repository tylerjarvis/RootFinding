import numpy as np
from scipy.linalg import eig
from groebner.polynomial import MultiCheb, MultiPower

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
        if method == 'M':
            return multPower(poly.coeff)
        else:
            return divPower(poly.coeff)
    else:
        if method == 'D':
            return multCheb(poly.coeff)
        else:
            return divCheb(poly.coeff)

def multPower(coeffs):
    n = len(coeffs)
    col = -coeffs[:-1]/coeffs[-1]
    col = col.reshape(n-1,1)
    mMatrix = np.hstack((np.vstack((np.zeros(n-2),np.eye(n-2))),col))
    vals = eig(mMatrix, right=False)
    return vals

def divPower(coeffs):
    n = len(coeffs)
    col = -coeffs[1:]/coeffs[0]
    col = col.reshape(n-1,1)
    dMatrix = np.hstack((col,np.vstack((np.eye(n-2),np.zeros(n-2)))))
    vals = eig(dMatrix, right=False)
    return 1/vals

def multCheb(coeffs):
    n = len(coeffs)
    mMatrix = np.zeros((n-1,n-1))
    mMatrix[1][0] = 1
    mMatrix[:-1,1:] += np.eye(n-2)/2
    mMatrix[2:,1:-1] += np.eye(n-3)/2
    mMatrix[:,-1] -= .5*coeffs[:-1]/coeffs[-1]
    vals = eig(mMatrix, right=False)
    return vals

def divCheb(coeffs):
    n = len(coeffs)
    curr = coeffs.copy()
    xinv = np.zeros(n-1)
    for i in range(1,n-1)[::-1]:
        val = -curr[i+1]
        curr[i+1] += val
        curr[i-1] += val
        xinv[i]+=2*val
    temp = -curr[1]
    curr[1]+=temp
    xinv[0]+=temp
    xinv/=curr[0]
    xinv
    dMatrix = np.zeros((n-1,n-1))
    for col in range(n-1):
        if col%2==0:
            if col%4==0:
                dMatrix[:,col]+=xinv
            else:
                dMatrix[:,col]-=xinv
        else:
            if (col-1)%4==0:
                dMatrix[0,col]+=1
            else:
                dMatrix[0,col]-=1
        sign = 1
        for spot in range(col%2+1,col,2)[::-1]:
            dMatrix[spot,col]+=2*sign
            sign*=-1
    vals = eig(dMatrix, right=False)
    return 1/vals