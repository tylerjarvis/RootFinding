# bezout.py

import numpy as np
from scipy.sparse import spdiags

def DLP(AA, v, a, b, c):
    '''DLP constructs the DL pencil with ansatz vector v.

    [X,Y] = DLP(AA,V,A,B,C) returns the DL pencil with the orthogonal
    basis defined by the recurrence relations A,B,C.
    
    '''
    n, m = AA.shape
    k = m // n - 1 # Should this not be integer divison?
    s = n * k
    M = spdiags([a,b,c],[0,1,2],k,k+1)
    M = np.kron(M, np.eye(n))

    S = np.kron(v, AA)
    for j in range(k): # Maybe change this to k-1?
        jj = np.array([num for num in range(n*j, n*j+n)])
        AA[:,jj] = AA[:,jj].getH()
    T = np.kron(v.getH(), AA.getH())
    R = M.getH() @ S-T @ M

    X = np.zeros((s,s))
    Y = X
    ii = np.array([num for num in range(n, n+s)])
    nn = np.array([num for num in range(n)])
    Y[nn-1,:] = np.linalg.lstsq(M[0].T, R[nn,ii].T)
    X[nn-1,:] = np.linalg.lstsq(M[0].T, T[nn,:].T)
    Y[nn+n-1,:] = (R[nn+n,ii]-M[0,n] @ Y[nn,:] + Y[nn,:] @ M[:,np.array([num for \
                                          num in range(n,s+n)])]) / M[n,n]
    X[nn+n-1,:] = (T[nn+n,:] - Y[nn,:] - M[0,n] @ X[nn,:]) / M[n,n]

    for i in range(2,k):
        ni = n*i
        jj = np.array([num for num in range(ni-n, ni)])
        j0 = jj-2*n
        j1 = jj-n
        M0, M1, m = M[ni-2*n-1, ni-1], M[ni-n-1, ni-1], M[ni-1, ni-1]
        Y0, Y1, X0, X1 = Y[j0-1,:], Y[j1-1,:], X[j0-1,:], X[j1-1,:]
        Y[jj-1,:] = (R[jj,ii] - M1 @ Y1 - M0 @ Y0 + Y1 @ M[:,\
                                np.array([num for num in range(n:s+n)]) / m
        X[jj,:] = (T[jj,:] - Y1 - M1 @ X1 - M0 @ X0) / m

    return X, Y
