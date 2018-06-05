# bezout.py

import numpy as np
from scipy.sparse import diags, kron, eye
import numalgsolve.utils as utils
import scipy.linalg as sLA
import time

def bivariate_roots(f, g):
    """Calculates the common roots of f and g using the bezout resultant method.

    Parameters
    ----------
    f, g : MultiCheb objects

    Returns
    -------
    roots

    """
    F, G = utils.match_size(f.coeff, g.coeff)
    n = max(F.shape)
    # F, G = np.zeros((n,n)), np.zeros((n,n))
    # F = F + utils.match_size(F, f_coeffs)[1] # Square matrix for f
    # G = G + utils.match_size(G, g_coeffs)[1] # Square matrix for g

    A = np.zeros((n-1, n-1, 2*n-1))
    a, b, c = (np.vstack([np.ones((n-1,1)), 2])/2).T, np.zeros(n), np.ones(n)/2
    for i in range(1, n+1):
        for j in range(1, n+1):
            AA = np.array([[0] + list(F[:,i-1][::-1])])
            v = G[:,j-1][::-1].reshape((-1,1))
            X,ignored = DLP(AA,v,a,b,c)
            if i==1 or j==1:
                cc = np.zeros(max(i,j))
                cc[0] = 1
            else:
                cc = np.zeros(i+j-1)
                cc[-1] = .5
                cc[abs(i-j)] = .5
                cc = cc[::-1]
            for k in range(len(cc)):
                A[:,:,-1-k] = A[:,:,-1-k] + X[1:,1:]*cc[-1-k]

    nrmA = np.linalg.norm(A[:,:,-1], 'fro')
    for ii in range(A.shape[2]):
        if np.linalg.norm(A[:,:,ii], 'fro')/nrmA > 1e-20:
            break
    A = A[:,:,ii:]

    ns = A.shape
    AA = A.reshape(ns[0], ns[1]*ns[2], order='F')

    n = ns[0]
    v = np.random.rand(n, 1)
    a, b, c = (np.vstack([np.ones((n-1,1)), 2])/2).T, np.zeros(n), np.ones(n)/2
    X,Y = DLP(AA,v,a,b,c)
    yvals,V = sLA.eig(Y,-X)

    y = yvals
    x = np.divide(V[-2,:],V[-1,:])

    t = np.copy(x)
    x = x[np.logical_and(np.logical_and(np.imag(y)==0,abs(y)<1),abs(x)<1)]
    y = y[np.logical_and(np.logical_and(np.imag(y)==0,abs(y)<1),abs(t)<1)]

    return list(zip(x,y))

def DLP(AA, v, a, b, c):
    '''DLP constructs the DL pencil with ansatz vector v.

    [X,Y] = DLP(AA,V,A,B,C) returns the DL pencil with the orthogonal
    basis defined by the recurrence relations A,B,C.

    Parameters
    ----------
    a, b, c : numpy array, row vectors

    '''
    n, m = AA.shape
    k = m // n - 1
    s = n * k
    M = diags(np.vstack([a,b,c]),[0,1,2],(k,k+1))
    M = kron(M, np.eye(n)).tocsr()

    S = np.kron(v, AA)
    for j in range(k):
        jj = np.array([num for num in range(n*j, n*j+n)])
        AA[:,jj] = AA[:,jj].conj().T
    T = np.kron(v.conj().T, AA.conj().T)
    R = (M.conj().T @ S) - (T @ M)

    X, Y = np.zeros((s,s)), np.zeros((s,s))
    ii = np.array([num for num in range(n, n+s)])
    nn = np.array([num for num in range(n)])
    Y[nn,:], X[nn,:] = R[nn][:,ii]/M[0,0], T[nn,:]/M[0,0]
    index = np.array([num for num in range(n,s+n)])
    Y[nn+n,:] = (R[nn+n][:,ii] - (M[0,n] * Y[nn,:]) + (Y[nn,:] @ M[:,index])) / M[n,n]
    X[nn+n,:] = (T[nn+n,:] - Y[nn,:] - (M[0,n] * X[nn,:])) / M[n,n]

    for i in range(3,k+1):
        ni = n*i-1
        jj = np.array([num for num in range(ni-n+1, ni+1)])
        j0 = jj-2*n
        j1 = jj-n
        M0, M1, m = M[ni-2*n, ni], M[ni-n, ni], M[ni, ni]
        Y0, Y1, X0, X1 = Y[j0,:], Y[j1,:], X[j0,:], X[j1,:]
        index = np.array([num for num in range(n, s+n)])
        Y[jj,:] = (R[jj][:,ii] - (M1 * Y1) - (M0 * Y0) + (Y1 @ M[:,index])) / m
        X[jj,:] = (T[jj,:] - Y1 - (M1 * X1) - (M0 * X0)) / m

    return X, Y
