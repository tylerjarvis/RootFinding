# bezout.py

import numpy as np
from scipy.sparse import diags, kron, eye
import groebner.utils as utils

def bivariate_roots(f, g):
    """Calculates the common roots of f and g using the bezout resultant method.

    Parameters
    ----------
    f, g : MultiCheb objects

    Returns
    -------
    roots

    """
    f_coeffs, g_coeffs = utils.match_size(f.coeff, g.coeff)
    n = max(f_coeffs.shape)
    F, G = np.zeros((n,n)), np.zeros((n,n))
    F = F + utils.match_size(F, f_coeffs)[1] # Square matrix for f
    G = G + utils.match_size(G, g_coeffs)[1] # Square matrix for g

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
    AA = A.reshape(ns[0], ns[1]*ns[2])

    n = ns[0]
    v = np.random.rand(n, 1)
    a, b, c = (np.vstack([np.ones((n-1,1)), 2])/2).T, np.zeros(n), np.ones(n)/2
    # X,Y = DLP(AA,v,a,b,c)
    # V,yvals = np.linalg.eig(Y,-X)
    #
    # y = np.diagonal(yvals)

    return A

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
    M = kron(M, eye(n))
    M = M.tocsr()

    S = kron(v, AA)
    S = S.tocsr()
    for j in range(k):
        jj = np.array([num for num in range(n*j, n*j+n)])
        AA[:,jj] = AA[:,jj].conj().T
    T = np.kron(v.conj().T, AA.conj().T)
    R = (M.conj().T @ S) - (T @ M)
    R = np.array(R)

    X = np.zeros((s,s))
    Y = np.copy(X)
    ii = np.array([num for num in range(n, n+s)])
    nn = np.array([num for num in range(n)])
    Y[nn,:] = R[nn,ii]/M[0,0]
    X[nn,:] = T[nn,:]/M[0,0]
    index = np.array([num for num in range(n,s+n)])
    Y[nn+n,:] = (R[nn+n,ii] - (M[0,n] * Y[nn,:]) + (Y[nn,:] @ M[:,index])) / M[n,n]
    X[nn+n,:] = (T[nn+n,:] - Y[nn,:] - (M[0,n] * X[nn,:])) / M[n,n]

    for i in range(3,k+1):
        ni = n*i
        jj = np.array([num for num in range(ni-n, ni)])
        j0 = jj-2*n
        j1 = jj-n
        M0, M1, m = M[ni-2*n-1, ni-1], M[ni-n-1, ni-1], M[ni-1, ni-1]
        Y0, Y1, X0, X1 = Y[j0,:], Y[j1,:], X[j0,:], X[j1,:]
        index = np.array([num for num in range(n, s+n)])
        Y[jj,:] = (R[jj,ii] - (M1 * Y1) - (M0 * Y0) + (Y1 @ M[:,index])) / m
        X[jj,:] = (T[jj,:] - Y1 - (M1 * X1) - (M0 * X0)) / m

    return X, Y
