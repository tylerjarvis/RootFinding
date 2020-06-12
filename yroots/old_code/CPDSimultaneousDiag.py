"""
Methods for using the tensor CPD to comute the eigenvalues of commuting multiplication
matrices simultaneously. Based off matlab code written by Telen and Tensorlab,
and Hayden's implicit rotation code. These functions were never completed, but
could be useful to revist in the future to compare for speed and accuracy against
the current simultaneous diagonalization code.
Suzanna Parkinson, 6/12/2020
"""
from scipy.linalg import solve_triangular, eig, schur, svd
import numpy as np

def msroots_GEVD_rotated(M):
    """Computes the roots to a system via the eigenvalues of the Möller-Stetter
    matrices. Implicitly performs a random rotation of the coordinate system
    to avoid repeated eigenvalues arising from special structure in the underlying
    polynomial system. Solves using the cpd gevd algorithm.

    Parameters
    ----------
    M : (n,n,dim) ndarray
        Array containing the nxn Möller-Stetter matrices, where the matrix
        corresponding to multiplication by x_i is M[...,i]

    Returns
    -------
    roots : (n,dim) ndarray
        Array containing the approximate roots of the system, where each row
        is a root.
    """
    num_roots,dim = M.shape[1:]

    # perform a random rotation with a random orthogonal Q
    Q,c = get_Q_c(dim) #todo... don't need c really
    M = (Q@M[...,np.newaxis])[...,0]

    #stack on a copy of the identity matrix
    #todo there's definitely a more efficient way to do this
    M_ = np.stack((*[M[...,i] for i in range(dim)],np.eye(num_roots)),axis=2)
    C = specialized_cpd_gevd(M_,rank=num_roots)
    #rescale result
    C[:-1] /= C[-1]
    # Rotate back before returning, transposing to match expected shape
    return (Q.T@C[:-1]).T

def specialized_cpd_gevd(T,rank):
    num_roots,num_slices = T.shape[1:]
    V,S,sv = mlsvd(T);
    GenVecs, GenVals = la.eig(S[:,:,1].T,S[:,:,2].T)
    T1 = T.reshape(num_roots,num_roots*num_slices)
    X = T1.T@np.conj(V[1])*GenVecs
    for r in range(rank):
        u,s = mlsvd(X[:,r].reshape((num_roots,num_slices)),[1,1])
        C = np.concatenate(C,u[1],axis=1)
    return C

def mlsvd(T):
    shape_tens = np.array(T.shape)
    dim = T.ndim
    U,S = [0]*dim,T
    for n in range(dim):
        print(n,'of',dim)
        U[n],s,v = svd(tens2mat(S,mode_row=n),full_matrices=False)
        print(v.shape)
        print(shape_tens)
        S = mat2tens(v,shape_tens,mode_col=n)
    return U,S

def mat2tens(M,shape_tens,mode_row=None,mode_col=None):
    # if mode_col is None:
    #     # mode_col = np.delete
    # TODO Fix
    return M.reshape(shape_tens)

def tens2mat(T,mode_row):
    col_list = list(T.shape)
    row = col_list.pop(mode_row)
    return T.reshape((row,np.prod(col_list)))
