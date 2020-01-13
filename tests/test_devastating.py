# Collection of methods for studying the devastating example for Möller-Setter
# matrix methods
# Author: Hayden Ringer

import yroots as yr
import numpy as np
from scipy.stats import ortho_group
from yroots.polynomial import MultiPower, MultiCheb, is_power
from yroots.Multiplication import MSMultMatrix, create_matrix
from yroots.MacaulayReduce import rrqr_reduceMacaulay, find_degree, add_polys
from yroots.utils import ConditioningError
import scipy.linalg as la
import matplotlib.pyplot as plt
# from yroots.subdivision import full_cheb_approximate, trim_coeffs

macheps = 2.220446049250313e-16
def condeig(A):
    """Calculates the condition numbers of the eigenvalues of A"""
    n = A.shape[0]
    w, vl, vr = la.eig(A,left=True)
    vl, vr = vl/la.norm(vl,axis=0), vr/la.norm(vr,axis=0)
    out = np.empty(n)
    for i in range(n):
        out[i] = 1/np.abs(np.dot(vl[:,i],vr[:,i]))
    return out

def build_macaulay(polys):
    power = is_power(polys)
    dim = polys[0].dim
    poly_coeff_list = []
    degree = find_degree(polys)

    for poly in polys:
        poly_coeff_list = add_polys(degree, poly, poly_coeff_list)

    #Creates the matrix
    matrix, matrix_terms, cuts = create_matrix(poly_coeff_list, degree, dim,[])
    return matrix, matrix_terms, cuts

def polyqeps(Q,eps):
    dim = Q.shape[0]
    polys = []
    for i in range(dim):
        coeff = np.zeros([3]*dim)
        spot = [0]*dim
        for j in range(dim):
            spot[j] = 1
            coeff[tuple(spot)] = Q[i,j]*eps
            spot[j] = 0
        spot[i] = 2
        coeff[tuple(spot)] = 1
        polys.append(MultiPower(coeff))
    return polys

def chebpolyqeps(Q,eps):
    dim = Q.shape[0]
    polys = []
    for i in range(dim):
        coeff = np.zeros([3]*dim)
        spot = [0]*dim
        coeff[tuple(spot)] = .5
        for j in range(dim):
            spot[j] = 1
            coeff[tuple(spot)] = Q[i,j]*eps
            spot[j] = 0
        spot[i] = 2
        coeff[tuple(spot)] = .5
        polys.append(MultiCheb(coeff))
    return polys

def spolyqeps(Q,eps):
    dim = Q.shape[0]
    polys = []
    const = (np.eye(dim)-eps*Q).sum(axis=1)
    for i in range(dim):
        coeff = np.zeros([3]*dim)
        spot = [0]*dim
        coeff[tuple(spot)] = const[i]
        for j in range(dim):
            spot[j] = 1
            coeff[tuple(spot)] = Q[i,j]*eps
            if i == j:
                coeff[tuple(spot)] -= 2
            spot[j] = 0
        spot[i] = 2
        coeff[tuple(spot)] = 1
        polys.append(MultiPower(coeff))
    return polys

def chebspolyqeps(Q,eps):
    dim = Q.shape[0]
    polys = []
    const = (1.5*np.eye(dim)-eps*Q).sum(axis=1)
    for i in range(dim):
        coeff = np.zeros([3]*dim)
        spot = [0]*dim
        coeff[tuple(spot)] = const[i]
        for j in range(dim):
            spot[j] = 1
            coeff[tuple(spot)] = Q[i,j]*eps
            if i == j:
                coeff[tuple(spot)] -= 2
            spot[j] = 0
        spot[i] = 2
        coeff[tuple(spot)] = .5
        polys.append(MultiCheb(coeff))
    return polys

def macaulayqeps(Q,eps):
    polys = polyqeps(Q,eps)
    return build_macaulay(polys)

def smacaulayqeps(Q,eps):
    polys = spolyqeps(Q,eps)
    return build_macaulay(polys)

def chebmacaulayqeps(Q,eps):
    polys = chebpolyqeps(Q,eps)
    return build_macaulay(polys)

def chebsmacaulayqeps(Q,eps):
    polys = chebspolyqeps(Q,eps)
    return build_macaulay(polys)

def redmacaulayqeps(Q,eps):
    matrix,matrix_terms,cuts = macaulayqeps(Q,eps)
    try:
        matrix, matrix_terms = rrqr_reduceMacaulay(matrix, matrix_terms, cuts, 1e20, 1e-14)
    except ConditioningError as e:
        raise e
    return matrix,matrix_terms

def sredmacaulayqeps(Q,eps):
    matrix,matrix_terms,cuts = smacaulayqeps(Q,eps)
    try:
        matrix, matrix_terms = rrqr_reduceMacaulay(matrix, matrix_terms, cuts, 1e20, 1e-14)
    except ConditioningError as e:
        raise e
    return matrix,matrix_terms

def msmatqeps(Q,eps,var):
    polys = polyqeps(Q,eps)
    return MSMultMatrix(polys,"MultiPower",1e20,1e-14,MSmatrix=var)[:2]

def smsmatqeps(Q,eps,var):
    polys = spolyqeps(Q,eps)
    return MSMultMatrix(polys,"MultiPower",1e20,1e-14,MSmatrix=var)[:2]

def mseigqeps(Q,eps,var):
    m = msmatqeps(Q,eps,var)[0]
    w,vl,vr = la.eig(m,left=True)
    i = np.argmin(np.abs(w))
    return np.abs(w[i]), 1/np.abs(vl[:,i]@vr[:,i])

def smseigqeps(Q,eps,var):
    m = smsmatqeps(Q,eps,var)[0]
    w,vl,vr = la.eig(m,left=True)
    i = np.argmin(np.abs(w-1))
    return np.abs(w[i]-1), 1/np.abs(vl[:,i]@vr[:,i])

def newsmseigqeps(Q,eps,var):
    m = newmsqeps(Q,eps,var)
    w,vl,vr = la.eig(m,left=True)
    i = np.argmin(np.abs(w-1))
    return np.abs(w[i]-1), 1/np.abs(vl[:,i]@vr[:,i])

def randq(dim):
    return ortho_group.rvs(dim)

def randpoly(dim,eps):
    """Returns MultiPower objects for a random devastating example of dimension
    dim and .
    """
    Q = randq(dim)
    return polyqeps(Q,eps)

def randspoly(dim,eps):
    """Returns MultiPower objects for a random devastating example of dimension
    dim and .
    """
    Q = randq(dim)
    return spolyqeps(Q,eps)

def randmacaulay(dim,eps):
    Q = randq(dim)
    return macaulayqeps(Q,eps)

def randsmacaulay(dim,eps):
    Q = randq(dim)
    return smacaulayqeps(Q,eps)

def randredmacaulay(dim,eps):
    Q = randq(dim)
    return redmacaulayqeps(Q,eps)

def randsredmacaulay(dim,eps):
    Q = randq(dim)
    return sredmacaulayqeps(Q,eps)

def randmsmat(dim,eps,var):
    Q = randq(dim)
    return msmatqeps(Q,eps,var)

def randsmsmat(dim,eps,var):
    Q = randq(dim)
    return smsmatqeps(Q,eps,var)

def chebapprox(polys,a,b,deg,atol=1e-8,rtol=1e-12,ttol=1e-10):
    chebcoeff = []
    inf_norms = []
    errors = []
    for poly in polys:
        coeff,_,inf_norm,error = full_cheb_approximate(poly,a,b,deg,atol,rtol)
        chebcoeff.append(coeff)
        inf_norms.append(inf_norm)
        errors.append(error)
    chebcoeff = trim_coeffs(chebcoeff,atol,rtol,ttol,inf_norms,errors)[0]
    chebpolys = []
    for coeff in chebcoeff:
        chebpolys.append(MultiCheb(coeff))
    return chebpolys

def chebmseig(m,a,b):
    root = 2*(1-a[0])/(b[0]-a[0])-1
    w,vl,vr = la.eig(m,left=True)
    i = np.argmin(np.abs(w-root))
    return np.abs(w[i]-root), 1/np.abs(vl[:,i]@vr[:,i])


def mx2d(Q,eps):
    M = np.zeros((4,4))
    M[[1,0],[3,2]] = 1
    M[[1,0],[1,0]] = -eps*Q[0,0]
    M[2,1] = -eps*Q[0,1]
    M[1,0] = (eps**2)*Q[0,1]*Q[1,0]
    M[2,0] = (eps**2)*Q[0,1]*Q[1,1]
    return M

def smx2d(Q,eps):
    A = eps*Q[0,0]-2
    B = eps*Q[0,1]
    C = eps*Q[1,0]
    D = eps*Q[1,1]-2
    E = eps*(Q[0,0]+Q[0,1])-1
    F = eps*(Q[1,0]+Q[1,1])-1
    M = np.array([[0,E,0,-B*F],
                  [1,-A,0,B*C],
                  [0,-B,0,B*D+E],
                  [0,0,1,-A]])

    M = np.array([[-A,0,1,0],
                  [B*C,-A,0,1],
                  [B*D+E,-B,0,0],
                  [-B*F,E,0,0]])
    return M

def new_reduceMacaulay(matrix, cut, max_cond=1e6):

    # QR reduce the highest-degree columns
    M = matrix.copy()
    cond_num = np.linalg.cond(M[:,:cut])
    if cond_num > max_cond:
        raise ConditioningError(f"Condition number of the Macaulay primary submatrix is {cond_num}")
    Q,M[:,:cut] = la.qr(M[:,:cut])
    M[:,cut:] = Q.T @ M[:,cut:]
    del Q

    # If the matrix is "tall", compute an orthogonal transformation of the remaining
    # columns, generating a new polynomial basis
    if cut < M.shape[0]:
        Q = la.qr(M[cut:,cut:].T,pivoting=True)[0]
        M[:cut,cut:] = M[:cut,cut:] @ Q # Apply column transform

    # Check numerical rank and chop the matrix
    s = la.svd(M, compute_uv=False)
    tol = max(M.shape)*s[0]*macheps
    rank = len(s[s>tol])
    M = M[:cut]
    M[:,cut:rank] = 0
    M[:,rank:] = la.solve_triangular(M[:,:cut],M[:,rank:])

    return M[:,rank:],Q[:,rank-M.shape[1]:]

def indexarray(matrix_terms,m,var):
    mults = matrix_terms[m:].copy()
    mults[:,var] += 1
    return np.argmin(np.abs(mults[:,np.newaxis] - matrix_terms[np.newaxis]).sum(axis=-1),axis=1)

def indexarray_cheb(matrix_terms,m,var):
    up = matrix_terms[m:].copy()
    up[:,var] += 1
    down = matrix_terms[m:].copy()
    down[:,var] -= 1
    down[down[:,var]==-1,var] += 2
    arr1 = np.argmin(np.abs(up[:,np.newaxis] - matrix_terms[np.newaxis]).sum(axis=-1),axis=1)
    arr2 = np.argmin(np.abs(down[:,np.newaxis] - matrix_terms[np.newaxis]).sum(axis=-1),axis=1)
    return arr1,arr2

def ms_matrices(E,Q,matrix_terms,dim):
    n = Q.shape[1]
    m = E.shape[0]
    M = np.empty((n,n,dim))
    A = np.hstack((-E.T,Q.T))
    for i in range(dim):
        arr = indexarray(matrix_terms,m,i)
        M[...,i] = A[:,arr]@Q
    return M

def ms_matrices_cheb(E,Q,matrix_terms,dim):
    n = Q.shape[1]
    m = E.shape[0]
    M = np.empty((n,n,dim))
    A = np.hstack((-E.T,Q.T))
    for i in range(dim):
        arr1,arr2 = indexarray_cheb(matrix_terms,m,i)
        M[...,i] = .5*(A[:,arr1]+A[:,arr2])@Q
    return M

def roots(M):
    # perform a random rotation
    dim = M.shape[-1]
    Q = ortho_group.rvs(dim)
    M = (Q@M[...,np.newaxis])[...,0]

    eigs = np.empty((dim,M.shape[0]),dtype='complex')
    T,Z = la.schur(M[...,0],output='complex')
    eigs[0] = np.diag(T)
    for i in range(1,dim):
        T = (Z.conj().T)@(M[...,i])@Z
        eigs[i] = np.diag(T)
    return (Q.T@eigs).T

def mx(E,Q,matrix_terms,var):
    arr = indexarray(matrix_terms,E.shape[0],var)
    return np.hstack((-E.T,Q.T))[:,arr]@Q
# def roots(M):
#     w,V = la.eig(M[0])
#     roots = np.empty((len(w),len(M)),dtype='complex')
#     roots[:,0] = w
#     for i in range(1,len(M)):
#         w = la.eig(M[i],right=False)
#
#         w1 = np.mean((M[i]@V)/V,axis=0)
#         print(w1,"\n-----")
#         roots[:,i] = w[np.argsort(np.abs(np.subtract.outer(w,w1)),axis=0)[0]]
#     print(roots)
#     return roots
#
# def Roots(Q,eps):
#     M = []
#     for i in range(len(Q)):
#         print(Q)
#         M.append(newmsqeps(Q,eps,i))
#     return roots(M)

def newmsqeps(Q,eps,var):
    M,matrix_terms,cuts = smacaulayqeps(Q,eps)
    E,Q2,d,n = new_reduceMacaulay(M,cuts[0])
    return mx(E,Q2,d,n,matrix_terms,var)
