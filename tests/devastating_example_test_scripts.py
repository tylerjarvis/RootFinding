# Collection of methods for studying the devastating example for Möller-Setter
# matrix methods
# Author: Hayden Ringer

import yroots as yr
import numpy as np
from scipy.stats import ortho_group
from yroots.polynomial import MultiPower, MultiCheb
from yroots.Multiplication import ms_matrices, ms_matrices_cheb, ms_matrices_p, build_macaulay, multiplication
from yroots.MacaulayReduce import reduce_macaulay_svd, reduce_macaulay_qrt, reduce_macaulay_tvb, reduce_macaulay_p
from yroots.utils import ConditioningError
import scipy.linalg as la
import matplotlib.pyplot as plt
from yroots.subdivision import full_cheb_approximate, trim_coeffs

def condeig(A,eig,v):
    """Calculates the condition number of an eigenvalue of A"""
    n = A.shape[0]
    Q = hh(v)
    B = ((Q.conj().T)@A@Q)
    R = la.qr(B[1:,1:]-eig*np.eye(n-1))[1]
    z = la.solve_triangular(R,-B[0,1:],trans=2)
    return (1+la.norm(z))**.5

def hh(x):
    u = x.copy().astype('complex')
    u[0] += np.exp(1j*np.angle(x[0]))*la.norm(x)
    u = u/la.norm(u)
    return np.eye(len(u)) - 2*np.outer(u,u.conj())

def condeigs(A):
    n = A.shape[0]
    w,v = la.eig(A)
    cond = np.zeros(n)
    for i,eig in enumerate(w):
        Q = hh(v[:,i])
        B = ((Q.conj().T)@A@Q)
        R = la.qr(B[1:,1:]-eig*np.eye(n-1))[1]
        z = la.solve_triangular(R,-B[0,1:],trans=2)
        cond[i] = (1+la.norm(z))**.5
    return w,cond

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

def macaulayqeps(Q,eps,kind):
    if kind == 'power': func = polyqeps
    elif kind == 'spower': func = spolyqeps
    elif kind == 'cheb': func = chebpolyqeps
    else: func = chebspolyqeps
    polys = func(Q,eps)
    return build_macaulay(polys)

def macaulaypolys(polys):
    return build_macaulay(polys)

def redmacaulayqeps(Q,eps,kind,method,P=None):
    matrix,matrix_terms,cut = macaulayqeps(Q,eps,kind)
    if method == 'qrt': func = reduce_macaulay
    elif method == 'tvb': func = reduce_macaulay_tvb
    elif method == 'byu': func = reduce_macaulay_byu
    elif method == 'p':
        try:
            E, Q2 = reduce_macaulay_p(matrix, cut, P, 1e5)
        except ConditioningError as e:
            raise e
        return E,Q2,matrix_terms,cut
    try:
        E, Q2 = func(matrix, cut, 1e5)
    except ConditioningError as e:
        raise e
    return E,Q2,matrix_terms,cut

def redmacaulaypolys(polys,method,P=None):
    matrix,matrix_terms,cut = macaulaypolys(polys)
    if method == 'qrt': func = reduce_macaulay
    elif method == 'tvb': func = reduce_macaulay_tvb
    elif method == 'byu': func = reduce_macaulay_byu
    elif method == 'p':
        try:
            E, Q2 = reduce_macaulay_p(matrix, cut, P, 1e5)
        except ConditioningError as e:
            raise e
        return E,Q2,matrix_terms,cut
    try:
        E, Q2 = func(matrix, cut, 1e5)
    except ConditioningError as e:
        raise e
    return E,Q2,matrix_terms,cut

def msmatqeps(Q,eps,kind,method,P=None):
    E,Q2,matrix_terms,cut = redmacaulayqeps(Q,eps,kind,method,P)
    if method == 'qrt':
        if kind in ['power','spower']:
            return ms_matrices(E,Q2,matrix_terms,Q.shape[0])
        else:
            return ms_matrices_cheb(E,Q2,matrix_terms,Q.shape[0])
    else:
        return ms_matrices_p(E,Q2,matrix_terms,Q.shape[0],cut)

def msmatpolys(polys,method,P=None):
    E,Q2,matrix_terms,cut = redmacaulaypolys(polys,method,P)
    if method == 'qrt':
        if isinstance(polys[0],MultiPower):
            return ms_matrices(E,Q2,matrix_terms,len(polys))
        else:
            return ms_matrices_cheb(E,Q2,matrix_terms,len(polys))
    else:
        return ms_matrices_p(E,Q2,matrix_terms,len(polys),cut)

def mseigqeps(Q,eps,var,kind,method,P=None):
    m = msmatqeps(Q,eps,kind,method,P)[...,var]
    w,v = la.eig(m)
    if kind in ['power','cheb']:
        i = np.argmin(np.abs(w))
        return np.abs(w[i]), condeig(A,w[i],v[:,i])
    else:
        i = np.argmin(np.abs(w-1))
        return np.abs(w[i]-1), condeig(A,w[i],v[:,i])

def mseigpolys(polys,var,kind,method,P=None):
    m = msmatpolys(polys,method,P)[...,var]
    w,vl,vr = la.eig(m,left=True)
    if kind in ['power','cheb']:
        i = np.argmin(np.abs(w))
        return np.abs(w[i]), 1/np.abs(vl[:,i]@vr[:,i])
    else:
        i = np.argmin(np.abs(w-1))
        return np.abs(w[i]-1), 1/np.abs(vl[:,i]@vr[:,i])

def randq(dim):
    return ortho_group.rvs(dim)

def randpoly(dim,eps,kind):
    """Returns MultiPower objects for a random devastating example of dimension
    dim and parameter value of eps.
    """
    Q = randq(dim)
    if kind == 'power': func = polyqeps
    elif kind == 'spower': func = spolyqeps
    elif kind == 'cheb': func = chebpolyqeps
    else: func = chebspolyqeps
    return func(Q,eps)

def randmacaulay(dim,eps,kind):
    Q = randq(dim)
    return macaulayqeps(Q,eps,kind)

def randredmacaulay(dim,eps,kind,method):
    Q = randq(dim)
    return redmacaulayqeps(Q,eps,kind,method)

def randmsmat(dim,eps,kind):
    Q = randq(dim)
    return msmatqeps(Q,eps,kind,method)

def perturbpoly(dim,deg,basis,eps):
    if basis == 'power': MultiX = MultiPower
    else: MultiX = MultiCheb
    coeff = eps*rand_coeffs(dim,deg,1)[0,0]
    return MultiX(coeff)

def perturb(polys,eps):
    dim = polys[0].dim
    if isinstance(polys[0],MultiPower): basis = 'power'
    else: basis = MultiCheb
    newpolys = []
    for poly in polys:
        newpolys.append(poly + perturbpoly(dim,2,basis,eps))
    return newpolys

def chebapprox(polys,a,b,deg,atol=1e-15,rtol=1e-15,ttol=1e-15):
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
