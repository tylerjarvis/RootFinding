# Collection of methods for studying the devastating example for Möller-Setter
# matrix methods
# Author: Hayden Ringer

import yroots as yr
import numpy as np
from scipy.stats import ortho_group
from yroots.polynomial import MultiPower, MultiCheb, is_power
from yroots.Multiplication import ms_matrices, ms_matrices_cheb, build_macaulay, multiplication
from yroots.MacaulayReduce import reduce_macaulay, find_degree, add_polys
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

def redmacaulayqeps(Q,eps,kind):
    matrix,matrix_terms,cut = macaulayqeps(Q,eps,kind)
    try:
        E, Q = reduce_macaulay(matrix, cut, 1e5)
    except ConditioningError as e:
        raise e
    return E,Q,matrix_terms,cut

def msmatqeps(Q,eps,kind):
    E,Q2,matrix_terms,cut = redmacaulayqeps(Q,eps,kind)
    if kind in ['power','spower']:
        return ms_matrices(E,Q2,matrix_terms,Q.shape[0])
    else:
        return ms_matrices_cheb(E,Q2,matrix_terms,Q.shape[0])

def mseigqeps(Q,eps,var,kind):
    m = msmatqeps(Q,eps,kind)[...,var]
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
    return polyqeps(Q,eps,kind)

def randmacaulay(dim,eps,kind):
    Q = randq(dim)
    return macaulayqeps(Q,eps,kind)

def randredmacaulay(dim,eps,kind):
    Q = randq(dim)
    return redmacaulayqeps(Q,eps,kind)

def randmsmat(dim,eps,kind):
    Q = randq(dim)
    return msmatqeps(Q,eps,kind)

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
