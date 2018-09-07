"""
Code for determining if a system of polynomials has a root at infinity
August 17, 2018
"""
import numpy as np
from numalgsolve.polynomial import MultiPower
from numalgsolve.OneDimension import solve

def common_root_at_inf(polys):
    '''
    Tests if a system of bivariate power-basis polynomials has a common root at infinity.

    Parameters
    ----------
    polys (list, 2): polynomial objects

    returns
    -------
    bool : whether or not there is at least one root at infinity
    '''
    f, g = polys

    #get the roots at infinty of f
    inf_roots_f_d = roots_at_inf(f)

    #check if these are roots at infinity for g too
    common_roots = list()
    g_d = MultiPower(np.diag(np.diag(np.fliplr(pad_with_zeros(g.coeff)))[::-1]))
    for inf_root in inf_roots_f_d:
        if np.isclose(g_d(inf_root), 0):
            common_roots.append(inf_root)

    if len(common_roots) > 0:
        return True
    else:
        return False

def roots_at_inf(poly):
    '''
    Finds the roots at infinity of a homogenous, bivarite power-basis polynomial.

    Parameters
    ----------
    poly: a bivariate power-basis polynomial

    returns
    -------
    the roots at infinity of the polynomial (list of tuples). In the form (x,y). Since at infinity, z = 0
    '''
    #get the coefficients of the homogenous part of f: [x^d ... y^d]
    f_d = np.diag(np.fliplr(pad_with_zeros(poly.coeff)))[::-1]
    #find all x s.t. f_d(x,1) == 0
    #essentially, solve a 1D polynomial with coefficients from f_d
    x_coords = solve(MultiPower(f_d[::-1])) #1D solver takes list of coefficients from constant to high degree term
    inf_roots_f_d = list(zip(x_coords, np.ones(len(x_coords))))

    #check if f_d(x,1) == 0
    if np.isclose(f_d[0],0):
        inf_roots_f_d.append((1,0))

    return inf_roots_f_d

def pad_with_zeros(matrix):
    '''
    Extends a nonsquare matrix into a square matrix with zeros in it.
    e.g. if A is a tall matrix, returns [A|0]

    Parameters
    ----------
    matrix (np.array): a nonsquare matrix

    returns
    -------
    square matrix with zeros in it (np.array)
    '''
    m,n = matrix.shape
    l = max(m,n)
    square_matrix = np.zeros((l, l))
    square_matrix[:m,:n] = matrix
    return square_matrix
