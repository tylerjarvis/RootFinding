"""
Code for determining if a system of polynomials has a root at infinity
August 17, 2018
"""
import numpy as np
from yroots.polynomial import MultiPower
from yroots.OneDimension import solve

def common_root_at_inf(polys, return_root=False):
    '''
    Tests if a system of upper-triagular bivariate power-basis polynomials has a common root at infinity.

    Parameters
    ----------
    polys (list, 2): polynomial objects
    return_root (bool): whether to return the first root at infinity found, if any exist

    returns
    -------
    bool : whether or not there is at least one root at infinity
    root : (tuple) the first root at infinity it found, if any exist
    '''
    f, g = polys
    f_is_first = True

    #get the roots at infinty of the first poly
    try:
        inf_roots = roots_at_inf(f)
    except:
        inf_roots = roots_at_inf(g)
        f_is_first = False

    #check if these are roots at infinity for the second poly too
    if f_is_first:
        second_poly = MultiPower(np.diag(np.diag(np.fliplr(pad_with_zeros(g.coeff)))[::-1]))
    else:
        second_poly = MultiPower(np.diag(np.diag(np.fliplr(pad_with_zeros(f.coeff)))[::-1]))
    for inf_root in inf_roots:
        if np.isclose(second_poly(inf_root), 0):
            if return_root:
                return True, inf_root
            else:
                return True
    return False

def roots_at_inf(f):
    '''
    Finds the roots at infinity of a homogenous, bivarite power-basis polynomial.

    Parameters
    ----------
    poly: a bivariate power-basis polynomial

    returns
    -------
    the roots at infinity of the polynomial (list of tuples). In the form (x,y). Since at infinity, z = 0
    '''
    #get the coefficients of the homogenous part of f: [y^d ... x^d]
    f_d_coefs = np.diag(np.fliplr(pad_with_zeros(f.coeff)))

    #find all x s.t. f_d(x,1) == 0
    #essentially, set y = 1 and solve a 1D polynomial with coefficients from f_d
    #if there is only 1 nonzero coefficient in this 1D polynomial have to handle separately
    if sum(f_d_coefs != 0) == 1:
        if np.where(f_d_coefs != 0) == np.array([0]): #if the nonzero coef in the 1D poly is the constant, only root at inf is (1,0)
            return [(1,0)]
        else: #f_d(x,y) = x^a y^b, so 0 = f_d(x,1) == x^a implies x = 0
            x_coords = np.array([0])
    else:
        x_coords = solve(MultiPower(f_d_coefs))

    inf_roots_f_d = list(zip(x_coords, np.ones(len(x_coords))))

    #check if f_d(1,0) == 0
    #f_d(1,0) = the coef of x^d
    if np.isclose(f_d_coefs[-1],0):
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
