import numpy as np
import itertools
from numalgsolve import OneDimension as oneD
from numalgsolve.polynomial import MultiCheb, MultiPower, is_power
from numalgsolve.Division import division
from numalgsolve.Multiplication import multiplication
from numalgsolve.utils import Term, get_var_list, divides, MacaulayError, InstabilityWarning, match_size, match_poly_dimensions

def solve(polys, MSmatrix=0, eigvals=True, verbose=False):
    '''
    Finds the roots of the given list of polynomials.

    Parameters
    ----------
    polys : list of polynomial objects
        Polynomials to find the common roots of.
    MSmatrix : int
        Controls which Moller-Stetter matrix is constructed
        For a univariate polynomial, the options are:
            0 (default) -- The companion or colleague matrix, rotated 180 degrees
            1 -- The unrotated companion or colleague matrix
            -1 -- The inverse of the companion or colleague matrix
        For a multivariate polynomial, the options are:
            0 (default) -- The Moller-Stetter matrix of a random polynomial
            Some positive integer i <= dimension -- The Moller-Stetter matrix of x_i, where variables are index from x1, ..., xn
            Some negative integer i >= -dimension -- The Moller-Stetter matrix of x_i-inverse
    eigvals : bool
        Whether to compute roots of univariate polynomials from eigenvalues (True) or eigenvectors (False).
        Roots of multivariate polynomials are always comptued from eigenvectors
    verbose : bool
        Prints information about how the roots are computed.

    returns
    -------
    roots : numpy array
        The common roots of the polynomials. Each row is a root.
    '''
    polys = match_poly_dimensions(polys)
    # Determine polynomial type and dimension of the system
    poly_type = is_power(polys, return_string = True)
    dim = polys[0].dim

    if dim == 1:
        if len(polys) == 1:
            return oneD.solve(polys[0], MSmatrix=MSmatrix, eigvals=eigvals, verbose=verbose)
        else:
            zeros = np.unique(oneD.solve(polys[0], MSmatrix=MSmatrix, eigvals=eigvals, verbose=verbose))
            #Finds the roots of each succesive polynomial and checks which roots are common.
            for poly in polys[1:]:
                if len(zeros) == 0:
                    break
                zeros2 = np.unique(oneD.solve(poly, MSmatrix=MSmatrix, eigvals=eigvals, verbose=verbose))
                common = list()
                tol = 1.e-10
                for zero in zeros2:
                    spot = np.where(np.abs(zeros-zero)<tol)
                    if len(spot[0]) > 0:
                        common.append(zero)
                zeros = common
            return zeros
    else:
        if MSmatrix < 0:
            return division(polys, verbose=verbose, divisor_var=-MSmatrix-1)
        else:
            return multiplication(polys, verbose=verbose, MSmatrix=MSmatrix)
