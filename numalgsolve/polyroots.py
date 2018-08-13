import numpy as np
import itertools
from numalgsolve import OneDimension as oneD
from numalgsolve.polynomial import MultiCheb, MultiPower, is_power
from numalgsolve.Division import division
from numalgsolve.Multiplication import multiplication
from numalgsolve.utils import Term, get_var_list, divides, TVBError, InstabilityWarning, match_size, match_poly_dimensions

def solve(polys, method = 'multR', verbose=False):
    '''
    Finds the roots of the given list of polynomials.

    Parameters
    ----------
    polys : list of polynomial objects
        Polynomials to find the common roots of.
    method : string
        The root finding method to be used. Can be either 'mult', 'div', or 'multR', 'multrand'.
            'mult': makes a M_x matrix
            'multR': makes a M_x matrix and rotates it 180 degrees (for a univariate polynomial)
            'div': makes a division M_1/x matrix
            'multrand': Makes a M_f matrix of a pseudorandom polynomial f
    returns
    -------
    roots : numpy array
        The common roots of the polynomials. Each row is a root.
    '''
    polys = match_poly_dimensions(polys)
    # Determine polynomial type
    poly_type = is_power(polys, return_string = True)
    dim = polys[0].dim

    if dim == 1:
        if len(polys) == 1:
            return oneD.solve(polys[0], method, verbose=verbose)
        else:
            zeros = np.unique(oneD.solve(polys[0], method))
            #Finds the roots of each succesive polynomial and checks which roots are common.
            for poly in polys[1:]:
                if len(zeros) == 0:
                    break
                zeros2 = np.unique(oneD.solve(poly, method))
                common = list()
                tol = 1.e-10
                for zero in zeros2:
                    spot = np.where(np.abs(zeros-zero)<tol)
                    if len(spot[0]) > 0:
                        common.append(zero)
                zeros = common
            return zeros
    else:
        if method == 'mult':
            return multiplication(polys, verbose=verbose, rand_poly=False, rotate=False)
        elif method == 'multR'
            return multiplication(polys, verbose=verbose, rand_poly=False, rotate=True)
        elif method == 'multrand':
            return multiplication(polys, verbose=verbose, , rand_poly=True, rotate=False)
        else:
            return division(polys, verbose=verbose)
