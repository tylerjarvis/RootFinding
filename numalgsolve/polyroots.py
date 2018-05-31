import numpy as np
import itertools
from numalgsolve import OneDimension as oneD
from numalgsolve.polynomial import MultiCheb, MultiPower, is_power
from numalgsolve.TVBCore import TelenVanBarel
from numalgsolve.PowerDivision import division_power
from numalgsolve.Division import division_cheb
from numalgsolve.Multiplication import multiplication
from numalgsolve.utils import Term, get_var_list, divides, TVBError, InstabilityWarning, match_size, match_poly_dimensions

def solve(polys, method = 'mult'):
    '''
    Finds the roots of the given list of polynomials.

    Parameters
    ----------
    polys : list of polynomial objects
        Polynomials to find the common roots of.
    method : string
        The root finding method to be used. Can be either 'Mult' or 'Div'.
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
            return oneD.solve(polys[0], method)
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
            return multiplication(polys)
        else:
            if poly_type == 'MultiPower':
                return division_power(polys)
            else:
                return division_cheb(polys)
