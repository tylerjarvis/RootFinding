import numpy as np
import itertools
from numalgsolve import OneDimension as oneD
from numalgsolve.polynomial import MultiCheb, MultiPower, is_power
from numalgsolve.TelenVanBarel import TelenVanBarel
from numalgsolve.PowerDivision import division_power
from numalgsolve.ChebyshevDivision import division_cheb
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

def multiplication(polys):
    '''
    Finds the roots of the given list of multidimensional polynomials using a multiplication matrix. 

    Parameters
    ----------
    polys : list of polynomial objects
        Polynomials to find the common roots of.
    returns
    -------
    roots : numpy array
        The common roots of the polynomials. Each row is a root.
    '''    
    poly_type = is_power(polys, return_string = True)
    dim = polys[0].dim

    m_f, var_dict = TVBMultMatrix(polys, poly_type)

    # both TVBMultMatrix and groebnerMultMatrix will return m_f as
    # -1 if the ideal is not zero dimensional or if there are no roots
    if type(m_f) == int:
        return -1

    # Get list of indexes of single variables and store vars that were not
    # in the vector space basis.
    var_spots = list()
    spot = np.zeros(dim)
    for i in range(dim):
        spot[i] = 1
        var_spots.append(var_dict[tuple(spot)])
        spot[i] = 0

    # Get left eigenvectors

    vals,vecs = np.linalg.eig(m_f.T)
    
    zeros_spot = var_dict[tuple(0 for i in range(dim))]
        
    vecs = vecs[:,np.abs(vecs[zeros_spot]) > 1.e-10]

    roots = vecs[var_spots]/vecs[zeros_spot]
    return roots.T

def TVBMultMatrix(polys, poly_type):
    '''
    Finds the multiplication matrix using the reduced Macaulay matrix from the
    TVB method.

    Parameters
    ----------
    polys : array-like
        The polynomials to find the common zeros of
    poly_type : string
        The type of the polynomials in polys

    Returns
    -------
    multiplicationMatrix : 2D numpy array
        The multiplication matrix for a random polynomial f
    var_dict : dictionary
        Maps each variable to its position in the vector space basis
    '''
    basisDict, VB, degree = TelenVanBarel(polys)
        
    dim = max(f.dim for f in polys)

    # Get random polynomial f
    f = _random_poly(poly_type, dim)[0]
    
    #Dictionary of terms in the vector basis their spots in the matrix.
    VBdict = {}
    spot = 0
    for row in VB:
        VBdict[tuple(row)] = spot
        spot+=1

    # Build multiplication matrix m_f
    mMatrix = np.zeros((len(VB), len(VB)))
    for i in range(VB.shape[0]):
        f_coeff = f.mon_mult(VB[i], returnType = 'Matrix')
        for term in zip(*np.where(f_coeff != 0)):
            if term in VBdict:
                mMatrix[VBdict[term]][i] += f_coeff[term]
            else:
                mMatrix[:,i] -= f_coeff[term]*basisDict[term]

    # Construct var_dict
    var_dict = {}
    for i in range(len(VB)):
        mon = VB[i]
        if np.sum(mon) == 1 or np.sum(mon) == 0:
            var_dict[tuple(mon)] = i

    return mMatrix, var_dict

def _random_poly(_type, dim):
    '''
    Generates a random polynomial that has the form
    c_1x_1 + c_2x_2 + ... + c_nx_n where n = dim and each c_i is a randomly
    chosen integer between 0 and 1000.

    Parameters
    ----------
    _type : string
        Type of Polynomial to generate. "MultiCheb" or "MultiPower".
    dim : int
        Degree of polynomial to generate (?).

    Returns
    -------
    Polynomial
        Randomly generated Polynomial.
    '''
    _vars = get_var_list(dim)

    random_poly_shape = [2 for i in range(dim)]

    random_poly_coeff = np.zeros(tuple(random_poly_shape), dtype=int)
    for var in _vars:
        random_poly_coeff[var] = np.random.randint(1000)

    if _type == 'MultiCheb':
        return MultiCheb(random_poly_coeff), _vars
    else:
        return MultiPower(random_poly_coeff), _vars