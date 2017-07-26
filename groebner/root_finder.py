import numpy as np
from groebner.multi_power import MultiPower
from groebner.multi_cheb import MultiCheb
import itertools
from groebner.groebner_class import Groebner
from groebner.Macaulay import Macaulay

'''
This module contains the tools necessary to find the points of the variety of the
ideal generated by a Groebner basis.
'''


def roots(polys, method = 'Groebner'):
    '''
    Finds the roots of the given list of polynomials

    parameters
    ----------
    polys : list of polynomial objects
        polynomials to find the common roots of

    returns
    -------
    list of numpy arrays
        the common roots of the polynomials
    '''
    # Determine polynomial type
    poly_type = ''
    if (all(type(p) == MultiCheb for p in polys)):
        poly_type = 'MultiCheb'
    elif (all(type(p) == MultiPower for p in polys)):
        poly_type = 'MultiPower'
    else:
        raise ValueError('All polynomials must be the same type')

    # Calculate groebner basis
    if method == 'Groebner':
        G = Groebner(polys)
        GB = G.solve()
    else:
        GB = Macaulay(polys)
    dim = max(g.dim for g in GB) # dimension of the polynomials

    # Check for no solutions
    if len(GB) == 1 and all([i==1 for i in GB[0].coeff.shape]):
        print("No solutions")
        return -1

    # Make sure ideal is zero-dimensional and get random polynomial
    f, var_list = _random_poly(poly_type, dim)
    if not _test_zero_dimensional(var_list, GB):
        print("Ideal is not zero-dimensional; cannot calculate roots.")
        return -1

    # Get multiplication matrix
    VB, var_dict = vectorSpaceBasis(GB)
    #print("VB:", VB)
    #print("var_dict:", var_dict)
    m_f = multMatrix(f, GB, VB)

    # Get list of indexes of single variables and store vars that were not
    # in the vector space basis.
    var_indexes = np.array([-1 for i in range(dim)])
    vars_not_in_basis = {}
    for i in range(len(var_list)):
        var = var_list[i] # x_i
        if var in var_dict:
            var_indexes[i] = var_dict[var]
        else:
            # maps the position in the root to its variable
            vars_not_in_basis[i] = var

    vnib = False
    if len(vars_not_in_basis) != 0:
        vnib = True

    # Get left eigenvectors
    eig = np.linalg.eig(m_f.T)[1]
    num_vectors = eig.shape[1]
    eig_vectors = [eig[:,i].tolist() for i in range(num_vectors)] # columns of eig

    roots = []
    for v in eig_vectors:
        root = np.zeros(dim, dtype=complex)
        # This will always work because var_indexes and root have the
        # same length - dim - and var_indexes has the variables in the
        # order they should be in the root
        for i in range(dim):
            x_i_pos = var_indexes[i]
            if x_i_pos != -1:
                root[i] = v[x_i_pos]/v[0]
        if vnib:
            # Go through the indexes of variables not in the basis in
            # decreasing order. It must be done in decreasing order for the
            # roots to be calculated correctly, since the vars with lower
            # indexes depend on the ones with higher indexes
            for pos in list(vars_not_in_basis.keys())[::-1]:
                GB_poly = _get_poly_with_LT(vars_not_in_basis[pos], GB)
                var_value = GB_poly.evaluate_at(root) * -1
                root[pos] = var_value
        roots.append(root)

    return roots

def multMatrix(poly, GB, basis):
    '''
    Finds the matrix of the linear operator m_f on A = C[x_1,...,x_n]/I
    where f is the polynomial argument. The linear operator m_f is defined
    as m_f([g]) = [f]*[g] where [f] represents the coset of f in
    A. Since m_f is a linear operator on A, it can be represented by its
    matrix with respect to the vector space basis.

    parameters
    ----------
    poly : polynomial object
        The polynomial f for which to find the matrix m_f.
    GB: list of polynomial objects
        Polynomials that make up a Groebner basis for the ideal
    basis : list of tuples
        The monomials that make up a basis for the vector space A

    return
    ------
    multOperatorMatrix : square numpy array
        The matrix m_f
    '''

    # All polys in GB will be in the same dimension, so just match poly with
    # the first Groebner basis element
    poly = _match_poly_dim(poly, GB[0])[0]

    dim = len(basis) # Dimension of the vector space basis
    multMatrix = np.zeros((dim, dim))

    for i in range(dim):
        monomial = basis[i]
        poly_ = poly.mon_mult(monomial)

        multMatrix[:,i] = coordinateVector(poly_, GB, basis)

    return multMatrix

def vectorSpaceBasis(GB):
    '''
    parameters
    ----------
    GB: list
        polynomial objects that make up a Groebner basis for the ideal

    return
    ------
    basis : list
        tuples representing the monomials in the vector space basis
    var_to_pos_dict : dictionary
        maps each variable to its position in the vector space basis
    '''
    LT_G = [f.lead_term for f in GB]
    print("LT_G:", LT_G)
    possibleVarDegrees = [range(max(tup)) for tup in zip(*LT_G)]
    possibleMonomials = itertools.product(*possibleVarDegrees)
    basis = []
    var_to_pos_dict = {}

    for mon in possibleMonomials:
        divisible = False
        for LT in LT_G:
            if divides(LT, mon):
                 divisible = True
                 break
        if not divisible:
            basis.append(mon)
            if (sum(mon) == 1):
                var_to_pos_dict[mon] = basis.index(mon)

    return basis, var_to_pos_dict

def coordinateVector(poly, GB, basis):
    '''
    parameters
    ----------
    reducedPoly : polynomial object
        The polynomial for which to find the coordinate vector of its coset.
    GB : list of polynomial objects
        Polynomials that make up a Groebner basis for the ideal
    basis : list of tuples
        The monomials that make up a basis for the vector space

    return
    ------
    coordinateVector : list
        The coordinate vector of the given polynomial's coset in
        A = C[x_1,...x_n]/I as a vector space over C
    '''
    dim = len(basis) # Dimension of vector space basis
    poly = reduce_poly(poly, GB)

    poly_terms = poly.monomialList()[::-1]
    assert(len(poly_terms) <= dim)

    coordinateVector = [0] * dim
    for monomial in poly_terms:
        coordinateVector[basis.index(monomial)] = \
            poly.coeff[monomial]

    return coordinateVector

def divides(mon1, mon2):
    '''
    parameters
    ----------
    mon1 : tuple
        contains the exponents of the monomial divisor
    mon2 : tuple
        contains the exponents of the monomial dividend

    return
    ------
    boolean
        true if mon1 divides mon2, false otherwise
    '''
    return all(np.subtract(mon2, mon1) >= 0)

def reduce_poly(poly, divisors, permitted_round_error=1e-10):
    '''
    Divides a polynomial by a set of divisor polynomials using the standard
    multivariate division algorithm and returns the remainder

    parameters
    ----------
    polynomial : polynomial object
        the polynomial to be divided by the Groebner basis
    divisors : list of polynomial objects
        polynomials to divide poly by

    return
    ------
    polynomial object
        the remainder of poly / divisors
    '''
    # init remainder polynomial
    if type(poly) == MultiCheb:
        remainder = MultiCheb(np.zeros((1,1)))
    else:
        remainder = MultiPower(np.zeros((1,1)))

    # while poly is not the zero polynomial
    while np.any(poly.coeff):
        divisible = False
        # Go through polynomials in set of divisors
        for divisor in divisors:
            poly, divisor = _match_poly_dim(poly, divisor)
            # If the LT of the divisor divides the LT of poly
            if divides(divisor.lead_term, poly.lead_term):
                # Get the quotient LT(poly)/LT(divisor)
                LT_quotient = tuple(np.subtract(
                    poly.lead_term, divisor.lead_term))

                poly_to_subtract = divisor.mon_mult(LT_quotient)

                # Match sizes of poly_to_subtract and poly so
                # poly_to_subtract.coeff can be subtracted from poly.coeff
                poly, poly_to_subtract = poly.match_size(poly, poly_to_subtract)

                new_coeff = poly.coeff - \
                    (poly.lead_coeff/divisor.lead_coeff)*poly_to_subtract.coeff
                new_coeff[np.where(abs(new_coeff) < permitted_round_error)]=0
                poly.__init__(new_coeff)

                divisible = True
                break

        if not divisible:
            remainder, poly = remainder.match_size(remainder, poly)

            polyLT = poly.lead_term

            # Add lead term to remainder
            remainder_coeff = remainder.coeff
            remainder_coeff[polyLT] = poly.coeff[polyLT]
            remainder.__init__(remainder_coeff)

            # Subtract LT from poly
            new_coeff = poly.coeff
            new_coeff[polyLT] = 0
            poly.__init__(new_coeff)

    return remainder

def _get_var_list(dim):
    _vars = [] # list of the variables: [x_1, x_2, ..., x_n]
    for i in range(dim):
        var = np.zeros(dim, dtype=int)
        var[i] = 1
        _vars.append(tuple(var))
    return _vars

def _random_poly(_type, dim):
    '''
    Generates a random polynomial that has the form
    c_1x_1 + c_2x_2 + ... + c_nx_n where n = dim and each c_i is a randomly
    chosen integer between 0 and 1000.
    '''
    _vars = _get_var_list(dim)

    random_poly_shape = [2 for i in range(dim)]

    random_poly_coeff = np.zeros(tuple(random_poly_shape), dtype=int)
    for var in _vars:
        random_poly_coeff[var] = np.random.randint(1000)

    if _type == 'MultiCheb':
        return MultiCheb(random_poly_coeff), _vars
    else:
        return MultiPower(random_poly_coeff), _vars

def _get_poly_with_LT(LT, GB):
    for poly in GB:
        if poly.lead_term == LT:
            return poly

def _test_zero_dimensional(_vars, GB):
    LT_list = [p.lead_term for p in GB]

    for var in _vars:
        exists_multiple = False
        for LT in LT_list:
            if np.linalg.matrix_rank(np.array([list(var), list(LT)])) == 1:
                exists_multiple = True
                break
        if not exists_multiple:
            return False

    return True

def _match_poly_dim(poly1, poly2):
    # Do nothing if they are already the same dimension
    if poly1.dim == poly2.dim:
        return poly1, poly2

    poly_type = ''
    if type(poly1) == MultiPower and type(poly2) == MultiPower:
        poly_type = 'MultiPower'
    elif type(poly1) == MultiCheb and type(poly2) == MultiCheb:
        poly_type = 'MultiCheb'
    else:
        raise ValueError('Polynomials must be the same type')

    poly1_vars = poly1.dim
    poly2_vars = poly2.dim
    max_vars = max(poly1_vars, poly2_vars)

    if poly1_vars < max_vars:
         for j in range(max_vars-poly1_vars):
             coeff_reshaped = poly1.coeff[...,np.newaxis]
         if poly_type == 'MultiPower':
             poly1 = MultiPower(coeff_reshaped)
         elif poly_type == 'MultiCheb':
             poly1 = MultiCheb(coeff_reshaped)
    elif poly2_vars < max_vars:
        for j in range(max_vars-poly2_vars):
            coeff_reshaped = poly2.coeff[...,np.newaxis]
        if poly_type == 'MultiPower':
            poly2 = MultiPower(coeff_reshaped)
        elif poly_type == 'MultiCheb':
            poly2 = MultiCheb(coeff_reshaped)

    return poly1, poly2
