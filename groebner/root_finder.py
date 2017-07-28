import numpy as np
from groebner.multi_power import MultiPower
from groebner.multi_cheb import MultiCheb
from groebner.polynomial import Polynomial
import itertools
from groebner.groebner_class import Groebner
from groebner.Macaulay import Macaulay
from groebner.maxheap import Term
import time

times = {}
times["reducePoly"] = 0
times["buildMatrix"] = 0
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

    times["reducePoly"] = 0
    times["buildMatrix"] = 0
    Polynomial.clearTime()

    startTime = time.time()

    # Determine polynomial type
    poly_type = ''
    if (all(type(p) == MultiCheb for p in polys)):
        poly_type = 'MultiCheb'
    elif (all(type(p) == MultiPower for p in polys)):
        poly_type = 'MultiPower'
    else:
        raise ValueError('All polynomials must be the same type')
    # Calculate groebner basis
    startBasis = time.time()
    if method == 'Groebner':
        G = Groebner(polys)
        GB = G.solve()
    else:
        GB = Macaulay(polys)
    endBasis = time.time()
    times["basis"] = (endBasis - startBasis)

    dim = max(g.dim for g in GB) # dimension of the polynomials

    # Check for no solutions
    if len(GB) == 1 and all([i==1 for i in GB[0].coeff.shape]):
        print("No solutions")
        return -1

    startRandPoly = time.time()
    # Make sure ideal is zero-dimensional and get random polynomial
    f, var_list = _random_poly(poly_type, dim)
    if not _test_zero_dimensional(var_list, GB):
        print("Ideal is not zero-dimensional; cannot calculate roots.")
        return -1
    endRandPoly = time.time()
    times["randPoly"] = (endRandPoly - startRandPoly)

    # Get multiplication matrix
    startVectorBasis = time.time()
    VB, var_dict = vectorSpaceBasis(GB)
    endVectorBasis = time.time()
    times["vectorBasis"] = (endVectorBasis - startVectorBasis)

    startMultMatrix = time.time()
    m_f = multMatrix(f, GB, VB)
    endMultMatrix = time.time()
    times["multMatrix"] = (endMultMatrix - startMultMatrix)

    startEndStuff = time.time()
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
    e = np.linalg.eig(m_f.T)
    eig = e[1]
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
    endEndStuff = time.time()
    times["endStuff"] = (endEndStuff - startEndStuff)
    endTime = time.time()
    totalTime = (endTime - startTime)

    print("Total run time for roots is {}".format(totalTime))
    print(times)
    MultiCheb.printTime()
    MultiPower.printTime()
    Polynomial.printTime()
    print((times["basis"]+times["multMatrix"])/totalTime)
    return roots

def sorted_polys_coeff(polys):
    '''
    Sorts the polynomials by how much bigger the leading coefficient is than the rest of the coeff matrix.
    '''
    lead_coeffs = list()
    for poly in polys:
        lead_coeffs.append(abs(poly.lead_coeff)/np.sum(np.abs(poly.coeff))) #The lead_coeff to other stuff ratio.
    argsort_list = sorted(range(len(lead_coeffs)), key=lead_coeffs.__getitem__)[::-1]
    sorted_polys = list()
    for i in argsort_list:
        sorted_polys.append(polys[i])
    return sorted_polys

def clean_matrix(matrix, matrix_terms, basisSet):
    '''
    Gets rid of rows in the matrix that are all zero and returns it and the updated matrix_terms.
    '''
    non_zero_row = [(i in basisSet) for i in matrix_terms]
    matrix = matrix[non_zero_row] #Only keeps the non_zero_monomials
    matrix_terms = matrix_terms[non_zero_row] #Only keeps the non_zero_monomials
    return matrix, matrix_terms

def sort_matrix(matrix, matrix_terms, basisList):
    '''
    Takes a matrix and matrix_terms (holding the terms in each row of the matrix), and sorts the matrix so
    the terms are in the same order as in basisList.
    Returns the sorted matrix.
    '''
    matrix_terms = list(matrix_terms)
    order = np.zeros(len(basisList), dtype = int)
    for i in range(len(basisList)):
        order[i] = matrix_terms.index(basisList[i])
    return matrix[order]

def multMatrix(poly, GB, basisList):
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

    returns
    -------
    multOperatorMatrix : square numpy array
        The matrix m_f
    '''
    basisSet = set(basisList)
    
    GB = sorted_polys_coeff(GB)

    # All polys in GB will be in the same dimension, so just match poly with
    # the first Groebner basis element
    poly = _match_poly_dim(poly, GB[0])[0]

    dim = len(basisList) # Dimension of the vector space basis
    matrix_coeffs = list()
    
    for i in range(dim):
        monomial = basisList[i]
        poly_ = poly.mon_mult(monomial)
        matrix_coeffs.append(coordinateVector(poly_, GB, basisList, basisSet).flatten())
    multMatrix = np.vstack(matrix_coeffs)
    multMatrix = multMatrix.T
    
    remainder_shape = np.maximum.reduce([p.shape for p in GB])
    terms = np.zeros(remainder_shape, dtype = tuple)
    for i,j in np.ndenumerate(terms):
        terms[i] = tuple(i)
    matrix_terms = terms.ravel()
    
    multMatrix, matrix_terms = clean_matrix(multMatrix, matrix_terms, basisSet)
    multMatrix = sort_matrix(multMatrix, matrix_terms, basisList)
    return multMatrix

def vectorSpaceBasis(GB):
    '''
    parameters
    ----------
    GB: list
        polynomial objects that make up a Groebner basis for the ideal

    returns
    -------
    basis : list
        tuples representing the monomials in the vector space basis
    var_to_pos_dict : dictionary
        maps each variable to its position in the vector space basis
    '''
    LT_G = [f.lead_term for f in GB]
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

def coordinateVector(poly, GB, basisList, basisSet):
    '''
    parameters
    ----------
    reducedPoly : polynomial object
        The polynomial for which to find the coordinate vector of its coset.
    GB : list of polynomial objects
        Polynomials that make up a Groebner basis for the ideal
    basis : list of tuples
        The monomials that make up a basis for the vector space
    leadTermDict: A dictionary of the leadTerms in GB to the polynomials in GB.

    returns
    -------
    coordinateVector : list
        The coordinate vector of the given polynomial's coset in
        A = C[x_1,...x_n]/I as a vector space over C
    '''
    dim = len(basisList) # Dimension of vector space basis

    poly_coeff = reduce_poly(poly, GB, basisSet)
    return poly_coeff

def divides(mon1, mon2):
    '''
    parameters
    ----------
    mon1 : tuple
        contains the exponents of the monomial divisor
    mon2 : tuple
        contains the exponents of the monomial dividend

    returns
    -------
    boolean
        true if mon1 divides mon2, false otherwise
    '''
    return all(np.subtract(mon2, mon1) >= 0)


def reduce_poly(poly, divisors, basisSet, permitted_round_error=1e-10):
    '''
    Divides a polynomial by a set of divisor polynomials using the standard
    multivariate division algorithm and returns the remainder

    parameters
    ----------
    poly : polynomial object
        the polynomial to be divided by the Groebner basis
    divisors : list of polynomial objects
        polynomials to divide poly by
    leadTermDict: A dictionary of the leadTerms in GB to the polynomials in GB.

    returns
    -------
    polynomial object
        the remainder of poly / divisors
    '''
    startTime = time.time()

    remainder_shape = np.maximum.reduce([p.shape for p in divisors])
    remainder = np.zeros(remainder_shape)
    
    for term in zip(*np.where(poly.coeff != 0)):
        if term in basisSet:
            remainder[term] += poly.coeff[term]
            poly.coeff[term] = 0
    poly.__init__(poly.coeff, clean_zeros = False)

    # while poly is not the zero polynomial
    while np.any(poly.coeff):
        divisible = False
        # Go through polynomials in set of divisors
        for divisor in divisors:
            poly, divisor = _match_poly_dim(poly, divisor)
            # If the LT of the divisor divides the LT of poly
            if divides(divisor.lead_term, poly.lead_term):
                # Get the quotient LT(poly)/LT(divisor)
                LT_quotient = np.subtract(poly.lead_term, divisor.lead_term)

                poly_to_subtract_coeff = divisor.mon_mult(LT_quotient, returnType = 'Matrix')
                # Match sizes of poly_to_subtract and poly so
                # poly_to_subtract.coeff can be subtracted from poly.coeff
                poly_coeff, poly_to_subtract_coeff = poly.match_size(poly.coeff, poly_to_subtract_coeff)
                new_coeff = poly_coeff - \
                    (poly.lead_coeff/poly_to_subtract_coeff[tuple(divisor.lead_term+LT_quotient)])*poly_to_subtract_coeff

                new_coeff[np.where(abs(new_coeff) < permitted_round_error)]=0
                
                for term in zip(*np.where(new_coeff != 0)):
                    if term in basisSet:
                        remainder[term] += new_coeff[term]
                        new_coeff[term] = 0

                poly.__init__(new_coeff, clean_zeros = False)
                divisible = True
                break

    endTime = time.time()
    times["reducePoly"] += (endTime - startTime)
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
