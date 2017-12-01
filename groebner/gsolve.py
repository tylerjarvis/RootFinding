# imports from groebner
from groebner.polynomial import Polynomial, MultiCheb, MultiPower, is_power
from groebner.utils import Term
import groebner.utils as utils

# other imports
from operator import itemgetter
import itertools
import numpy as np
import math
from scipy.linalg import lu, qr, solve_triangular
from scipy.sparse import csc_matrix, vstack
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
from groebner.utils import InstabilityWarning, slice_top

def F4(polys, reducedGroebner = True, accuracy = 1.e-10, phi = True):
    '''Uses the F4 algorithm to find a Groebner Basis.

    Parameters
    ----------
    polys : list
        The polynomails for which a Groebner basis is computed.
    reducedGroebner : bool
        Defaults to True. If True then a reduced Groebner Basis is found. If False, just a Groebner Basis is found.
    accuracy : float
        Defaults to 1.e-10. What is determined to be zero in the code.

    Returns
    -------
    groebner_basis : list
        The polynomials in the Groebner Basis.
    '''

    power = is_power(polys)
    old_polys = list()
    new_polys = polys
    polys_were_added = True
    while len(new_polys) > 0:
        old_polys, new_polys, matrix_polys = sort_reducible_polys(old_polys, new_polys)
        matrix_polys = add_phi_to_matrix(old_polys, new_polys, matrix_polys, phi = phi)
        matrix_polys, matrix_terms = add_r_to_matrix(matrix_polys, old_polys + new_polys)
        matrix, matrix_terms = create_matrix(matrix_polys, matrix_terms)
        old_polys += new_polys
        new_polys = get_new_polys(matrix, matrix_terms, accuracy = accuracy, power = power)
    groebner_basis = old_polys
    if reducedGroebner:
        groebner_basis = reduce_groebner_basis(groebner_basis, power)
    return groebner_basis

def sort_reducible_polys(old_polys, new_polys):
    '''Finds which polynomials are reducible.
    The polynomials that are reducible aren't used in phi and r calculations, they are just added
    to the matrix. They are also removed from the poly list they are in, as whatever they are reduced
    down to will be pulled out of the matrix at the end.

    A polynomial is considered reducible if the leading term of another polynomial divides it's leading term.
    In the case of multiple polynomials having the same leading term, one is considered non-reducible and the
    rest are reducible.

    Parameters
    ----------
    old_polys : list
        The polynomails that have already gone through the reduction before.
    new_polys : list
        The polynomials that have not gone through the reduction before.

    Returns
    -------
    old_polys : list
        The old_polys that are not reducible.
    new_polys : list
        The new_polys that are not reducible.
    matrix_polys : list
        The polynomials that are being put in the matrix. Any polynomial that is reducible is put in the matrix,
        and if it is reducible because some other polynomials lead term divides it's lead term than the other
        polynomial is multiplied by th monomial needed to give it the same lead term, and that is put in the matrix.
    '''
    matrix_polys = list()

    old = old_polys
    new  = new_polys
    polys = old + new

    polys = utils.sorted_polys_monomial(polys)

    old_polys = list()
    new_polys = list()

    lms = defaultdict(list)
    for p in polys:
        lms[p.lead_term].append(p)

    # This list will contain one polynomial for each leading term,
    # so all the leading terms will be unique
    polys_with_unique_lm = list()

    for i in lms:
        # If there are multiple polynomials with the same leading term
        # we add just one of them to polys_with_unique_lm and add the
        # rest to the matrix for reduction.
        if len(lms[i]) > 1:
            polys_with_unique_lm.append(lms[i][0])
            lms[i].remove(lms[i][0])
            for p in lms[i]:
                matrix_polys.append(p)
        else:
            polys_with_unique_lm.append(lms[i][0])

    divides_out = list()

    # Checks if anything in old_polys or new_polys can divide each other
    # Example: if f1 divides f2, then f2 and  LT(f2)/LT(f1) * f1 are
    # added to the matrix
    for i,j in itertools.permutations(polys_with_unique_lm,2):
        if i in divides_out:
            continue
        if utils.divides(j.lead_term,i.lead_term): # j divides into i
            divides_out.append(i)
            matrix_polys.append(i)
            matrix_polys.append(j.mon_mult(tuple(a-b for a,b in zip(i.lead_term,j.lead_term))))

    # Now add everything that couldn't be divided out to the matrix,
    # and put them back in either self.old_polys or self.new_polys,
    # whichever they belonged to before.
    #
    # This means that the ones that got divided out are essentially
    # removed from the list they belonged to, so they won't be
    # used for phi or r calculations.
    for i in polys_with_unique_lm:
        if i not in divides_out:
            matrix_polys.append(i)
            if i in old:
                old_polys.append(i)
            elif i in new:
                new_polys.append(i)
            else:
                raise ValueError("Where did this poly come from?")
    return old_polys, new_polys, matrix_polys

def add_phi_to_matrix(old_polys, new_polys, matrix_polys, phi = True):
    '''Adds phi polynomials to the matrix.

    Given two polynomials we define two phi polynomials as follows. If the two polynomials A and B have leading
    terms A.lt and B.lt, then call the least common multiple of these is lcm. Then the two phis are A*lcm/A.lt and
    B*lcm/B.lt.

    Parameters
    ----------
    old_polys : list
        The polynomials that have already gone through the reduction before.
    new_polys : list
        The polynomials that have not gone through the reduction before.

    Returns
    -------
    matrix_polys : list
        The polynomials that are being put in the matrix. Both the ones that were put in earlier and the new
        phi polynomials that are bing added.
    '''
    # Find the set of all pairs of index the function will run through

    # Index_new iterate the tuple of every combination of the new_polys.
    index_new = itertools.combinations(range(len(new_polys)),2)
    # Index_oldnew iterates the tuple of every combination of new and old polynomials
    index_oldnew = itertools.product(range(len(new_polys)),range(len(new_polys),
                                                                      len(old_polys)+len(new_polys)))
    all_index_combinations = set(itertools.chain(index_new,index_oldnew))

    # Iterating through both possible combinations.
    all_polys = new_polys + old_polys
    while all_index_combinations:
        i,j = all_index_combinations.pop()
        if phi_criterion(all_polys, i, j, all_index_combinations, phi):
            #calculate the phi's.
            phi_a , phi_b = calc_phi(all_polys[i],all_polys[j])
            # add the phi's on to the Groebner Matrix.
            matrix_polys.append(phi_a)
            matrix_polys.append(phi_b)
    return matrix_polys

def phi_criterion(all_polys,i,j,B,phi):
    '''Evaluates the phi criterion, given by:
        False if:
            1) The polynomials at index i and j are relative primes or
            2) there exists an l such that (i,l) or (j,l) will not be considered in
            the add_phi_to_matrix() method and LT(l) divides lcm(LT(i),LT(j)).
        Otherwise, true.

        See proposition 8 in "Section 10: Improvements on Buchburger's algorithm.

    Parameters
    ----------
    all_polys : list
        List of all the polynomials.
    i : int
        Index of the first polynomial
    j : int
        Index of the second polynomial
    B : set
        Index of the set of polynomials to be considered.

    Returns
    -------
    bool
        Truth value of the phi criterion
    '''
    if phi == False:
        return True

    # Relative Prime check: If the lead terms of i and j are relative primes, phi is not needed
    if all([a*b == 0 for a,b in zip(all_polys[i].lead_term,all_polys[j].lead_term)]):
        return False

    # Another criterion
    else:
        for l in range(len(all_polys)):
            # Checks that l is not j or i.
            if l == j or l == i:
                #print("\t{} is i or j".format(l))
                continue

            # Sorts the tuple (i,l) or (l,i) in order of smaller to bigger.
            i_tuple = tuple(sorted((i,l)))
            j_tuple = tuple(sorted((j,l)))

            # i_tuple and j_tuple needs to not be in B.
            if j_tuple in B or i_tuple in B:
                continue

            lcm = utils.lcm(all_polys[i],all_polys[j])
            lead_l = all_polys[l].lead_term

            # See if LT(poly[l]) divides lcm(LT(i),LT(j))
            if all([i-j>=0 for i,j in zip(lcm,lead_l)]) :
                return False

    # Function will return True and calculate phi if none of the checks passed for all l's.
    return True

def calc_phi(a,b):
    '''
    Calculates the phi-polynomials of the polynomials a and b.

    Parameters
    ----------
    a, b : Polynomial
        Input polynomials.

    Returns
    -------
    Polynomial
        The calculated phi polynomial for a.
    Polynomial
        The calculated phi polynomial for b.

    Notes
    -----
    Phi polynomials are defined to be
    .. math::
        \frac{lcm(LT(a), LT(b))}_{LT(a)} * a\\
        \frac{lcm(LT(a), LT(b))}_{LT(b)} * b

    The reasoning behind this definition is that both phi polynomials will have the
    same leading term so they can be linearly reduced to produce a new,
    smaller polynomial in the ideal.

    '''

    lcm = utils.lcm(a,b)
    a_quo = utils.quotient(lcm, a.lead_term)
    b_quo = utils.quotient(lcm, b.lead_term)
    return a.mon_mult(a_quo), b.mon_mult(b_quo)

def add_r_to_matrix(matrix_polys, all_polys):
    '''
    Finds the r polynomials and adds them to the matrix.
    First makes Heap out of all potential monomials, then finds polynomials
    with leading terms that divide it and add them to the matrix.

    Parameters
    ----------
    matrix_polys : list
    all_polys : list

    Returns
    -------
    matrix_polys : list
    matrix_terms : ndarray
    '''
    matrixTermSet = set()
    leadTermSet = set()

    for poly in matrix_polys:
        for mon in zip(*np.where(poly.coeff != 0)):
            matrixTermSet.add(tuple(mon))
        leadTermSet.add(poly.lead_term)

    others = list()
    for term in matrixTermSet:
        if term not in leadTermSet:
            others.append(term)

    sorted_polys = utils.sorted_polys_coeff(all_polys)

    for term in others:
        r = calc_r(term, sorted_polys)
        if r is not None:
            for mon in zip(*np.where(r.coeff != 0)):
                if mon not in matrixTermSet and mon is not r.lead_term:
                    others.append(mon)
                    matrixTermSet.add(mon)
            matrix_polys.append(r)

    matrix_terms = np.array(matrixTermSet.pop())
    for term in matrixTermSet:
        matrix_terms = np.vstack((matrix_terms,term))

    return matrix_polys, matrix_terms

def calc_r(m, polys):
    '''Calculates an r polynomial that has a leading monomial m.

    Parameters
    ----------
    m : array-like
        The leading monomial that the r polynomial should have.
    polys : array-like
        Contains polynomial objects from which to create the r polynomial.

    Returns
    -------
    Polynomial or None
        If no polynomial divides m, returns None. Otherwise, returns
        the r polynomial with leading monomial m.

    Notes
    -----
    The r polynomial corresponding to m is defined as follows:

        Find a polynomial p such that the leading monomial of p divides m.
        Then the r polynomial is

        .. math::
            r = \frac{m}_{LT(p)} * p

    The reason we use r polynomials is because now any polynomial with
    m as a term will be linearly reduced by r.

    '''
    for poly in polys:
        LT_p = list(poly.lead_term)
        if len(LT_p) == len(m) and utils.divides(LT_p, m):
            quotient = utils.quotient(m, LT_p)
            if not LT_p == m: #Make sure c isn't all 0
                return poly.mon_mult(quotient)
    return None

def sort_matrix_terms(matrix_terms):
    '''Sorts the matrix_terms by term order.
    So the highest terms come first, the lowest ones last.

    Parameters
    ----------
    matrix_terms : ndarray
        Array where each row is one of the terms in the matrix.

    Returns
    -------
    matrix_terms : ndarray
        The sorted matrix_terms.
    '''
    termList = list()
    for term in matrix_terms:
        termList.append(Term(term))
    argsort_list = np.argsort(termList)[::-1]
    return matrix_terms[argsort_list]

def create_matrix(matrix_polys, matrix_terms = None):
    ''' Builds a Macaulay matrix.

    If there is only one term in the matrix it won't work.

    Parameters
    ----------
    matrix_polys : list.
        Contains numpy arrays that hold the polynomials to be put in the matrix.
    matrix_terms : ndarray
        The terms that will exist in the matrix. Not sorted yet.
        Defaults to None, in which case it will be found in the function.
    Returns
    -------
    matrix : ndarray
        The Macaulay matrix.
    '''
    bigShape = np.maximum.reduce([p.coeff.shape for p in matrix_polys])
    if matrix_terms is None:
        #Finds the matrix terms.
        non_zeroSet = set()
        for poly in matrix_polys:
            for term in zip(*np.where(poly.coeff != 0)):
                non_zeroSet.add(term)
        matrix_terms = np.array(non_zeroSet.pop())
        for term in non_zeroSet:
            matrix_terms = np.vstack((matrix_terms,term))

    matrix_terms = sort_matrix_terms(matrix_terms)

    #Get the slices needed to pull the matrix_terms from the coeff matrix.
    matrix_term_indexes = list()
    for i in range(len(bigShape)):
        matrix_term_indexes.append(matrix_terms.T[i])

    #Adds the poly_coeffs to flat_polys, using added_zeros to make sure every term is in there.
    added_zeros = np.zeros(bigShape)
    flat_polys = list()
    for poly in matrix_polys:
        coeff = poly.coeff
        slices = slice_top(coeff)
        added_zeros[slices] = coeff
        flat_polys.append(added_zeros[matrix_term_indexes])
        added_zeros[slices] = np.zeros_like(coeff)

    #Make the matrix
    matrix = np.vstack(flat_polys[::-1])

    #Sorts the rows of the matrix so it is close to upper triangular.
    matrix = utils.row_swap_matrix(matrix)
    return matrix, matrix_terms


def get_polys_from_matrix(matrix, matrix_terms, rows, power):
    '''Creates polynomial objects from the specified rows of the given matrix.

    Parameters
    ----------
    matrix : (M,N) ndarray
        The matrix with rows corresponding to polynomials, columns corresponding
        to monomials, and entries corresponding to coefficients.
    matrix_terms : array-like
        The column labels for matrix in order. Contains Term objects.
    rows : iterable
        The rows for which to create polynomial objects. Contains integers.
    power : bool
        If true, the polynomials returned will be MultiPower objects.
        Otherwise, they will be MultiCheb.
    Returns
    -------
    poly_list : list
        Polynomial objects corresponding to the specified rows.
    '''

    shape = []
    p_list = []
    shape = np.maximum.reduce([term for term in matrix_terms])
    shape += np.ones_like(shape)
    spots = list()
    for dim in range(matrix_terms.shape[1]):
        spots.append(matrix_terms.T[dim])

    # Grabs each polynomial, makes coeff matrix and constructs object
    for i in rows:
        p = matrix[i]
        coeff = np.zeros(shape)
        coeff[spots] = p
        if power:
            poly = MultiPower(coeff)
        else:
            poly = MultiCheb(coeff)

        if poly.lead_term != None:
            p_list.append(poly)
    return p_list

def row_echelon(matrix, accuracy=1.e-10):
    '''Reduces the matrix to row echelon form and removes all zero rows.

    Parameters
    ----------
    matrix : (M,N) ndarray
        The matrix of interest.

    Returns
    -------
    reduced_matrix : (M,N) ndarray
        The matrix in row echelon form with all zero rows removed.

    '''
    independent_rows, dependent_rows, Q = utils.row_linear_dependencies(matrix, accuracy=accuracy)
    full_rank_matrix = matrix[independent_rows]

    reduced_matrix = utils.rrqr_reduce2(full_rank_matrix)
    reduced_matrix = utils.clean_zeros_from_matrix(reduced_matrix)

    non_zero_rows = np.sum(abs(reduced_matrix),axis=1) != 0
    if np.sum(non_zero_rows) != reduced_matrix.shape[0]:
        warnings.warn("Full rank matrix has zero rows.", InstabilityWarning)

    reduced_matrix = reduced_matrix[non_zero_rows,:] #Only keeps the non-zero polymonials

    return reduced_matrix

def lead_term_columns(matrix):
    '''Finds all columns that correspond to the leading term of some polynomial
    in the matrix.

    Parameters
    ----------
    matrix : (M,N) ndarray
        The matrix of interest.

    Returns
    -------
    LT_columns : set
        The set of column indexes that correspond to leading terms
    '''
    LT_columns = set()

    already_looked_at = set()
    for i, j in zip(*np.where(matrix!=0)):
        if i not in already_looked_at:
            LT_columns.add(j)
            already_looked_at.add(i)

    return LT_columns

def get_new_polys(matrix, matrix_terms, accuracy=1.e-10, power=False):
    '''Reduces the given matrix and finds all polynomials that have new
    leading terms after reduction.

    Parameters
    ----------
    matrix : (M,N) ndarray
        The matrix where rows correspond to polynomials, columns to terms,
        and entries to coefficients.
    matrix_terms : (M,N) ndarray
        Each row corresponds to a column in matrix, each column corresponds
        to a variable, and entries correspond to the exponent of that variable
    accuracy : float
        Entries in matrix lower than accuracy will be counted as zero during
        the reduction process
    power : bool
        True if the polynomials are in the power basis, false for chebyshev.

    Returns
    -------
    new_polys : list
        Contains polynomial objects whose leading terms weren't leading terms
        in the matrix passed in, but the were after reduction.
    '''
    lead_term_before = lead_term_columns(matrix)
    reduced_matrix = row_echelon(matrix, accuracy=accuracy)
    lead_term_after = lead_term_columns(reduced_matrix)

    new_lead_terms = lead_term_after - lead_term_before

    #Get the new polynomials
    new_poly_spots = list()
    already_looked_at = set() #rows whose leading monomial we've already checked
    for i, j in zip(*np.where(reduced_matrix!=0)):
        if i not in already_looked_at:
            if j in new_lead_terms:
                new_poly_spots.append(i)
            already_looked_at.add(i)

    new_polys = get_polys_from_matrix(reduced_matrix, \
        matrix_terms, new_poly_spots, power=power)

    return new_polys

def reduce_groebner_basis(groebner_basis, power):
    '''
    Uses triangular solve to get a fully reduced Groebner basis.

    Parameters
    ----------
    groebner_basis : list
        List of polynomials forming a Groebner basis.
    power : bool
        If true, the polynomials returned will be MultiPower objects.
        Otherwise, they will be MultiCheb.

    Returns
    -------
    list
        List of Polynomials forming a fully reduced Groebner basis.
    '''
    if len(groebner_basis) == 1:
        poly = groebner_basis[0]
        poly.coeff = poly.coeff/poly.lead_coeff
        groebner_basis[0] = poly
        return groebner_basis

    #This should be done eventually, Just make sure to pull out only the original GB.
    #matrix_polys, matrix_terms = add_r_to_matrix(groebner_basis, groebner_basis)
    #matrix, matrix_terms = create_matrix(matrix_polys, matrix_terms)

    matrix, matrix_terms = create_matrix(groebner_basis)
    matrix = utils.triangular_solve(matrix)
    rows = np.arange(matrix.shape[0])
    return get_polys_from_matrix(matrix, matrix_terms, rows, power)
