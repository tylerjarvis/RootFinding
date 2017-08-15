# imports from groebner
from groebner.polynomial import Polynomial, MultiCheb, MultiPower
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
from groebner.utils import InstabilityWarning

def F4(polys, reducedGroebner = True):
    '''
    The main function. Initializes the matrix, adds the phi's and r's, and then reduces it. Repeats until the reduction
    no longer adds any more polynomials to the matrix. Print statements let us see the progress of the code.
    '''
    power = isinstance(polys[0],MultiPower)
    old_polys = list()
    new_polys = polys
    polys_were_added = True
    while len(new_polys) > 0:
        old_polys, new_polys, matrix_polys = initialize_np_matrix(old_polys, new_polys)
        matrix_polys = add_phi_to_matrix(old_polys, new_polys, matrix_polys)
        matrix_polys, matrix_terms = add_r_to_matrix(matrix_polys, old_polys + new_polys)
        matrix, matrix_terms = create_matrix(matrix_polys, matrix_terms)
        old_polys += new_polys
        new_polys = get_new_polys(matrix, matrix_terms, power)

    groebner_basis = old_polys
    if reducedGroebner:
        groebner_basis = reduce_groebner_basis(groebner_basis, power)
    return groebner_basis

def initialize_np_matrix(old_polys, new_polys, final_time = False):
    '''
    Initialzes self.np_matrix to having just old_polys and new_polys in it
    matrix_terms is the header of the matrix, it lines up each column with a monomial

    Now it sorts through the polynomials and if a polynomial is going to be reduced this time through
    it adds it and it's reducer to the matrix but doesn't use it for phi or r calculations.
    This makes the code WAY faster.
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
    '''
    Takes all new possible combinations of phi polynomials and adds them to the Groebner Matrix
    Includes some checks to throw out unnecessary phi's
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
    '''
    Parameters:
    i (int) : the index of the first polynomial
    j (int) : the index of the second polynomial
    B (set) : index of the set of polynomials to be considered.

    Returns:
       (bool) : returns False if
            1) The polynomials at index i and j are relative primes or
            2) there exists an l such that (i,l) or (j,l) will not be considered in
            the add_phi_to_matrix() method and LT(l) divides lcm(LT(i),LT(j)),
            otherwise, returns True.
       * See proposition 8 in "Section 10: Improvements on Buchburger's algorithm."
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
    Calculates the phi-polynomial's of the polynomials a and b.

    Parameters
    ----------
    a, b : Polynomial objects

    Returns
    -------
    2 Polynomial objects
        The calculated phi polynomials for a and b.

    Notes
    -----
    Phi polynomials are defined to be
    .. math::
        \frac{lcm(LT(a), LT(b))}_{LT(a)} * a\\
        \frac{lcm(LT(a), LT(b))}_{LT(b)} * b

    The reasoning behind this definition is that both phis will have the
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
    Polynomial object or None
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
    So the highest terms come first, the lowest ones last/.
    Parameters
    ----------
    matrix_terms : numpy array.
        Each row is one of the terms in the matrix.
    Returns
    -------
    matrix_terms : numpy array
        The sorted matrix_terms.
    '''
    termList = list()
    for term in matrix_terms:
        termList.append(Term(term))
    argsort_list, termList = utils.argsort_dec(termList)
    return matrix_terms[argsort_list]

def coeff_slice(coeff):
    ''' Gets the n-d slices that corespond to the dimenison of a coeff matrix.
    Parameters
    ----------
    coeff : numpy matrix.
        The matrix of interest.
    Returns
    -------
    slices : list
        Each value of the list is a slice of the matrix in some dimension. It is exactly the size of the matrix.
    '''
    slices = list()
    for i in coeff.shape:
        slices.append(slice(0,i))
    return slices

def create_matrix(matrix_polys, matrix_terms = None):
    ''' Builds a Macaulay matrix.

    Parameters
    ----------
    matrix_polys : list.
        Contains numpy arrays that hold the polynomials to be put in the matrix.
    matrix_terms : numpy array
        The terms that will exist in the matrix. Not sorted yet.
    Returns
    -------
    matrix : 2D numpy array
        The Macaulay matrix.
    '''
    bigShape = np.maximum.reduce([p.coeff.shape for p in matrix_polys])
    if matrix_terms is None:
        #Finds the matrix terms.
        non_zeroSet = set()
        for polys in matrix_polys:
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
        slices = coeff_slice(coeff)
        added_zeros[slices] = coeff
        flat_polys.append(added_zeros[matrix_term_indexes])
        added_zeros[slices] = np.zeros_like(coeff)

    #Make the matrix
    matrix = np.vstack(flat_polys[::-1])

    #Sorts the rows of the matrix so it is close to upper triangular.
    matrix = utils.row_swap_matrix(matrix)
    plt.matshow([i == 0 for i in matrix])
    return matrix, matrix_terms


def get_polys_from_matrix(matrix, matrix_terms, rows, power):
    '''Creates polynomial objects from the specified rows of the given matrix.

    Parameters
    ----------
    matrix : 2D numpy array
        The matrix with rows corresponding to polynomials, columns corresponding
        to monomials, and entries corresponding to coefficients.
    matrix_terms : array-like, contains Term objects
        The column labels for matrix in order.
    rows : iterable, contains integers
        The rows for which to create polynomial objects.
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
    matrix : 2D numpy array
        The matrix of interest.

    Returns
    -------
    reduced_matrix : 2D numpy array
        The matrix in row echelon form with all zero rows removed.

    '''

    independent_rows, dependent_rows, Q = utils.fullRank(matrix, accuracy=accuracy)
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
    matrix : 2D numpy array
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
    matrix : 2D numpy array
        The matrix where rows correspond to polynomials, columns to terms,
        and entries to coefficients.
    matrix_terms : 2D numpy array
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
    '''
    matrix, matrix_terms = create_matrix(groebner_basis)
    matrix = utils.triangular_solve(matrix)
    rows = np.arange(matrix.shape[0])
    return get_polys_from_matrix(matrix, matrix_terms, rows, power)
