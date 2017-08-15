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

def F4(polys, reducedGroebner = True):
    '''
    The main function. Initializes the matrix, adds the phi's and r's, and then reduces it. Repeats until the reduction
    no longer adds any more polynomials to the matrix. Print statements let us see the progress of the code.
    '''
    polys_were_added = True
    while polys_were_added:
        self.initialize_np_matrix()
        self.add_phi_to_matrix()
        self.add_r_to_matrix()
        self.create_matrix()
        polys_were_added = self.reduce_matrix(qr_reduction = qr_reduction, triangular_solve = False)

    self.get_groebner()
    if reducedGroebner:
        self.reduce_groebner_basis()
    return self.groebner_basis

def initialize_np_matrix(self, final_time = False):
    '''
    Initialzes self.np_matrix to having just old_polys and new_polys in it
    matrix_terms is the header of the matrix, it lines up each column with a monomial

    Now it sorts through the polynomials and if a polynomial is going to be reduced this time through
    it adds it and it's reducer to the matrix but doesn't use it for phi or r calculations.
    This makes the code WAY faster.
    '''
    matrix_polys = list()
    original_lms

    self.matrix_terms = []
    self.np_matrix = np.array([])
    self.term_set = set()
    self.lead_term_set = set()
    self.original_lms = set()
    self.matrix_polys = list()

    if final_time:
        self._add_polys(self.new_polys + self.old_polys)
        for poly in self.new_polys + self.old_polys:
            self.original_lms.add(Term(poly.lead_term))
    else:
        old_polys = self.old_polys
        new_polys = self.new_polys
        polys = old_polys + new_polys

        polys = utils.sorted_polys_monomial(polys)

        self.old_polys = list()
        self.new_polys = list()

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
                # Why is this next line happening??
                self.new_polys.append(lms[i][0])
                # Removing ^ still passes all tests so....

                polys_with_unique_lm.append(lms[i][0])
                lms[i].remove(lms[i][0])
                self._add_polys(lms[i]) #Still lets stuff be reduced if a poly reduces all of them.
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
                self._add_poly_to_matrix(i)
                self._add_poly_to_matrix(j.mon_mult(tuple(a-b for a,b in zip(i.lead_term,j.lead_term))))

        # Now add everything that couldn't be divided out to the matrix,
        # and put them back in either self.old_polys or self.new_polys,
        # whichever they belonged to before.
        #
        # This means that the ones that got divided out are essentially
        # removed from the list they belonged to, so they won't be
        # used for phi or r calculations.
        for i in polys_with_unique_lm:
            if i not in divides_out:
                self._add_poly_to_matrix(i)
                if i in old_polys:
                    self.old_polys.append(i)
                elif i in new_polys:
                    self.new_polys.append(i)
                else:
                    raise ValueError("Where did this poly come from?")


def build_maxheap(term_set, lead_term_set):
    '''Builds a maxheap of Term objects for use in r polynomial calculation.

    Parameters
    ----------
    term_set : set, contains Term objects
        The set of all terms that appear as column labels in the main matrix
    lead_term_set : set, contains Term objects
        The set of all terms that appear as leading terms of some polynomial
        in the main matrix

    Returns
    -------
    monheap : MaxHeap object, contains Term objects
        A max heap of all terms that do not appear as leading terms of any
        polynomial

    '''

    monheap = utils.MaxHeap()

    for mon in term_set:
        if mon not in lead_term_set: #Adds every monomial that isn't a lead term to the heap
            monheap.heappush(mon)

    return monheap

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

def clean_matrix(matrix, matrix_terms):
    '''
    Gets rid of columns in the matrix that are all zero and returns it and the updated matrix_terms.
    '''
    non_zero_monomial = np.sum(abs(matrix), axis=0) != 0
    matrix = matrix[:,non_zero_monomial] #Only keeps the non_zero_monomials
    matrix_terms = matrix_terms[non_zero_monomial] #Only keeps the non_zero_monomials
    return matrix, matrix_terms

def create_matrix(polys):
    '''
    Takes a list of polynomial objects (polys) and uses them to create a matrix. That is ordered by the monomial
    ordering. Returns the matrix and the matrix_terms, a list of the monomials corresponding to the rows of the matrix.
    '''
    #Gets an empty polynomial whose lm all other polynomial divide into.
    bigShape = np.maximum.reduce([p.coeff.shape for p in polys])
    #Gets a list of all the flattened polynomials.
    flat_polys = list()
    for poly in polys:
        #Gets a matrix that is padded so it is the same size as biggest, and flattens it. This is so
        #all flattened polynomials look the same.
        newMatrix = fill_size(bigShape, poly.coeff)
        flat_polys.append(newMatrix.ravel())

    #Make the matrix
    matrix = np.vstack(flat_polys[::-1])

    #Makes matrix_terms, a list of all the terms in the matrix.
    startTerms = time.time()
    terms = np.zeros(bigShape, dtype = Term)
    for i,j in np.ndenumerate(terms):
        terms[i] = Term(i)
    matrix_terms = terms.ravel()
    endTerms = time.time()
    #print(endTerms - startTerms)

    #Gets rid of any columns that are all 0.
    matrix, matrix_terms = clean_matrix(matrix, matrix_terms)

    #Sorts the matrix and matrix_terms by term order.
    matrix, matrix_terms = sort_matrix(matrix, matrix_terms)

    #Sorts the rows of the matrix so it is close to upper triangular.
    matrix = utils.row_swap_matrix(matrix)
    return matrix, matrix_terms

def divides(a,b): #This is not the divides used in utils.
    '''
    Takes two terms, a and b. Returns True if b divides a. False otherwise.
    '''
    diff = tuple(i-j for i,j in zip(a.val,b.val))
    return all(i >= 0 for i in diff)

def fill_size(bigShape,smallPolyCoeff):
    '''
    Pads the smallPolyCoeff so it has the same shape as bigShape. Does this by making a matrix with the shape of
    bigShape and then dropping smallPolyCoeff into the top of it with slicing.
    Returns the padded smallPolyCoeff.
    '''
    if (smallPolyCoeff.shape == bigShape).all():
        return smallPolyCoeff
    matrix = np.zeros(bigShape)

    slices = list()
    for i in smallPolyCoeff.shape:
        s = slice(0,i)
        slices.append(s)
    matrix[slices] = smallPolyCoeff
    return matrix

def find_degree(poly_list):
    """
    Takes a list of polynomials and finds the degree needed for a Macaulay matrix.
    Adds the degree of each polynomial and then subtracts the total number of polynomials and adds one.

    Example:
        For polynomials [P1,P2,P3] with degree [d1,d2,d3] the function returns d1+d2+d3-3+1

    """
    degree_needed = 0
    for poly in poly_list:
        degree_needed += poly.degree
    return ((degree_needed - len(poly_list)) + 1)

def get_good_rows(matrix, matrix_terms):
    '''
    Gets the rows in a matrix whose leading monomial is not divisible by the leading monomial of any other row.
    Returns a list of rows.
    This function could probably be improved, but for now it is good enough.
    '''
    rowLMs = dict()
    already_looked_at = set()
    #Finds the leading terms of each row.
    for i, j in zip(*np.where(matrix!=0)):
        if i in already_looked_at:
            continue
        else:
            already_looked_at.add(i)
            rowLMs[i] = matrix_terms[j]
    keys= list(rowLMs.keys())
    keys = keys[::-1]
    spot = 0
    #Uses a sieve to find which of the rows to keep.
    while spot != len(keys):
        term1 = rowLMs[keys[spot]]
        toRemove = list()
        for i in range(spot+1, len(keys)):
            term2 = rowLMs[keys[i]]
            if divides(term2,term1):
                toRemove.append(keys[i])
        for i in toRemove:
            keys.remove(i)
        spot += 1
    return keys

def get_polys_from_matrix(matrix, matrix_terms, rows, power=False, clean=False, accuracy=1.e-10):
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
    clean : bool, optional
        If true, any row whose absolute sum is less than accuracy will not be
        converted to a polynomial object.
    accuracy : float, optional
        Any row whose absolute sum is less than this value will not be
        converted to polynomial objects if clean is True.

    Returns
    -------
    poly_list : list
        Polynomial objects corresponding to the specified rows.

    '''
    shape = []
    p_list = []
    matrix_term_vals = [i.val for i in matrix_terms]

    # Finds the maximum size needed for each of the poly coeff tensors
    for i in range(len(matrix_term_vals[0])):
        # add 1 to each to compensate for constant term
        shape.append(max(matrix_term_vals, key=itemgetter(i))[i]+1)
    # Grabs each polynomial, makes coeff matrix and constructs object
    for i in rows:
        p = matrix[i]
        if clean:
            if np.sum(np.abs(p)) < accuracy:
                continue
        coeff = np.zeros(shape)
        for j,term in enumerate(matrix_term_vals):
            coeff[term] = p[j]

        if power:
            poly = MultiPower(coeff)
        else:
            poly = MultiCheb(coeff)

        if poly.lead_term != None:
            p_list.append(poly)
    return p_list

def rrqr_reduce(matrix, clean = False, global_accuracy = 1.e-10):
    '''
    Recursively reduces the matrix using rrqr reduction so it returns a reduced matrix, where each row has
    a unique leading monomial.
    '''
    if matrix.shape[0]==0 or matrix.shape[1]==0:
        return matrix
    height = matrix.shape[0]
    A = matrix[:height,:height] #Get the square submatrix
    B = matrix[:,height:] #The rest of the matrix to the right
    Q,R,P = qr(A, pivoting = True) #rrqr reduce it
    PT = utils.inverse_P(P)
    diagonals = np.diagonal(R) #Go along the diagonals to find the rank
    rank = np.sum(np.abs(diagonals)>global_accuracy)
    if rank == height: #full rank, do qr on it
        Q,R = qr(A)
        A = R #qr reduce A
        B = Q.T.dot(B) #Transform B the same way
    else: #not full rank
        A = R[:,PT] #Switch the columns back
        if clean:
            Q[np.where(abs(Q) < global_accuracy)]=0
        B = Q.T.dot(B) #Multiply B by Q transpose
        if clean:
            B[np.where(abs(B) < global_accuracy)]=0
        #sub1 is the top part of the matrix, we will recursively reduce this
        #sub2 is the bottom part of A, we will set this all to 0
        #sub3 is the bottom part of B, we will recursively reduce this.
        #All submatrices are then put back in the matrix and it is returned.
        sub1 = np.hstack((A[:rank,],B[:rank,])) #Takes the top parts of A and B
        result = rrqr_reduce(sub1) #Reduces it
        A[:rank,] = result[:,:height] #Puts the A part back in A
        B[:rank,] = result[:,height:] #And the B part back in B

        sub2 = A[rank:,]
        zeros = np.zeros_like(sub2)
        A[rank:,] = np.zeros_like(sub2)

        sub3 = B[rank:,]
        B[rank:,] = rrqr_reduce(sub3)

    reduced_matrix = np.hstack((A,B))
    return reduced_matrix

def sort_matrix(matrix, matrix_terms):
    '''
    Takes a matrix and matrix_terms (holding the terms in each column of the matrix), and sorts them both
    by term order.
    Returns the sorted matrix and matrix_terms.
    '''
    #argsort_list gives the ordering by which the matrix should be sorted.
    argsort_list = utils.argsort_dec(matrix_terms)[0]
    matrix_terms.sort()
    matrix = matrix[:,argsort_list]
    return matrix, matrix_terms[::-1]
