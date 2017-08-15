import itertools
import numpy as np
import groebner.utils as utils
import groebner.gsolve as gsolve
import math
from groebner.polynomial import MultiCheb, MultiPower, Polynomial
from scipy.linalg import lu, qr, solve_triangular
from groebner.utils import Term
import matplotlib.pyplot as plt
from collections import defaultdict
from operator import itemgetter

#What we determine to be zero throughout the code
global_accuracy = 1.e-10
#If clean is true then at a couple of places (end of rrqr_reduce and end of add r to matrix) things close to 0 will be made 0.
#Might make it more stable, might make it less stable. Not sure.
clean = False

class Groebner(object):

    def __init__(self,polys):
        '''
        polys -- a list of polynomials that generate your ideal
        self.old_polys - The polynomials that have already gone through the solve loop once. Starts as none.
        self.new_polys - New polynomials that have never been through the solve loop. All of them at first.
        self.np_matrix - The full matrix of polynomials.
        self.term_set - The set of monomials in the matrix. Contains Terms.
        self.lead_term_set - The set of monomials that are lead terms of some polynomial in the matrix. Contains Terms.
        These next three are used to determine what polynomials to keep after reduction.
        self.original_lms - The leading Terms of the original polynomials (not phi's or r's). Keep these as old_polys.
        self.original_lm_dict - A dictionary of the leading terms to their polynomials
        self.not_needed_lms - The leading terms that have another leading term that divided them. We won't keep these.
        '''
        # Check polynomial types
        if all([type(p) == MultiPower for p in polys]):
            self.power = True
        elif all([type(p) == MultiCheb for p in polys]):
            self.power = False
        else:
            print([type(p) == MultiPower for p in polys])
            raise ValueError('Bad polynomials in list')

        self.old_polys = list()
        self.new_polys = polys
        self.np_matrix = np.array([])
        self.term_set = set()
        self.lead_term_set = set()
        self.original_lms = set()
        self.matrix_polys = list()

    def initialize_np_matrix(self, final_time = False):
        '''
        Initialzes self.np_matrix to having just old_polys and new_polys in it
        matrix_terms is the header of the matrix, it lines up each column with a monomial

        Now it sorts through the polynomials and if a polynomial is going to be reduced this time through
        it adds it and it's reducer to the matrix but doesn't use it for phi or r calculations.
        This makes the code WAY faster.
        '''
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

    def solve(self, qr_reduction = True, reducedGroebner = True):
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

    def reduce_groebner_basis(self):
        '''
        Goes through one more time using triangular solve to get a fully reduced Groebner basis.
        '''
        self.new_polys = list()
        self.old_polys = list()
        for poly in self.groebner_basis:
            self.old_polys.append(poly)
        self.initialize_np_matrix(final_time = True)
        self.add_r_to_matrix()
        self.create_matrix()
        self.reduce_matrix(triangular_solve = True)
        for poly in self.old_polys:
            poly.coeff[abs(poly.coeff) < global_accuracy] = 0
        self.groebner_basis = self.old_polys

    def get_groebner(self):
        '''
        Checks to see if our basis includes 1. If so, that is the basis. Also, removes 0 polynomials. Then makes an
        attribute self.groebner_basis that holds the groebner basis, not neccesarily reduced.
        '''
        self.groebner_basis = list()
        for poly in self.old_polys:
            if np.all([i == 1 for i in poly.coeff.shape]): #This if statement can be removed when done testing. Or can it?
                self.groebner_basis = list()
                self.groebner_basis.append(poly)
                return
            poly.coeff[np.where(abs(poly.coeff) < global_accuracy)] = 0
            self.groebner_basis.append(poly)

    def _add_poly_to_matrix(self, p, adding_r = False):
        '''
        Saves the polynomial to the set of polynomials that will be used in create_matrix.
        Also updates the list of leading terms and monomials in the matrix with the new values.

        adding_r is only true when the r's are being added, this way it knows to keep adding new monomials to the heap
        for further r calculation
        '''
        if p is None:
            return
        if p.lead_term is None:
            return
        self.matrix_polys.append(p)
        self.lead_term_set.add(Term(p.lead_term))

        for idx in zip(*np.where(p.coeff != 0)):
            idx_term = Term(tuple(idx)) #Get a term object
            if idx_term not in self.term_set:
                self.term_set.add(idx_term)
                #If r's being added, adds new monomial to the heap
                if adding_r:
                    if(idx_term not in self.lead_term_set):
                        self.monheap.heappush(idx_term)
        return

    def _add_polys(self, p_list):
        '''
        p_list - a list of polynomials
        Adds the polynomials to self.np_matrix
        '''
        for p in p_list:
            self._add_poly_to_matrix(p)

    def add_phi_to_matrix(self,phi = True):
        '''
        Takes all new possible combinations of phi polynomials and adds them to the Groebner Matrix
        Includes some checks to throw out unnecessary phi's
        '''
        # Find the set of all pairs of index the function will run through

        # Index_new iterate the tuple of every combination of the new_polys.
        index_new = itertools.combinations(range(len(self.new_polys)),2)
        # Index_oldnew iterates the tuple of every combination of new and old polynomials
        index_oldnew = itertools.product(range(len(self.new_polys)),range(len(self.new_polys),
                                               len(self.old_polys)+len(self.new_polys)))
        all_index_combinations = set(itertools.chain(index_new,index_oldnew))

        # Iterating through both possible combinations.
        all_polys = self.new_polys + self.old_polys
        while all_index_combinations:
            i,j = all_index_combinations.pop()
            if self.phi_criterion(i,j,all_index_combinations,phi):
                #calculate the phi's.
                phi_a , phi_b = gsolve.calc_phi(all_polys[i],all_polys[j])
                # add the phi's on to the Groebner Matrix.
                self._add_poly_to_matrix(phi_a)
                self._add_poly_to_matrix(phi_b)

    def phi_criterion(self,i,j,B,phi):
        # Need to run tests
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

        # List of new and old polynomials.
        polys = self.new_polys+self.old_polys

        # Relative Prime check: If the lead terms of i and j are relative primes, phi is not needed
        if all([a*b == 0 for a,b in zip(polys[i].lead_term,polys[j].lead_term)]):
            return False

        # Another criterion
        else:
            for l in range(len(polys)):
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

                lcm = utils.lcm(polys[i],polys[j])
                lead_l = polys[l].lead_term

                # See if LT(poly[l]) divides lcm(LT(i),LT(j))
                if all([i-j>=0 for i,j in zip(lcm,lead_l)]) :
                    return False

        # Function will return True and calculate phi if none of the checks passed for all l's.
            return True

    def add_r_to_matrix(self):
        '''
        Finds the r polynomials and adds them to the matrix.
        First makes Heap out of all potential monomials, then finds polynomials
        with leading terms that divide it and add them to the matrix.
        '''
        self.monheap = gsolve.build_maxheap(self.term_set, self.lead_term_set)
        sorted_polys = utils.sorted_polys_coeff(self.new_polys+self.old_polys)
        while len(self.monheap) > 0:
            m = list(self.monheap.heappop().val)
            r = gsolve.calc_r(m, sorted_polys)
            self._add_poly_to_matrix(r, adding_r = True)

    def fill_size(self,bigMatrix,smallMatrix):
        '''
        Fits the small matrix inside of the big matrix and returns it.
        Returns just the coeff matrix as that is all we need in the Groebner
        create_matrix code.
        '''
        if smallMatrix.shape == bigMatrix.shape:
            return smallMatrix
        matrix = np.zeros_like(bigMatrix) #Even though bigMatrix is all zeros, use this because it makes a copy

        slices = list()
        for i in smallMatrix.shape:
            s = slice(0,i)
            slices.append(s)
        matrix[slices] = smallMatrix
        return matrix

    def create_matrix(self):
        '''
        Creates the matrix from the polynomials in self.matrix_polys, which
        at this point contains all of self.new_polys, self.old_polys, the
        phi's and the r's. Each column of the matrix corresponds to a specific
        monomial, and each row corresponds to a polynomial.
        '''
        biggest_shape = np.maximum.reduce([p.coeff.shape for p in self.matrix_polys])

        if self.power:
            biggest = MultiPower(np.zeros(biggest_shape), clean_zeros = False)
        else:
            biggest = MultiCheb(np.zeros(biggest_shape), clean_zeros = False)
        self.np_matrix = biggest.coeff.flatten()
        self.np_matrix = np.array(self.np_matrix, dtype = np.longdouble)

        flat_polys = list()
        for poly in self.matrix_polys:
            newMatrix = self.fill_size(biggest.coeff, poly.coeff)
            flat_polys.append(newMatrix.ravel())

        self.np_matrix = np.vstack(flat_polys[::-1])

        # Note that 'terms' is an array of Term objects. We then flatten this
        # to become 'self.matrix_terms'. The position of each monomial in
        # 'self.matrix_terms' is the same as the column corresponding to that
        # monomial.
        terms = np.zeros(biggest_shape, dtype = Term)
        for i,j in np.ndenumerate(terms):
            terms[i] = Term(i)

        self.matrix_terms = terms.flatten()
        self.np_matrix, self.matrix_terms = utils.clean_matrix(self.np_matrix, self.matrix_terms)
        self.np_matrix, self.matrix_terms = utils.sort_matrix(self.np_matrix, self.matrix_terms)

        self.np_matrix = utils.row_swap_matrix(self.np_matrix)

    def reduce_matrix(self, qr_reduction=True, triangular_solve = False):
        '''
        Reduces the matrix fully using either QR or LU decomposition. Adds the new_poly's to old_poly's, and adds to
        new_poly's any polynomials created by the reduction that have new leading monomials.
        Returns-True if new polynomials were found, False otherwise.
        '''
        if qr_reduction:
            independentRows, dependentRows, Q = utils.fullRank(self.np_matrix)
            fullRankMatrix = self.np_matrix[independentRows]

            reduced_matrix = utils.rrqr_reduce2(fullRankMatrix)
            reduced_matrix = utils.clean_zeros_from_matrix(reduced_matrix)

            non_zero_rows = np.sum(abs(reduced_matrix),axis=1) != 0

            reduced_matrix = reduced_matrix[non_zero_rows,:] #Only keeps the non_zero_polymonials

            if triangular_solve:
                reduced_matrix = utils.triangular_solve(reduced_matrix)
                if clean:
                    reduced_matrix = utils.clean_zeros_from_matrix(reduced_matrix)
        else:
            #This thing is very behind the times.
            P,L,U = lu(self.np_matrix)
            reduced_matrix = U
            #reduced_matrix = self.fully_reduce(reduced_matrix, qr_reduction = False)
        #Get the new polynomials
        new_poly_spots = list()
        old_poly_spots = list()
        already_looked_at = set() #rows whose leading monomial we've already checked
        for i, j in zip(*np.where(reduced_matrix!=0)):
            if i in already_looked_at: #We've already looked at this row
                continue
            elif self.matrix_terms[j] in self.lead_term_set: #The leading monomial is not new.
                if self.matrix_terms[j] in self.original_lms: #Reduced old poly. Only used if triangular solve is done.
                    old_poly_spots.append(i)
                already_looked_at.add(i)
                continue
            else:
                already_looked_at.add(i)
                new_poly_spots.append(i) #This row gives a new leading monomial

        if triangular_solve:
            self.old_polys = gsolve.get_polys_from_matrix(reduced_matrix, \
                self.matrix_terms, old_poly_spots, power=self.power)
        else:
            self.old_polys = self.new_polys + self.old_polys

        self.new_polys = gsolve.get_polys_from_matrix(reduced_matrix, \
            self.matrix_terms, new_poly_spots, power=self.power)

        if len(self.old_polys+self.new_polys) == 0:
            print("ERROR ERROR ERROR ERROR ERROR NOT GOOD NO POLYNOMIALS IN THE BASIS FIX THIS ASAP!!!!!!!!!!!!!")
            print(reduced_matrix)

        return len(self.new_polys) > 0
