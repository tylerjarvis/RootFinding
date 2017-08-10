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

    def _lcm(self,a,b):
        '''
        Finds the LCM of the two leading terms of Polynomial a,b

        Params:
        a, b - polynomail objects

        returns:
        LCM - the np.array of the lead_term of the lcm polynomial
        '''
        return np.maximum(a.lead_term, b.lead_term)

    def calc_phi(self,a,b):
        '''
        Calculates the phi-polynomial's of the polynomials a and b.

        Phi polynomials are defined to be:
        (lcm(LT(a), LT(b)) / LT(a)) * a and
        (lcm(LT(a), LT(b)) / LT(b)) * b

        The reasoning behind this definition is that both phis will have the
        same leading term so they can be linearly reduced to produce a new,
        smaller polynomial in the ideal.

        Returns:
            A tuple of the calculated phi's.
        '''
        lcm = self._lcm(a,b)

        a_diff = tuple([i-j for i,j in zip(lcm, a.lead_term)])
        b_diff = tuple([i-j for i,j in zip(lcm, b.lead_term)])
        return a.mon_mult(a_diff), b.mon_mult(b_diff)

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
                phi_a , phi_b = self.calc_phi(all_polys[i],all_polys[j])
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

                lcm = self._lcm(polys[i],polys[j])
                lead_l = polys[l].lead_term

                # See if LT(poly[l]) divides lcm(LT(i),LT(j))
                if all([i-j>=0 for i,j in zip(lcm,lead_l)]) :
                    return False

        # Function will return True and calculate phi if none of the checks passed for all l's.
            return True

    def _build_maxheap(self):
        '''
        Builds a maxheap for use in r polynomial calculation
        '''

        self.monheap = utils.MaxHeap()

        for mon in self.term_set:
            if mon not in self.lead_term_set: #Adds every monomial that isn't a lead term to the heap
                self.monheap.heappush(mon)

    def sorted_polys_coeff(self):
        '''
        Sorts the polynomials by how much bigger the leading coefficient is than the rest of the coeff matrix.
        '''
        polys = self.new_polys+self.old_polys
        lead_coeffs = list()
        for poly in polys:
            lead_coeffs.append(abs(poly.lead_coeff)/np.sum(np.abs(poly.coeff))) #The lead_coeff to other stuff ratio.
        argsort_list = sorted(range(len(lead_coeffs)), key=lead_coeffs.__getitem__)[::-1]
        sorted_polys = list()
        for i in argsort_list:
            sorted_polys.append(polys[i])
        return sorted_polys

    def calc_r(self, m, sorted_polys):
        '''
        Finds the r polynomial that has a leading monomial m.

        The r polynomials are defined as follows:
            First look at all monomials that are not leading terms. For each one
            of those monomials m, if there is a polynoial p that divides it,
            calculate r = (m / LT(p)) * p. This is the r polynomial
            corresponding to m.
            The reason we use r polynomials is because now any polynomial with
            m in it will be linearly reduced even further.

        Returns the polynomial.
        '''
        for p in sorted_polys:
            LT_p = list(p.lead_term)
            if all([i<=j for i,j in zip(LT_p,m)]) and len(LT_p) == len(m): #Checks to see if LT_p divides m
                c = [j-i for i,j in zip(LT_p,m)]
                if not LT_p == m: #Make sure c isn't all 0
                    return p.mon_mult(c)

    def add_r_to_matrix(self):
        '''
        Finds the r polynomials and adds them to the matrix.
        First makes Heap out of all potential monomials, then finds polynomials
        with leading terms that divide it and add them to the matrix.
        '''
        self._build_maxheap()
        sorted_polys = self.sorted_polys_coeff()
        while len(self.monheap) > 0:
            m = list(self.monheap.heappop().val)
            r = self.calc_r(m,sorted_polys)
            self._add_poly_to_matrix(r, adding_r = True)

    def row_swap_matrix(self, matrix):
        '''
        rearange the rows of matrix so it starts close to upper traingular
        '''
        rows, columns = np.where(matrix != 0)
        lms = {}
        last_i = -1
        lms = list()
        for i,j in zip(rows,columns):
            if i == last_i:
                continue
            else:
                lms.append(j)
                last_i = i
        argsort_list = sorted(range(len(lms)), key=lms.__getitem__)[::]
        return matrix[argsort_list]

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

        self.np_matrix = self.row_swap_matrix(self.np_matrix)

    def reduce_matrix(self, qr_reduction=True, triangular_solve = False):
        '''
        Reduces the matrix fully using either QR or LU decomposition. Adds the new_poly's to old_poly's, and adds to
        new_poly's any polynomials created by the reduction that have new leading monomials.
        Returns-True if new polynomials were found, False otherwise.
        '''
        if qr_reduction:
            independentRows, dependentRows, Q = utils.fullRank(self.np_matrix)
            fullRankMatrix = self.np_matrix[independentRows]

            reduced_matrix = self.rrqr_reduce2(fullRankMatrix)
            reduced_matrix = self.clean_zeros_from_matrix(reduced_matrix)

            non_zero_rows = np.sum(abs(reduced_matrix),axis=1) != 0

            reduced_matrix = reduced_matrix[non_zero_rows,:] #Only keeps the non_zero_polymonials

            if triangular_solve:
                reduced_matrix = self.triangular_solve(reduced_matrix)
                if clean:
                    reduced_matrix = self.clean_zeros_from_matrix(reduced_matrix)
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

    def clean_zeros_from_matrix(self,matrix):
        '''
        Sets all points in the matrix less than the gloabal accuracy to 0.
        '''
        matrix[np.where(np.abs(matrix) < global_accuracy)]=0
        return matrix

    def hasFullRank(self, matrix):
        height = matrix.shape[0]
        if height == 0:
            return True
        try:
            Q,R,P = qr(matrix, pivoting = True)
        except ValueError:
            print("VALUE ERROR")
            print(matrix)
        diagonals = np.diagonal(R) #Go along the diagonals to find the rank
        rank = np.sum(np.abs(diagonals)>global_accuracy)
        if rank == height:
            return True
        else:
            print(rank,height)
            return False

    def rrqr_reduce2(self, matrix): #My new sort of working one. Still appears to have some problems. Possibly from fullRank.
        if matrix.shape[0] <= 1 or matrix.shape[0]==1 or  matrix.shape[1]==0:
            return matrix
        height = matrix.shape[0]
        A = matrix[:height,:height] #Get the square submatrix
        B = matrix[:,height:] #The rest of the matrix to the right
        independentRows, dependentRows, Q = utils.fullRank(A)
        nullSpaceSize = len(dependentRows)
        if nullSpaceSize == 0: #A is full rank
            #print("FULL RANK")
            Q,R = qr(matrix)
            return R
        else: #A is not full rank
            #print("NOT FULL RANK")
            #sub1 is the independentRows of the matrix, we will recursively reduce this
            #sub2 is the dependentRows of A, we will set this all to 0
            #sub3 is the dependentRows of Q.T@B, we will recursively reduce this.
            #We then return sub1 stacked on top of sub2+sub3
            bottom = matrix[dependentRows]
            BCopy = B.copy()
            sub3 = bottom[:,height:]
            sub3 = Q.T[-nullSpaceSize:]@BCopy
            sub3 = self.rrqr_reduce(sub3)

            sub1 = matrix[independentRows]
            sub1 = self.rrqr_reduce(sub1)

            sub2 = bottom[:,:height]
            sub2[:] = np.zeros_like(sub2)

            reduced_matrix = np.vstack((sub1,np.hstack((sub2,sub3))))
            return reduced_matrix

    def rrqr_reduce(self, matrix): #Original One. Seems to be the more stable one from testing.
        if matrix.shape[0]==0 or matrix.shape[1]==0:
            return matrix
        if clean:
            matrix = self.clean_zeros_from_matrix(matrix)
        height = matrix.shape[0]
        A = matrix[:height,:height] #Get the square submatrix
        B = matrix[:,height:] #The rest of the matrix to the right
        Q,R,P = qr(A, pivoting = True) #rrqr reduce it
        PT = self.inverse_P(P)
        diagonals = np.diagonal(R) #Go along the diagonals to find the rank
        rank = np.sum(np.abs(diagonals)>global_accuracy)
        if clean:
            R = self.clean_zeros_from_matrix(R)

        if rank == height: #full rank, do qr on it
            Q,R = qr(A)
            A = R #qr reduce A
            B = Q.T.dot(B) #Transform B the same way
        else: #not full rank
            A = R[:,PT] #Switch the columns back
            B = Q.T.dot(B) #Multiply B by Q transpose
            #sub1 is the top part of the matrix, we will recursively reduce this
            #sub2 is the bottom part of A, we will set this all to 0
            #sub3 is the bottom part of B, we will recursively reduce this.
            #All submatrices are then put back in the matrix and it is returned.
            sub1 = np.hstack((A[:rank,],B[:rank,])) #Takes the top parts of A and B
            result = self.rrqr_reduce(sub1) #Reduces it
            A[:rank,] = result[:,:height] #Puts the A part back in A
            B[:rank,] = result[:,height:] #And the B part back in B

            sub2 = A[rank:,]
            zeros = np.zeros_like(sub2)
            A[rank:,] = np.zeros_like(sub2)

            sub3 = B[rank:,]
            B[rank:,] = self.rrqr_reduce(sub3)

        reduced_matrix = np.hstack((A,B))

        if not clean:
            return reduced_matrix
        else:
            return self.clean_zeros_from_matrix(reduced_matrix)

    def inverse_P(self,p):
        '''
        Takes in the one dimentional array of column switching.
        Returns the one dimentional array of switching it back.
        '''
        # The elementry matrix that flips the columns of given matrix.
        P = np.eye(len(p))[:,p]
        # This finds the index that equals 1 of each row of P.
        #(This is what we want since we want the index of 1 at each column of P.T)
        return np.where(P==1)[1]

    def triangular_solve(self,matrix):
        " Reduces the upper block triangular matrix. "
        m,n = matrix.shape
        j = 0  # The row index.
        k = 0  # The column index.
        c = [] # It will contain the columns that make an upper triangular matrix.
        d = [] # It will contain the rest of the columns.
        order_c = [] # List to keep track of original index of the columns in c.
        order_d = [] # List to keep track of the original index of the columns in d.

        # Checks if the given matrix is not a square matrix.
        if m != n:
            # Makes sure the indicies are within the matrix.
            while j < m and k < n:
                if matrix[j,k]!= 0:
                    c.append(matrix[:,k])
                    order_c.append(k)
                    # Move to the diagonal if the index is non-zero.
                    j+=1
                    k+=1
                else:
                    d.append(matrix[:,k])
                    order_d.append(k)
                    # Check the next column in the same row if index is zero.
                    k+=1
            # C will be the square matrix that is upper triangular with no zeros on the diagonals.
            C = np.vstack(c).T
            # If d is not empty, add the rest of the columns not checked into the matrix.
            if d:
                D = np.vstack(d).T
                D = np.hstack((D,matrix[:,k:]))
            else:
                D = matrix[:,k:]
            # Append the index of the rest of the columns to the order_d list.
            for i in range(n-k):
                order_d.append(k)
                k+=1

            # Solve for the CX = D
            X = solve_triangular(C,D)

            # Add I to X. [I|X]
            solver = np.hstack((np.eye(X.shape[0]),X))

            # Find the order to reverse the columns back.
            order = self.inverse_P(order_c+order_d)

            # Reverse the columns back.
            solver = solver[:,order]
            # Temporary checker. Plots the non-zero part of the matrix.
            #plt.matshow(~np.isclose(solver,0))

            return solver

        else:
        # The case where the matrix passed in is a square matrix
            return np.eye(m)

    def fully_reduce(self, matrix, qr_reduction = True):
        '''
        This function isn't really used any more as it seems less stable. But it's good for testing purposes.

        Fully reduces the matrix by making sure all submatrices formed by taking out columns of zeros are
        also in upper triangular form. Does this recursively. Returns the reduced matrix.
        '''
        matrix = self.clean_zeros_from_matrix(matrix)
        diagonals = np.diagonal(matrix).copy()
        zero_diagonals = np.where(abs(diagonals)==0)[0]
        if(len(zero_diagonals != 0)):
            first_zero = zero_diagonals[0]
            i = first_zero
            #Checks how many rows we can go down that are all 0.
            while all([k==0 for k in matrix[first_zero:,i:i+1]]):
                i+=1
                if(i == matrix.shape[1]):
                    i = -1
                    break

            if(i != -1):
                sub_matrix = matrix[first_zero: , i:]
                if qr_reduction:
                    Q,R = qr(sub_matrix)
                    sub_matrix = self.fully_reduce(R)
                else:
                    P,L,U = lu(sub_matrix)
                    #ERROR HERE BECAUSE OF THE PERMUATION MATRIX, I'M NOT SURE HOW TO FIX IT
                    sub_matrix = self.fully_reduce(U, qr_reduction = False)

                matrix[first_zero: , i:] = sub_matrix
        if clean:
            return self.clean_zeros_from_matrix(matrix)
        else:
            return self.clean_zeros_from_matrix(matrix)
