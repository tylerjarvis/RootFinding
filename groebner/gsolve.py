# Dependencies
from operator import itemgetter
import itertools
import numpy as np
import math
from scipy.linalg import lu, qr, solve_triangular
from scipy.sparse import csc_matrix, vstack
import matplotlib.pyplot as plt
import time
from collections import defaultdict

# From the groebner module
from groebner.polynomial import Polynomial, MultiCheb, MultiPower
from groebner.utils import Term, MaxHeap

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
        self.duplicate_lms - The leading terms that occur multiple times. Keep these as old_polys
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
        startTime = time.time()
        
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

            polys = self.sorted_polys_monomial(polys)

            self.old_polys = list()
            self.new_polys = list()

            lms = defaultdict(list)
            for p in polys:
                lms[p.lead_term].append(p)

            polys_with_unique_lm = list()

            for i in lms:
                if len(lms[i]) > 1: #It's duplicated
                    #self.duplicate_lms.add(Term(i))
                    self.new_polys.append(lms[i][0])
                    polys_with_unique_lm.append(lms[i][0])
                    lms[i].remove(lms[i][0])
                    self._add_polys(lms[i]) #Still lets stuff be reduced if a poly reduces all of them.
                else:
                    polys_with_unique_lm.append(lms[i][0])

            divides_out = list()

            for i,j in itertools.permutations(polys_with_unique_lm,2):
                if i in divides_out:
                    continue
                if self.divides(i,j): # j divides into i
                    startStuff = time.time()
                    divides_out.append(i)
                    self._add_poly_to_matrix(i)
                    self._add_poly_to_matrix(j.mon_mult(tuple(a-b for a,b in zip(i.lead_term,j.lead_term))))
                    
            
            for i in polys_with_unique_lm:
                if i not in divides_out:
                    self._add_poly_to_matrix(i)
                    if i in old_polys:
                        self.old_polys.append(i)
                    elif i in new_polys:
                        self.new_polys.append(i)
                    else:
                        raise ValueError("Where did this poly come from?")
        
        endTime = time.time()
        times["initialize"] += (endTime - startTime)

    def solve(self, qr_reduction = True, reducedGroebner = True):
        '''
        The main function. Initializes the matrix, adds the phi's and r's, and then reduces it. Repeats until the reduction
        no longer adds any more polynomials to the matrix. Print statements let us see the progress of the code.
        '''
        MultiCheb.clearTime()
        MultiPower.clearTime()
        startTime = time.time()
        
        polys_were_added = True
        i=1 #Tracks what loop we are on.
        while polys_were_added:
            #print("Starting Loop #"+str(i))
            #print("Num Polys - ", len(self.new_polys + self.old_polys))
            self.initialize_np_matrix()
            self.add_phi_to_matrix()
            self.add_r_to_matrix()
            self.create_matrix()
            #print(self.np_matrix.shape)
            polys_were_added = self.reduce_matrix(qr_reduction = qr_reduction, triangular_solve = False) #Get rid of triangular solve when done testing
            i+=1

        #print("Basis found!")    
        
        self.get_groebner()
        if reducedGroebner:
            self.reduce_groebner_basis()
        
        endTime = time.time()
        #print("WE WIN")
        print("Run time was {} seconds".format(endTime - startTime))
        print(times)
        MultiCheb.printTime()
        MultiPower.printTime()
        #print("Basis - ")
        #for poly in self.groebner_basis:
        #    print(poly.coeff)
            #break #print just one
        
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
        #print(self.np_matrix.shape)
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

    def sort_matrix(self):
        '''
        Sorts the matrix into degrevlex order.
        '''
        start = time.time()
        argsort_list, self.matrix_terms = self.argsort(self.matrix_terms)        
        self.np_matrix = self.np_matrix[:,argsort_list]
        end = time.time()
        times["sort"] += (end-start)

    def clean_matrix(self):
        '''
        Gets rid of rows and columns in the np_matrix that are all zero.
        '''
        start = time.time()
        ##This would replace all small values in the matrix with 0.
        if clean:
            self.np_matrix[np.where(abs(self.np_matrix) < global_accuracy)]=0

        #Removes all 0 monomials
        non_zero_monomial = np.sum(abs(self.np_matrix), axis=0) != 0
        self.np_matrix = self.np_matrix[:,non_zero_monomial] #only keeps the non_zero_monomials
        self.matrix_terms = self.matrix_terms[non_zero_monomial]

        end = time.time()
        times["clean"] += (end-start)

    def get_polys_from_matrix(self,rows,reduced_matrix):
        '''
        Takes a list of indicies corresponding to the rows of the reduced matrix and
        returns a list of polynomial objects
        '''
        startTime = time.time()
        shape = []
        p_list = []
        matrix_term_vals = [i.val for i in self.matrix_terms]

        # Finds the maximum size needed for each of the poly coeff tensors
        for i in range(len(matrix_term_vals[0])):
            # add 1 to each to compensate for constant term
            shape.append(max(matrix_term_vals, key=itemgetter(i))[i]+1)
        # Grabs each polynomial, makes coeff matrix and constructs object
        for i in rows:
            p = reduced_matrix[i]
            if clean:
                if np.sum(np.abs(p)) < global_accuracy:
                    continue
            #p[np.where(abs(p) < global_accuracy/1.e5)] = 0
            coeff = np.zeros(shape)
            for j,term in enumerate(matrix_term_vals):
                coeff[term] = p[j]

            if self.power:
                poly = MultiPower(coeff)
            else:
                poly = MultiCheb(coeff)

            if poly.lead_term != None:
                #poly.coeff = poly.coeff/poly.lead_coeff  #This code is maybe sketchy, maybe good.
                #print(poly.coeff)
                p_list.append(poly)
        endTime = time.time()
        times["get_poly_from_matrix"] += (endTime - startTime)
        return p_list

    def _add_poly_to_matrix(self, p, adding_r = False):
        '''
        Saves the polynomial to the set of polynomials that will be used in create_matrix.
        Also updates the list of leading terms and monomials in the matrix with the new values.
        
        adding_r is only true when the r's are being added, this way it knows to keep adding new monomials to the heap
        for further r calculation
        '''
        startTime = time.time()
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
        endTime = time.time()
        times["_add_poly_to_matrix"] += (endTime - startTime)
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
        '''Calculates the phi-polynomial's of the polynomials a and b.
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
        startTime = time.time()
        # Find the set of all pairs of index the function will run through

        # Index_new iterate the tuple of every combination of the new_polys.
        index_new = itertools.combinations(range(len(self.new_polys)),2)
        # Index_oldnew iterates the tuple of every combination of new and old polynomials
        index_oldnew = itertools.product(range(len(self.new_polys)),range(len(self.new_polys),
                                               len(self.old_polys)+len(self.new_polys)))
        B = set(itertools.chain(index_new,index_oldnew))

        # Iterating through both possible combinations.
        while B:
            i,j = B.pop()
            if self.phi_criterion(i,j,B,phi)== True:
                #calculate the phi's.
                poly = self.new_polys + self.old_polys
                p_a , p_b = self.calc_phi(poly[i],poly[j])
                # add the phi's on to the Groebner Matrix.
                self._add_poly_to_matrix(p_a)
                self._add_poly_to_matrix(p_b)
        endTime = time.time()
        times["calc_phi"] += (endTime - startTime)

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
        startTime = time.time()

        if phi == False:
            endTime = time.time()
            times["phi_criterion"] += (endTime - startTime)
            return True        
        
        # List of new and old polynomials.
        polys = self.new_polys+self.old_polys

        # Relative Prime check: If the lead terms of i and j are relative primes, phi is not needed
        if all([a*b == 0 for a,b in zip(polys[i].lead_term,polys[j].lead_term)]):
            endTime = time.time()
            times["phi_criterion"] += (endTime - startTime)
            return False        
        
        # Another criterion
        else:
            for l in range(len(polys)):
                #print ("For l = {}:".format(l))

                # Checks that l is not j or i.
                if l == j or l == i:
                    #print("\t{} is i or j".format(l))
                    continue

                # Sorts the tuple (i,l) or (l,i) in order of smaller to bigger.
                i_tuple = tuple(sorted((i,l)))
                j_tuple = tuple(sorted((j,l)))

                # i_tuple and j_tuple needs to not be in B.
                if j_tuple in B or i_tuple in B:
                    #print('\t{} or {} is in B'.format(j_tuple,i_tuple))
                    continue

                lcm = self._lcm(polys[i],polys[j])
                lead_l = polys[l].lead_term

                # See if LT(poly[l]) divides lcm(LT(i),LT(j))
                if all([i-j>=0 for i,j in zip(lcm,lead_l)]) :
                    #print("\tLT of poly[l] divides lcm(LT(i),LT(j)")
                    endTime = time.time()
                    times["phi_criterion"] += (endTime - startTime)
                    return False

        # Function will return True and calculate phi if none of the checks passed for all l's.
            endTime = time.time()
            times["phi_criterion"] += (endTime - startTime)
            return True


    def _build_maxheap(self):
        '''
        Builds a maxheap for use in r polynomial calculation
        '''
        startTime = time.time()
        self.monheap = MaxHeap()
        for mon in self.term_set:
            if mon not in self.lead_term_set: #Adds every monomial that isn't a lead term to the heap
                self.monheap.heappush(mon)
        endTime = time.time()
        times["buildHeap"] += (endTime - startTime)
    
    def sorted_polys_coeff(self):
        '''
        Sorts the polynomials by how much bigger the leading coefficient is than the rest of the coeff matrix.
        '''
        startTime = time.time()
        polys = self.new_polys+self.old_polys
        lead_coeffs = list()
        for poly in polys:
            lead_coeffs.append(poly.lead_coeff/np.sum(np.abs(poly.coeff))) #The lead_coeff to other stuff ratio.
        argsort_list = sorted(range(len(lead_coeffs)), key=lead_coeffs.__getitem__)[::]
        sorted_polys = list()
        for i in argsort_list:
            sorted_polys.append(polys[i])
        endTime = time.time()
        times["sorted_polys_coeff"] += (endTime - startTime)
        return sorted_polys
        
    def add_r_to_matrix(self):
        '''
        Finds the r polynomials and adds them to the matrix.
        First makes Heap out of all potential monomials, then finds polynomials with leading terms that divide it and
        add them to the matrix.
        '''
        startTime = time.time()
        self._build_maxheap()
        sorted_polys = self.sorted_polys_coeff()
        #sorted_polys = self.new_polys+self.old_polys #Use this if we don't want to sort the r's.
        while len(self.monheap) > 0:
            m = list(self.monheap.heappop().val)
            r = self.calc_r(m,sorted_polys)
            self._add_poly_to_matrix(r, adding_r = True)

        endTime = time.time()
        times["calc_r"] += (endTime - startTime)
    
   
   
    def create_matrix(self):
        startTime = time.time()
        
        biggest_shape = np.maximum.reduce([p.coeff.shape for p in self.matrix_polys])
        
        if self.power:
            biggest = MultiPower(np.zeros(biggest_shape), clean_zeros = False)
        else:
            biggest = MultiCheb(np.zeros(biggest_shape), clean_zeros = False)
        self.np_matrix = biggest.coeff.flatten()
        self.np_matrix = np.array(self.np_matrix, dtype = np.longdouble)
        
        flat_polys = list()
        for poly in self.matrix_polys:
            startFill = time.time()
            newMatrix = self.fill_size(biggest.coeff, poly.coeff)
            flat_polys.append(newMatrix.flatten())
            endFill = time.time()
            times["fill"] += (endFill - startFill)
        
        self.np_matrix = np.vstack(flat_polys[::-1])
                
        terms = np.zeros(biggest_shape, dtype = Term)
        startTerms = time.time()
        for i,j in np.ndenumerate(terms):
            terms[i] = Term(i)
        endTerms = time.time()
        times["terms"] += (endTerms - startTerms)
        
        self.matrix_terms = terms.flatten()
        self.sort_matrix()
        self.clean_matrix()

        self.np_matrix = self.row_swap_matrix(self.np_matrix)
                
        endTime = time.time()
        times["create_matrix"] += (endTime - startTime)

    def reduce_matrix(self, qr_reduction=True, triangular_solve = False):
        '''
        Reduces the matrix fully using either QR or LU decomposition. Adds the new_poly's to old_poly's, and adds to
        new_poly's any polynomials created by the reduction that have new leading monomials.
        Returns-True if new polynomials were found, False otherwise.
        '''
        startTime = time.time()
        if qr_reduction:            
            independentRows, dependentRows, Q = self.fullRank(self.np_matrix)
            fullRankMatrix = self.np_matrix[independentRows]
            
            startRRQR = time.time()
            reduced_matrix = self.rrqr_reduce(fullRankMatrix)
            
            non_zero_rows = np.sum(abs(reduced_matrix),axis=1) != 0
            
            reduced_matrix = reduced_matrix[non_zero_rows,:] #Only keeps the non_zero_polymonials
            endRRQR = time.time()
            times["rrqr_reduce"] += (endRRQR - startRRQR)
            
            '''
            #If I decide to use the fully reduce method.
            Q,R = qr(self.np_matrix)
            reduced_matrix = self.fully_reduce(R)
            non_zero_rows = np.sum(abs(reduced_matrix),axis=1)>0 ##Increasing this will get rid of small things.
            reduced_matrix = reduced_matrix[non_zero_rows,:] #Only keeps the non_zero_polymonials
            '''
            
            startTri = time.time()
            if triangular_solve:
                reduced_matrix = self.triangular_solve(reduced_matrix)
                if clean:
                    reduced_matrix = self.clean_zeros_from_matrix(reduced_matrix)
            endTri = time.time()
            times["triangular_solve"] += (endTri - startTri)

            #plt.matshow(reduced_matrix)
            #plt.matshow([i==0 for i in reduced_matrix])
            #plt.matshow([abs(i)<self.global_accuracy for i in reduced_matrix])
        else:
            #This thing is very behind the times.
            P,L,U = lu(self.np_matrix)
            reduced_matrix = U
            #reduced_matrix = self.fully_reduce(reduced_matrix, qr_reduction = False)

        
        #Get the new polynomials
        new_poly_spots = list()
        old_poly_spots = list()
        startLooking = time.time()
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
        endLooking = time.time()
        times["looking"] += (endLooking - startLooking)
        
        
        if triangular_solve:
            self.old_polys = self.get_polys_from_matrix(old_poly_spots, reduced_matrix)
        else:
            self.old_polys = self.new_polys + self.old_polys
        self.new_polys = self.get_polys_from_matrix(new_poly_spots, reduced_matrix)
        
        endTime = time.time()
        times["reduce_matrix"] += (endTime - startTime)
        
        if len(self.old_polys+self.new_polys) == 0:
            print("ERROR ERROR ERROR ERROR ERROR NOT GOOD NO POLYNOMIALS IN THE BASIS FIX THIS ASAP!!!!!!!!!!!!!")
            print(reduced_matrix)
            print(times)
            MultiCheb.printTime()

        return len(self.new_polys) > 0

   
   
    '''
    def rrqr_reduce(self, matrix): #My new sort of working one. Still appears to have some problems. Possibly from fullRank.
        if matrix.shape[0] <= 1 or matrix.shape[0]==1 or  matrix.shape[1]==0:
            return matrix
        height = matrix.shape[0]
        A = matrix[:height,:height] #Get the square submatrix
        B = matrix[:,height:] #The rest of the matrix to the right
        independentRows, dependentRows, Q = self.fullRank(A)
        nullSpaceSize = len(dependentRows)
        if nullSpaceSize == 0: #A is full rank
            #print("FULL RANK")
            Q,R = np.linalg.qr(matrix)
            if clean:
                return self.clean_zeros_from_matrix(R)
            else:
                return R
        else: #A is not full rank
            #print("NOT FULL RANK")
            #sub1 is the independentRows of the matrix, we will recursively reduce this
            #sub2 is the dependentRows of A, we will set this all to 0
            #sub3 is the dependentRows of Q.T@B, we will recursively reduce this.
            #We then return sub1 stacked on top of sub2+sub3
            
            Q[np.where(abs(Q) < global_accuracy)]=0
            bottom = matrix[dependentRows]
            BCopy = B.copy()
            sub3 = bottom[:,height:]
            sub3 = Q.T[-nullSpaceSize:]@BCopy
            if clean:
                sub3 = self.clean_zeros_from_matrix(sub3)
            sub3 = self.rrqr_reduce(sub3)
            
            sub1 = matrix[independentRows]
            sub1 = self.rrqr_reduce(sub1)            
            
            sub2 = bottom[:,:height]
            sub2[:] = np.zeros_like(sub2)
            
            reduced_matrix = np.vstack((sub1,np.hstack((sub2,sub3))))
            if clean:
                return self.clean_zeros_from_matrix(reduced_matrix)
            else:
                return reduced_matrix
    
    '''
        
    def rrqr_reduce(self, matrix): #Original One. Seems to be the more stable one from testing.
        if matrix.shape[0]==0 or matrix.shape[1]==0:
            return matrix
        if clean:
            matrix = self.clean_zeros_from_matrix(matrix)
        height = matrix.shape[0]
        A = matrix[:height,:height] #Get the square submatrix
        B = matrix[:,height:] #The rest of the matrix to the right
        startMatrix = time.time()
        Q,R,P = qr(A, pivoting = True) #rrqr reduce it
        endMatrix = time.time()
        times["matrixStuff"] += (endMatrix - startMatrix)        
        PT = self.inverse_P(P)
        diagonals = np.diagonal(R) #Go along the diagonals to find the rank
        rank = np.sum(np.abs(diagonals)>global_accuracy)
        if clean:
            R = self.clean_zeros_from_matrix(R)

        if rank == height: #full rank, do qr on it
            startMatrix = time.time()
            Q,R = qr(A)
            endMatrix = time.time()
            times["matrixStuff"] += (endMatrix - startMatrix)
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
    
    

###############################################################################
###############################################################################



#What we determine to be zero throughout the code
global_accuracy = 1.e-10
#If clean is true then at a couple of places (end of rrqr_reduce and end of add r to matrix) things close to 0 will be made 0.
#Might make it more stable, might make it less stable. Not sure.
clean = False

times = {} #Global dictionary to track the run times of each part of the code.
times["initialize"] = 0
times["sort"] = 0
times["clean"] = 0
times["get_poly_from_matrix"] = 0
times["_add_poly_to_matrix"] = 0
times["calc_phi"] = 0
times["phi_criterion"] = 0
times["calc_r"] = 0
times["reduce_matrix"] = 0
times["create_matrix"] = 0
times["terms"] = 0
times["fill"] = 0
times["looking"] = 0
times["triangular_solve"] = 0
times["rrqr_reduce"] = 0
times["fullRank"] = 0
times["matrixStuff"] = 0
times["sorted_polys_coeff"] = 0
times["buildHeap"] = 0
 
