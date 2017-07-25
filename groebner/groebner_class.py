from operator import itemgetter
import itertools
import numpy as np
from groebner import maxheap
import math
from groebner.multi_cheb import MultiCheb
from groebner.multi_power import MultiPower
from scipy.linalg import lu, qr, solve_triangular
from groebner.maxheap import Term
import matplotlib.pyplot as plt
import time
from collections import defaultdict

#What we determine to be zero throughout the code
global_accuracy = 1.e-12
#If clean is true then at a couple of places (end of rrqr_reduce and end of add r to matrix) things close to 0 will be made 0.
#Might make it more stable, might make it less stable. Not sure.
clean = True

times = {} #Global dictionary to track the run times of each part of the code.

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
        self.duplicate_lms - The leading terms that occur multiple times. Keep these as old_polys
        '''
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
        self.original_lm_dict = {}
        self.not_needed_lms = set()
        self.duplicate_lms = set()
        self.matrix_polys = list()
        pass
    
    def divides(self,a,b):
        '''
        Takes two polynomials, a and b. Returns True if the lm of b divides the lm of a. False otherwise.
        '''
        diff = tuple(i-j for i,j in zip(a.lead_term,b.lead_term))
        return all(i >= 0 for i in diff)
    
    def sorted_polys_monomial(self, polys):
        '''
        Sorts the polynomials by the number of monomials they have, the ones with the least amount first.
        '''
        num_monomials = list()
        for poly in polys:
            num_monomials.append(len(np.where(poly.coeff != 0)[0]))
        argsort_list = sorted(range(len(num_monomials)), key=num_monomials.__getitem__)[::]
        sorted_polys = list()
        for i in argsort_list:
            sorted_polys.append(polys[i])
        return sorted_polys
    
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
        self.original_lm_dict = {}
        self.original_lms = set()
        self.not_needed_lms = set()
        self.duplicate_lms = set()
        
        self.matrix_polys = list()
        
        if final_time:
            self._add_polys(self.new_polys + self.old_polys)
            for poly in self.new_polys + self.old_polys:
                self.original_lms.add(Term(poly.lead_term))
        else:
            for p in self.new_polys + self.old_polys:
                if p.lead_term != None:
                    self.original_lms.add(Term(p.lead_term))
                    self.original_lm_dict[Term(p.lead_term)] = p
                pass

            old_polys = self.old_polys
            new_polys = self.new_polys
            polys = old_polys + new_polys

            polys = self.sorted_polys_monomial(polys)

            self.old_polys = list()
            self.new_polys = list()

            lms = defaultdict(list)
            for i in polys:
                lms[i.lead_term].append(i)
                pass

            polys = list()

            for i in lms:
                if len(lms[i]) > 1: #It's duplicated
                    #self.duplicate_lms.add(Term(i))
                    self.new_polys.append(lms[i][0])
                    polys.append(lms[i][0])
                    lms[i].remove(lms[i][0])
                    self._add_polys(lms[i]) #Still lets stuff be reduced if a poly reduces all of them.
                else:
                    polys.append(lms[i][0])

            divides_out = list()

            for i in polys:
                for j in polys:
                    if i != j:
                        if self.divides(i,j): # j divides into i
                            divides_out.append(i)
                            self.not_needed_lms.add(Term(i.lead_term))
                            self._add_poly_to_matrix(i)
                            self._add_poly_to_matrix(j.mon_mult(tuple(a-b for a,b in zip(i.lead_term,j.lead_term))))
                            break
                pass

            for i in polys:
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
        pass

    def solve(self, qr_reduction = True, reducedGroebner = True):
        '''
        The main function. Initializes the matrix, adds the phi's and r's, and then reduces it. Repeats until the reduction
        no longer adds any more polynomials to the matrix. Print statements let us see the progress of the code.
        '''
        startTime = time.time()
        
        polys_were_added = True
        i=1 #Tracks what loop we are on.
        while polys_were_added:
            print("Starting Loop #"+str(i))
            print("Num Polys - ", len(self.new_polys + self.old_polys))
            self.initialize_np_matrix()
            self.add_phi_to_matrix()
            self.add_r_to_matrix()
            self.create_matrix()
            print(self.np_matrix.shape)
            polys_were_added = self.reduce_matrix(qr_reduction = qr_reduction, triangular_solve = False) #Get rid of triangular solve when done testing
            i+=1

        self.get_groebner()
        if reducedGroebner:
            self.reduce_groebner_basis()
        
        endTime = time.time()
        print("WE WIN")
        print("Run time was {} seconds".format(endTime - startTime))
        print(times)
        print("Basis - ")
        for poly in self.groebner_basis:
            print(poly.coeff)
            break #print just one
        
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
        print(self.np_matrix.shape)
        self.reduce_matrix(triangular_solve = True)
        for poly in self.old_polys:
            poly.coeff[abs(poly.coeff) < global_accuracy] = 0
        self.groebner_basis = self.old_polys
        pass
    
    def get_groebner(self):
        '''
        Checks to see if our basis includes 1. If so, that is the basis. Also, removes 0 polynomials. Then makes an
        attribute self.groebner_basis that holds the groebner basis, not neccesarily reduced.
        '''
        self.groebner_basis = list()
        for poly in self.old_polys:
            if all([i==1 for i in poly.coeff.shape]): #This if statement can be removed when done testing
                self.groebner_basis = list()
                self.groebner_basis.append(poly)
                return
            #if np.sum(np.abs(poly.coeff)) > global_accuracy:
            self.groebner_basis.append(poly)
        pass

    def sort_matrix(self):
        '''
        Sorts the matrix into degrevlex order.
        '''
        start = time.time()
        argsort_list, self.matrix_terms = self.argsort(self.matrix_terms)        
        self.np_matrix = self.np_matrix[:,argsort_list]
        end = time.time()
        times["sort"] += (end-start)
        pass

    def argsort(self, index_list):
        '''
        Returns an argsort list for the index, as well as sorts the list in place
        '''
        argsort_list = sorted(range(len(index_list)), key=index_list.__getitem__)[::-1]
        index_list.sort()
        return argsort_list, index_list[::-1]

    def clean_matrix(self):
        '''
        Gets rid of rows and columns in the np_matrix that are all zero.
        '''
        start = time.time()
        ##This would replace all small values in the matrix with 0.
        if clean:
            self.np_matrix[np.where(abs(self.np_matrix) < global_accuracy)]=0

        #Removes all 0 monomials
        non_zero_monomial = np.sum(abs(self.np_matrix), axis=0)>0 ##Increasing this will get rid of small things.
        self.np_matrix = self.np_matrix[:,non_zero_monomial] #only keeps the non_zero_monomials
        self.matrix_terms = self.matrix_terms[non_zero_monomial]

        #Removes all 0 polynomials. This shouldn't be needed, we should have none of these.
        #non_zero_polynomial = np.sum(abs(self.np_matrix),axis=1)>0 ##Increasing this will get rid of small things.
        #self.np_matrix = self.np_matrix[non_zero_polynomial,:] #Only keeps the non_zero_polymonials
        
        end = time.time()
        times["clean"] += (end-start)
        pass

    def get_poly_from_matrix(self,rows,reduced_matrix):
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

        #Keeping track of lead_terms
        if sum(a_diff)==0 and sum(b_diff)==0:
            self.duplicate_lms.add(Term(a.lead_term))
            return None, None                  #Put this back in when done with testing!
        elif sum(a_diff)==0:
            self.not_needed_lms.add(Term(a.lead_term))
            return None, b.mon_mult(b_diff)     #Put this back in when done with testing!
        elif sum(b_diff)==0:
            self.not_needed_lms.add(Term(b.lead_term))
            return a.mon_mult(a_diff), None      #Put this back in when done with testing!
        
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
                
                #print(p_a.coeff)
                #print(p_b.coeff)
                self._add_poly_to_matrix(p_a)
                self._add_poly_to_matrix(p_b)
        endTime = time.time()
        times["calc_phi"] += (endTime - startTime)
        pass

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
        self.monheap = maxheap.MaxHeap()
        for mon in self.term_set:
            if mon not in self.lead_term_set: #Adds every monomial that isn't a lead term to the heap
                self.monheap.heappush(mon)
        endTime = time.time()
        times["buildHeap"] += (endTime - startTime)
        pass
    
    def sorted_polys_coeff(self):
        '''
        Sorts the polynomials by their leading coefficient, largest ones first.
        '''
        startTime = time.time()
        polys = self.new_polys+self.old_polys
        lead_coeffs = list()
        for poly in polys:
            lead_coeffs.append(poly.lead_coeff)
        argsort_list = sorted(range(len(lead_coeffs)), key=lead_coeffs.__getitem__)[::]
        sorted_polys = list()
        for i in argsort_list:
            sorted_polys.append(polys[i])
        endTime = time.time()
        times["sorted_polys_coeff"] += (endTime - startTime)
        return sorted_polys
        
    def calc_r(self, m, sorted_polys):
        '''
        Finds the r polynomial that has a leading monomial m
        Returns the polynomial.
        '''
        for p in sorted_polys:
                l = list(p.lead_term)
                if all([i<=j for i,j in zip(l,m)]) and len(l) == len(m): #Checks to see if l divides m
                    c = [j-i for i,j in zip(l,m)]
                    if not l == m: #Make sure c isn't all 0
                        return p.mon_mult(c)
        pass
    
    def add_r_to_matrix(self):
        '''
        Finds the r polynomials and adds them to the matrix.
        First makes Heap out of all potential monomials, then finds polynomials with leading terms that divide it and
        add them to the matrix.
        '''
        startTime = time.time()
        self._build_maxheap()
        #sorted_polys = self.sorted_polys_coeff()
        sorted_polys = self.new_polys+self.old_polys #When done testing remove this line and uncomment the one above.
        while len(self.monheap) > 0:
            m = list(self.monheap.heappop().val)
            r = self.calc_r(m,sorted_polys)
            self._add_poly_to_matrix(r, adding_r = True)

        endTime = time.time()
        times["calc_r"] += (endTime - startTime)
        pass
    
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
        Returns just the coeff matrix as that is all we need in the Groebner create_matrix code.
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
        startTime = time.time()
        
        biggest_shape = np.maximum.reduce([p.shape for p in self.matrix_polys])
        
        if self.power:
            biggest = MultiPower(np.zeros(biggest_shape))
        else:
            biggest = MultiCheb(np.zeros(biggest_shape))
        self.np_matrix = biggest.coeff.flatten()
        self.np_matrix = np.array(self.np_matrix, dtype = np.longdouble)
        
        flat_polys = list()
        for poly in self.matrix_polys:
            #print(poly.lead_coeff)
            #poly.coeff = poly.coeff/poly.lead_coeff #THIS IS FOR TESTING
            startFill = time.time()
            newMatrix = self.fill_size(biggest.coeff, poly.coeff)
            flat_polys.append(newMatrix.ravel())
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
        pass

    def reduce_matrix(self, qr_reduction=True, triangular_solve = False):
        '''
        Reduces the matrix fully using either QR or LU decomposition. Adds the new_poly's to old_poly's, and adds to
        new_poly's any polynomials created by the reduction that have new leading monomials.
        Returns-True if new polynomials were found, False otherwise.
        '''
        startTime = time.time()
        if qr_reduction:
            #plt.matshow(self.np_matrix)
            #print(np.max(abs(self.np_matrix)))
            #plt.matshow([i!=0 for i in self.np_matrix])
            #plt.matshow([abs(i)>1.e-3 for i in self.np_matrix])
            #plt.matshow([abs(i)>1.e8 for i in self.np_matrix])
            
            independentRows, dependentRows, Q = self.fullRank(self.np_matrix)
            fullRankMatrix = self.np_matrix[independentRows]
            
            startRRQR = time.time()
            reduced_matrix = self.rrqr_reduce(fullRankMatrix)
            
            non_zero_rows = np.sum(abs(reduced_matrix),axis=1)>0 ##Increasing this will get rid of small things.
            
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

            #plt.matshow([i==0 for i in reduced_matrix])
            
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

        #print(reduced_matrix)
        
        #Get the new polynomials
        new_poly_spots = list()
        old_poly_spots = list()
        old_poly_lms = list()
        
        startLooking = time.time()
        already_looked_at = set() #rows whose leading monomial we've already checked
        for i, j in zip(*np.where(reduced_matrix!=0)):
            if i in already_looked_at: #We've already looked at this row
                continue
            elif self.matrix_terms[j] in self.lead_term_set: #The leading monomial is not new.
                if self.matrix_terms[j] in self.original_lms - (self.not_needed_lms - self.duplicate_lms): #Reduced old poly
                    old_poly_spots.append(i)
                    old_poly_lms.append(self.matrix_terms[j])
                    if j in self.duplicate_lms:
                        self.duplicate_lms.remove(j) #So if we have duplicates we only add one of them.
                already_looked_at.add(i)
                continue
            else:
                already_looked_at.add(i)
                new_poly_spots.append(i) #This row gives a new leading monomial
        endLooking = time.time()
        times["looking"] += (endLooking - startLooking)
        
        if triangular_solve:
            self.old_polys = self.get_poly_from_matrix(old_poly_spots, reduced_matrix)
        else:
            #self.old_polys = self.old_polys + self.new_polys #This is for testing
            self.old_polys = list()
            for i in old_poly_lms:
            #for i in self.original_lm_dict: #Testing instead of above line.
                self.old_polys.append(self.original_lm_dict[i])        
        
        self.new_polys = self.get_poly_from_matrix(new_poly_spots, reduced_matrix)
        
        endTime = time.time()
        times["reduce_matrix"] += (endTime - startTime)
        
        if len(self.old_polys+self.new_polys) == 0:
            print("ERROR ERROR ERROR ERROR ERROR NOT GOOD NO POLYNOMIALS IN THE BASIS FIX THIS ASAP!!!!!!!!!!!!!")
            for i in self.original_lm_dict:
                print(i, self.original_lm_dict[i])
            print(reduced_matrix)
        
        return len(self.new_polys) > 0

    def clean_zeros_from_matrix(self,matrix):
        '''
        Gets rid of rows and columns in the np_matrix that are all zero.
        '''
        ##This would replace all small values in the matrix with 0.
        matrix[np.where(np.abs(matrix) < global_accuracy)]=0
        return matrix

    def fullRank(self, matrix):
        '''
        Finds the full rank of a matrix.
        Returns independentRows - a list of rows that have full rank, and 
        dependentRows - rows that can be removed without affecting the rank
        Q - The Q matrix used in RRQR reduction in finding the rank
        '''
        height = matrix.shape[0]
        Q,R,P = qr(matrix, pivoting = True)
        diagonals = np.diagonal(R) #Go along the diagonals to find the rank
        rank = np.sum(np.abs(diagonals)>global_accuracy)
        #print(diagonals)
        numMissing = height - rank
        if numMissing == 0: #Full Rank. All rows independent
            return [i for i in range(height)],[],None
        else:
            #Find the rows we can take out. These are ones that are non-zero in the last rows of Q transpose, as QT*A=R.
            #To find multiple, we find the pivot columns of Q.T
            QMatrix = Q.T[-numMissing:]
            Q1,R1,P1 = qr(QMatrix, pivoting = True)
            independentRows = P1[R1.shape[0]:] #Other Columns
            dependentRows = P1[:R1.shape[0]] #Pivot Columns
            return independentRows,dependentRows,Q
    
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
        pass
    
    '''
        
    def rrqr_reduce(self, matrix): #Original One. Seems to be the more stable one from testing.
>>>>>>> 2c0dafc0c4662dc4ac4c0114d8ca92e2c0ecaef1
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
<<<<<<< HEAD
        rank = np.sum(np.abs(diagonals)>self.global_accuracy)
=======
        rank = np.sum(np.abs(diagonals)>global_accuracy)
        if clean:
            R = self.clean_zeros_from_matrix(R)

>>>>>>> 2c0dafc0c4662dc4ac4c0114d8ca92e2c0ecaef1
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
    '''

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
                pass

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
        pass
    
    
