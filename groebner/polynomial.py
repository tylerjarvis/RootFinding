from __future__ import division, print_function
import numpy as np
from scipy.signal import fftconvolve, convolve
import itertools
from groebner.maxheap import Term
import time

times = dict()
times["updateLeadTerm"] = 0
times["monomialsList"] = 0
times["leadTermCount"] = 0
times["cleanCoeff"] = 0
times["initialize"] = 0

class Polynomial(object):

    def printTime():
        print(times)

    def clearTime():
        times["updateLeadTerm"] = 0
        times["monomialsList"] = 0
        times["leadTermCount"] = 0
        times["cleanCoeff"] = 0
        times["initialize"] = 0

    def __init__(self, coeff, order='degrevlex', lead_term=None, clean_zeros = True):
        '''
        terms, int- number of chebyshev polynomials each variable can have. Each dimension will have term terms
        dim, int- number of different variables, how many dim our tensor will be
        order, string- how you want to order your polynomials. Grevlex is default
        '''
        start = time.time()
        self.coeff = coeff
        if clean_zeros:
            self.clean_coeff()
        self.dim = self.coeff.ndim
        self.order = order
        self.shape = self.coeff.shape
        if lead_term is None:
            self.update_lead_term()
        else:
            self.lead_term = lead_term
            self.degree = sum(self.lead_term)
            self.lead_coeff = self.coeff[tuple(self.lead_term)]

        end = time.time()
        times["initialize"] += (end - start)

    def clean_coeff(self):
        """
        Gets rid of any 0's on the outside of the coeff matrix, not giving any info.
        """
        start = time.time()
        for axis in range(self.coeff.ndim):
            change = True
            while change:
                change = False
                if self.coeff.shape[axis] == 1:
                    continue
                axisCount = 0
                slices = list()
                for i in self.coeff.shape:
                    if axisCount == axis:
                        s = slice(i-1,i)
                    else:
                        s = slice(0,i)
                    slices.append(s)
                    axisCount += 1
                if np.sum(abs(self.coeff[slices])) == 0:
                    self.coeff = np.delete(self.coeff,-1,axis=axis)
                    change = True
            pass
        end = time.time()
        times["cleanCoeff"] += (end - start)
        pass


    def match_size(self,a,b):
        '''
        Matches the shape of the matrixes of two polynomials. This might not be the best place for it.
        '''
        a_shape, b_shape = list(a.shape), list(b.shape)
        if len(a_shape) != len(b_shape):
            add_to_shape = 0
            if len(a_shape) < len(b_shape):
                add_to_shape = len(b_shape) - len(a_shape)
                for i in range(add_to_shape):
                    a_shape.insert(0,1)
                a = a.reshape(a_shape)
            else:
                add_to_shape = len(a_shape) - len(b_shape)
                for i in range(add_to_shape):
                    b_shape.insert(0,1)
                b = b.reshape(b_shape)

        new_shape = [max(i,j) for i,j in itertools.zip_longest(a.shape, b.shape, fillvalue = 0)] #finds the largest length in each dimension
        # finds the difference between the largest length and the original shapes in each dimension.
        add_a = [i-j for i,j in itertools.zip_longest(new_shape, a.shape, fillvalue = 0)]
        add_b = [i-j for i,j in itertools.zip_longest(new_shape, b.shape, fillvalue = 0)]
        #create 2 matrices with the number of rows equal to number of dimensions and 2 columns
        add_a_list = np.zeros((len(new_shape),2))
        add_b_list = np.zeros((len(new_shape),2))
        #changes the second column to the values of add_a and add_b.
        add_a_list[:,1] = add_a
        add_b_list[:,1] = add_b
        #uses add_a_list and add_b_list to pad each polynomial appropriately.
        a = np.pad(a,add_a_list.astype(int),'constant')
        b = np.pad(b,add_b_list.astype(int),'constant')
        return a,b

    def monomialList(self):
        '''
        return
        ------
        monomials : list of tuples
            list of monomials that make up the polynomial in degrevlex order
        '''
        start = time.time()
        monomialTerms = list()
        for i in zip(*np.where(self.coeff != 0)):
            monomialTerms.append(Term(i))
        monomialTerms.sort()

        monomials = list()
        for i in monomialTerms[::-1]:
            monomials.append(i.val)

        end = time.time()
        times["monomialsList"] += (end - start)
        self.sortedMonomials = monomials
        return monomials

    def monSort(self):
        self.sortedMonomials = self.monomialList()

    def update_lead_term(self,start = None):
        startTime = time.time()

        non_zeros = list()
        for i in zip(*np.where(self.coeff != 0)):
            non_zeros.append(Term(i))
        if len(non_zeros) != 0:
            self.lead_term = max(non_zeros).val
            self.degree = sum(self.lead_term)
            self.lead_coeff = self.coeff[tuple(self.lead_term)]
        else:
            self.lead_term = None
            self.lead_coeff = 0

        endTime = time.time()
        times["leadTermCount"] += 1
        times["updateLeadTerm"] += (endTime - startTime)

    def evaluate_at(self, point):
        '''
        Evaluates the polynomial at the given point.

        parameters
        ----------
        point : tuple or list
            the point at which to evaluate the polynomial

        returns
        -------
        complex
            value of the polynomial at the given point
        '''
        if len(point) != len(self.coeff.shape):
            raise ValueError('Cannot evaluate polynomial in {} variables at point {}'\
            .format(self.dim, point))

    def __eq__(self,other):
        '''
        check if coeff matrix is the same
        '''
        if self.shape != other.shape:
            return False
        return np.allclose(self.coeff, other.coeff)

    def __ne__(self,other):
        '''
        check if coeff matrix is not the same same
        '''
        return not (self == other)
