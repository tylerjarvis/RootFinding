from __future__ import division, print_function
import numpy as np
from scipy.signal import convolve, fftconvolve
from groebner.polynomial import Polynomial
from groebner.maxheap import Term
import itertools
import math
import time

"""
1/11/17
Author: Rex McArthur
Creates a class of n-dim Power Basis polynomials. Tracks leading term,
coefficents, and inculdes basic operations (+,*,scaler multip, etc.)
Assumes GRevLex ordering, but should be extended.
Mostly used for testing vs other solvers
"""

times = dict()
times["mon_mult_power"] = 0

class MultiPower(Polynomial):
    """
    _____ params _______
    dim: int, number of variables, dimension of polynomial system
    terms: int, highest term of single variable power polynomials
    coeff: list(terms**dim) or np.array ([terms,] * dim), coefficents in given ordering
    order: string, monomial ordering desired for Grobner calculations
    lead_term: list, the index of the current leading coefficent



    _____ methods ______
    next_step:
        input- Current: list, current location in ordering
        output- the next step in ordering
    """
    def printTime():
        print(times)
    
    def clearTime():
        times["mon_mult_power"] = 0

    def __init__(self, coeff, order='degrevlex', lead_term=None, clean_zeros = True):
        super(MultiPower, self).__init__(coeff, order, lead_term, clean_zeros)
        
    def __add__(self,other):
        '''
        Here we add an addition class.
        '''
        if self.shape != other.shape:
            new_self, new_other = self.match_size(self.coeff,other.coeff)
        else:
            new_self, new_other = self.coeff, other.coeff
        return MultiPower((new_self + new_other), clean_zeros = False)

    def __sub__(self,other, scale = 1):
        '''
        Here we subtract the two polys
        '''
        if self.shape != other.shape:
            new_self, new_other = self.match_size(self.coeff,other.coeff)
        else:
            new_self, new_other = self.coeff, other.coeff
        return MultiPower((new_self - (scale*new_other)), clean_zeros = False)
<<<<<<< HEAD
=======

    def __mul__(self,other):
        '''
        here we add leading terms?
        '''
        if self.shape != other.shape:
            new_self, new_other = self.match_size(self.coeff,other.coeff)
        else:
            new_self, new_other = self.coeff, other.coeff
>>>>>>> 31746bd7c94315f0cfa00bcfe2784160c46b31e4

        return MultiPower(convolve(new_self, new_other))

    def __mul__(self,other):
        '''
        here we add leading terms?
        '''
        if self.shape != other.shape:
            new_self, new_other = self.match_size(self.coeff,other.coeff)
        else:
            new_self, new_other = self.coeff, other.coeff

        return MultiPower(convolve(new_self, new_other))

    def __eq__(self,other):
        '''
        check if coeff matrix is the same
        '''
        if self.shape != other.shape:
            return False
        else:
            return np.allclose(self.coeff, other.coeff)

    def __ne__(self,other):
        '''
        check if coeff matrix is not the same same
        '''
        return not (self == other)

    def mon_mult(self,M):
        '''
        M is a tuple of the powers in the monomial.
            Ex: x^3*y^4*z^2 would be input as (3,4,2)
        #P is the polynomial.
        '''
        M = np.array(M)
        start = time.time()
        tuple1 = []
        for i in M:
            list1 = (i,0)
            tuple1.append(list1)
        poly = MultiPower(np.pad(self.coeff, tuple1, 'constant', constant_values = 0), 
                          clean_zeros = False, lead_term = self.lead_term + M)
        end = time.time()
        times["mon_mult_power"] += (end-start)
        return poly

    def evaluate_at(self, point):
        super(MultiPower, self).evaluate_at(point)

        poly_value = 0
        for mon in self.monomialList():
            mon_value = 1
            for i in range(len(point)):
                var_value = pow(point[i], mon[i])
                mon_value *= pow(point[i], mon[i])
            mon_value *= self.coeff[mon]
            poly_value += mon_value

        if abs(poly_value) < 1.e-10:
            return 0
        else:
            return poly_value
