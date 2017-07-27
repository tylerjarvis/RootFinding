from __future__ import division, print_function
import numpy as np
from scipy.signal import fftconvolve, convolve
import itertools
from groebner.maxheap import Term

class Polynomial(object):
    def __init__(self, coeff, order='degrevlex', lead_term=None, clean_zeros = True):
        '''
        terms, int- number of chebyshev polynomials each variable can have. Each dimension will have term terms
        dim, int- number of different variables, how many dim our tensor will be
        order, string- how you want to order your polynomials. Grevlex is default
        '''
        self.coeff = coeff
        if clean_zeros:
            self.clean_coeff()
        self.dim = self.coeff.ndim
        self.terms = np.prod(self.coeff.shape)
        self.order = order
        self.shape = self.coeff.shape
        self.max_term = np.max(self.shape) - 1
        if lead_term is None:
            self.update_lead_term()
        else:
            self.lead_term = lead_term

    def clean_coeff(self):
        """
        Gets rid of any 0's on the outside of the coeff matrix, not giving any info.
        """
        sum_values = np.sum(abs(self.coeff))
        if sum_values == 0:
            return
        for axis in range(self.coeff.ndim):
            change = False
            while not change:
                temp = np.delete(self.coeff,-1,axis=axis)
                sum_temp = np.sum(abs(temp))
                if abs(sum_temp - sum_values) < 1.e-15:
                    self.coeff = temp
                else:
                    change = True
                pass
            pass
        pass
    """
    def check_column_overload(self, max_values, current, column):
        '''
        Checks to make sure that we aren't going into the negatives, aka the current value can't ever be greater
        than the max_values value. We check at the column where we have just added stuff and might have an
        overflow
        Return true if the whole thing is full and needs to increment i again. False otherwise.
        '''
        initial_column = column
        if(current[column] > max_values[column]):
            initial_amount = current[column]
            extra = current[column] - max_values[column]
            current[column] = max_values[column]
            while(extra>0):
                if(column==0):
                    current[0] += extra
                    #Sets all the stuff back in the initial row, needed if the while loop is used.
                    for i in range(0, initial_column):
                        current[i+1] += current[i]
                        current[i] = 0
                    return True
                else:
                    column -=1
                    allowed = max_values[column] - current[column]
                    if(allowed > extra):
                        current[column] += extra
                        extra = 0
                    else:
                        current[column] += allowed
                        extra -= allowed
            return False
        else:
            return False

    def degrevlex_gen(self):
        '''
        yields grevlex ordering co-ordinates in order to find
        the leading coefficent
        Note - this is just meant to quickly find the leading coefficient. For trying to grab all the non-zero
        terms     for i in zip(*np.where(poly.coeff != 0)):  will be much faster.
        '''
        max_values = tuple(self.shape)-np.ones_like(self.shape)
        base = max_values
        #print("Base - ",base)
        current = np.zeros(self.dim)
        yield base-current
        while True:
            for i in range(1, sum(max_values)+1):
                onward = True
                #set the far right column to i
                current = np.zeros(self.dim)
                current[self.dim-1] = i
                #This can't return false, as we start at the begenning. Always has enough room to spill over.
                self.check_column_overload(max_values, current, self.dim-1)
                #print("Current - ",current)
                yield base - current
                while onward:
                    #Find the leftmost thing
                    #left_most_spot = np.where(current != 0)[0][0]
                    for j in range(0, self.dim):
                        if(current[j] != 0):
                            left_most_spot = j
                            break
                    if(left_most_spot != 0):
                        #Slide it to the left
                        current[left_most_spot] -= 1
                        current[left_most_spot-1] += 1
                        if self.check_column_overload(max_values, current, left_most_spot-1):
                            onward = False
                        else:
                            #print("Current - ",current)
                            yield base - current
                    elif(current[j] == i):
                        #Reset it for the next run
                        current[0] = 0
                        onward = False
                        #THIS IS WRONG, THE CURRENT SPOT MIGHT BE ABLE TO HOLD MORE!
                    else:
                        #if I'm at the end push back everything to the next leftmost thing and slide it plus 1
                        amount = current[0]
                        for j in range(1,self.dim):
                            if(current[j] != 0):
                                next_left_most_spot = j
                                break
                        current[0] = 0
                        current[next_left_most_spot] -= 1
                        current[next_left_most_spot-1] += amount+1

                        spot_to_check = next_left_most_spot-1
                        #Loops throught this until everything is balanced all right or we need to increase i
                        while(self.check_column_overload(max_values, current, spot_to_check)):
                            new_spot_to_check = -1
                            for j in range(spot_to_check+1, self.dim):
                                if(current[j] != 0):
                                    new_spot_to_check = j
                                    break
                            if(new_spot_to_check == -1):
                                onward = False
                                break
                            else:
                                amount = current[spot_to_check]
                                current[spot_to_check] = 0
                                current[new_spot_to_check] -=1
                                current[new_spot_to_check-1] += (amount+1)
                                spot_to_check = new_spot_to_check-1
                        if(onward):
                            #print("Current - ",current)
                            yield base-current
            return
    
    """
    def monomialList(self):
        '''
        return
        ------
        monomials : list of tuples
            list of monomials that make up the polynomial in degrevlex order
        '''
        monomialTerms = list()
        for i in zip(*np.where(self.coeff != 0)):
            monomialTerms.append(Term(i))
        monomialTerms.sort()
        
        monomials = list()
        for i in monomialTerms[::-1]:
            monomials.append(i.val)
        
        #gen = self.degrevlex_gen()
        #for index in gen:
        #    index = tuple(map(lambda i: int(i), index))
        #    if (self.coeff[index] != 0):
        #        monomials.append(index)
        return monomials
    
    def update_lead_term(self,start = None):
        found = False

        non_zeros = set()
        for i in zip(*np.where(self.coeff != 0)):
            non_zeros.add(Term(i))
        if len(non_zeros) != 0:
            self.lead_term = max(non_zeros).val
            self.lead_coeff = self.coeff[tuple(self.lead_term)]
        else:
            self.lead_term = None
            self.lead_coeff = 0

        """ THE GENERATOR IS BROKEN RIGHT NOW. UNTIL FIXED USE THIS NEW, ALTHOUGH POSSIBLY SLOWER CODE.
        if self.order == 'degrevlex':
            gen = self.degrevlex_gen()
            for idx in gen:
                print(idx)
                idx = tuple(map(lambda i: int(i), idx))
                if self.coeff[tuple(idx)] != 0:
                    self.lead_term = idx
                    self.lead_coeff = self.coeff[tuple(idx)]
                    found = True
                    break
        if not found:
            self.lead_term = None
            self.lead_coeff = 0
        """

        #print('Leading Coeff is {}'.format(self.lead_term))

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
