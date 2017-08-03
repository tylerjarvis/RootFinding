import numpy as np
from scipy.signal import fftconvolve, convolve
import itertools
from groebner.utils import Term
import time
from numpy.polynomial import chebyshev as cheb
import math


times = dict()
times["mon_mult_power"] = 0
times["mon_mult_cheb"] = 0
times["updateLeadTerm"] = 0
times["monomialsList"] = 0
times["leadTermCount"] = 0
times["cleanCoeff"] = 0
times["initialize"] = 0
times["match_size"] = 0

class Polynomial(object):

    def printTime():
        print(times)

    def printLeadTermCount():
        print(times["leadTermCount"])

    def clearTime():
        times["updateLeadTerm"] = 0
        times["monomialsList"] = 0
        times["leadTermCount"] = 0
        times["cleanCoeff"] = 0
        times["initialize"] = 0
        times["match_size"] = 0

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
            self.lead_term = tuple(lead_term)
            self.degree = sum(self.lead_term)
            self.lead_coeff = self.coeff[self.lead_term]

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
        start = time.time()
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
        end = time.time()
        times["match_size"] += (end-start)
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

    def update_lead_term(self):
        startTime = time.time()

        non_zeros = list()
        for i in zip(*np.where(self.coeff != 0)):
            non_zeros.append(Term(i))
        if len(non_zeros) != 0:
            self.lead_term = max(non_zeros).val
            self.degree = sum(self.lead_term)
            self.lead_coeff = self.coeff[self.lead_term]
        else:
            self.lead_term = None
            self.lead_coeff = 0

        endTime = time.time()
        times["leadTermCount"] += 1
        times["updateLeadTerm"] += (endTime - startTime)

    def evaluate_at(self, point):
        '''
        Evaluates the polynomial at the given point. This method is overridden
        by the MultiPower and MultiCheb classes, so this definition only
        checks if the polynomial can be evaluated at the given point.

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
        check if coeff matrix is not the same
        '''
        return not (self == other)


###############################################################################

#### MULTI_CHEB ###############################################################

class MultiCheb(Polynomial):
    """
    _____ params _______
    dim: int, number of variables, dimension of chebyshev system
    terms: int, highest term of single variable chebyshev polynomials
    coeff: list(terms**dim) or np.array ([terms,] * dim), coefficents in given ordering
    order: string, monomial ordering desired for Groebner calculations
    lead_term: list, the index of the current leading coefficent



    _____ methods ______
    next_step:
        input- Current: list, current location in ordering
        output- the next step in ordering
    """

    def printTime():
        print(times)

    def clearTime():
        times["mon_mult_cheb"] = 0

    def __init__(self, coeff, order='degrevlex', lead_term=None, clean_zeros = True):
        super(MultiCheb, self).__init__(coeff, order, lead_term, clean_zeros)


    def __add__(self,other):
        '''
        Here we add an addition method
        '''
        if self.shape != other.shape:
            new_self, new_other = self.match_size(self.coeff,other.coeff)
        else:
            new_self, new_other = self.coeff, other.coeff

        return MultiCheb(new_self + new_other)

    def __sub__(self,other, scale = 1):
        '''
        Here we subtract the two polys coeffs
        '''
        if self.shape != other.shape:
            new_self, new_other = self.match_size(self.coeff,other.coeff)
        else:
            new_self, new_other = self.coeff, other.coeff
        return MultiCheb((new_self - (scale*new_other)), clean_zeros = False)

    def _reverse_axes(self):
        """
        Reverse the axes of the coeff tensor.
        """
        return self.coeff.flatten()[::-1].reshape(self.coeff.shape)

    def fold_for_reg_mult(temp, half, dim_to_fold, dim):
        """
        Function folds matrix in the middle of each dimension.
        To fold we take the first half and reverse in the dimension we are folding
        and then adding that to the second half of the matrix.
        For example, [1,2,3,4] folded around 3 gives [3,6,1] since
        [1,2,3] becomes [3,2,1] and we add that to [3,4,0] then divide the first term by 2.
        """
        slice0 = slice(None, half+1, None) #slice to get first half of the matrix.
        slice1 = slice(None, None, -1) #slice to reverse first half of the matrix.
        slice2 = slice(half,None,None) #slice to get the second half ot the matrix.
        slice3 = slice(0,1,None) #slice to take the row/column/piece that was added twice.

        #creates an index with a slice for each dimension.
        indexer0 = [slice(None,None,None)]*dim #
        indexer1 = [slice(None,None,None)]*dim
        indexer2 = [slice(None,None,None)]*dim

        #Changes the index of the correct dimmension in the indexers for array slicing.
        indexer0[dim_to_fold] = slice0
        indexer1[dim_to_fold] = slice1
        indexer2[dim_to_fold] = slice2

        #Takes the first half, reverses it and adds it to the second half.
        p2 = temp[indexer0][indexer1] + temp[indexer2]
        #divides the first piece by 2.
        indexer2[dim_to_fold] = slice3
        p2[indexer2] = p2[indexer2]/2.

        return p2

    def fold_in_i_dir(solution_matrix, dim, i, x, fold_idx):
        """
        Folds around a fold_inx and returns new solution.
        solution_matrix is polynomial to be folded
        dim is the number of dimensions of solution_matrix
        i represents the dimensions being folded.
        x is the size of the solution matrix in the dimension being folded
        fold_idx is the index to fold around.
        """
        if fold_idx == 0:
            return solution_matrix

        sol = np.zeros_like(solution_matrix) #Matrix of zeroes used to insert the new values..
        slice_0 = slice(None, 1, None) # index to take first slice
        slice_1 = slice(fold_idx, fold_idx+1, None) # index to take slice that contains the axis folding around.

        #indexers are made with a slice index for every dimension.
        indexer1 = [slice(None)]*dim
        indexer2 = [slice(None)]*dim
        indexer3 = [slice(None)]*dim

        #Changes the index in each indexer for the correct dimension
        indexer1[i] = slice_0
        indexer2[i] = slice_1

        #makes first slice in sol equal to the slice we fold around in solution_matrix
        sol[indexer1] = solution_matrix[indexer2]

        #Loop adds the slices above and below the slice we rotate around and inserts solutions in sol.
        for n in range(x):

            slice_2 = slice(n+1, n+2, None) #Used to imput new values in sol.
            slice_3 = slice(fold_idx+n+1, fold_idx+n+2, None) #Used to find slices that are n above fold_idx
            slice_4 = slice(fold_idx-n-1, fold_idx-n, None) #Used to find slices that are n below fold_idx

            indexer1[i] = slice_2
            indexer2[i] = slice_3
            indexer3[i] = slice_4

            #if statement checks to ensure that slices to be added are contained in the matrix.
            if fold_idx-n-1 < 0:
                if fold_idx+n+2 > x:
                    break
                else:
                    sol[indexer1] = solution_matrix[indexer2]
            else:
                if fold_idx+n+2 > x:
                    sol[indexer1] = solution_matrix[indexer3]
                else:
                    sol[indexer1] = solution_matrix[indexer3] + solution_matrix[indexer2]

        return sol

    def mon_mult(self, idx, returnType = 'Poly'):
        """
        Multiplies a Chebyshev polynomial by a monomial
        -------
        Parameters:
            self: A MultiCheb object
            idx: The index of the monomial to multiply self by.
            returnType: if 'Poly' then returns a polynomial object
        -------
        Returns:
            MultiCheb object or a matrix if returnType is not 'Poly'
        -------
        """
        start = time.time()
        initial_matrix = self.coeff
        for i in range(len(idx)):
            idx_zeros = np.zeros(len(idx),dtype = int)
            idx_zeros[i] = idx[i]
            initial_matrix = MultiCheb.mon_mult1(initial_matrix, idx_zeros, i)
        if returnType == 'Poly':
            end = time.time()
            times["mon_mult_cheb"] += (end - start)
            return MultiCheb(initial_matrix, lead_term = self.lead_term + np.array(idx), clean_zeros = False)
        elif returnType == 'Matrix':
            end = time.time()
            times["mon_mult_cheb"] += (end - start)
            return initial_matrix

    def mon_mult1(initial_matrix, idx, dim_mult):
        """
        Executes monomial multiplication in one dimension
        -------
        Parameters:
            initial_matrix: matrix of coefficients that represents a Chebyshev polynomial
            idx: the index of a monomial of one variable to multiply the Chebyshev polynomial by
            dim_mult: the location of the non-zero value in idx.
        -------
        Returns:
            matrix of coeff that is the result of the one dimensial monomial multiplication.
        -------
        """
        pad_values = list()
        for i in idx: #iterates through monomial and creates a tuple of pad values for each dimension
            pad_dim_i = (i,0)
            #In np.pad each dimension is a tuple of (i,j) where i is how many to pad in front and j is how many to pad after.
            pad_values.append(pad_dim_i)
        p1 = np.pad(initial_matrix, (pad_values), 'constant')

        largest_idx = [i-1 for i in initial_matrix.shape]
        new_shape = [max(i,j) for i,j in itertools.zip_longest(largest_idx, idx, fillvalue = 0)] #finds the largest length in each dimmension
        if initial_matrix.shape[dim_mult] <= idx[dim_mult]:
            add_a = [i-j for i,j in itertools.zip_longest(new_shape, largest_idx, fillvalue = 0)]
            add_a_list = np.zeros((len(new_shape),2))
            #changes the second column to the values of add_a and add_b.
            add_a_list[:,1] = add_a
            #uses add_a_list and add_b_list to pad each polynomial appropriately.
            initial_matrix = np.pad(initial_matrix,add_a_list.astype(int),'constant')

        number_of_dim = initial_matrix.ndim
        shape_of_self = initial_matrix.shape

        #Loop iterates through each dimension of the polynomial and folds in that dimension
        for i in range(number_of_dim):
            if idx[i] != 0:
                initial_matrix = MultiCheb.fold_in_i_dir(initial_matrix, number_of_dim, i, shape_of_self[i], idx[i])
        if p1.shape != initial_matrix.shape:
            idx = [i-j for i,j in zip(p1.shape,initial_matrix.shape)]
            pad_values = list()
            for i in idx:
                pad_dim_i = (0,i)
                pad_values.append(pad_dim_i)
            initial_matrix = np.pad(initial_matrix, (pad_values), 'constant')
        Pf = p1 + initial_matrix
        return .5*Pf

    def evaluate_at(self, point):
        super(MultiCheb, self).evaluate_at(point)

        poly_value = complex(0)
        for mon in self.monomialList():
            mon_value = 1
            for i in range(len(point)):
                cheb_deg = mon[i]
                cheb_coeff = [0. for i in range(cheb_deg)]
                cheb_coeff.append(1.)
                cheb_val = cheb.chebval([point[i]], cheb_coeff)[0]
                mon_value *= cheb_val
            mon_value *= self.coeff[mon]
            poly_value += mon_value

        if abs(poly_value) < 1.e-10:
            return 0
        else:
            return poly_value

###############################################################################

#### MULTI_POWER ##############################################################

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

    def mon_mult(self, M, returnType = 'Poly'):
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
        if returnType == 'Poly':
            poly = MultiPower(np.pad(self.coeff, tuple1, 'constant', constant_values = 0), 
                          clean_zeros = False, lead_term = self.lead_term + M)
            end = time.time()
            times["mon_mult_power"] += (end-start)
            return poly
        elif returnType == 'Matrix':
            matrix = np.pad(self.coeff, tuple1, 'constant', constant_values = 0)
            end = time.time()
            times["mon_mult_power"] += (end-start)
        return matrix

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

###############################################################################

#### CONVERT_POLY #############################################################



def conv_cheb(T):
    """
    Convert a chebyshev polynomial to the power basis representation.
    Args:
        T (): The chebyshev polynomial to convert.
    Returns:
        new_conv (): The chebyshev polynomial converted to the power basis representation.

    """
    conv = cheb.cheb2poly(T)
    if conv.size == T.size:
        return conv
    else:
        pad = T.size - conv.size
        new_conv = np.pad(conv, ((0,pad)), 'constant')
        return new_conv

def conv_poly(P):
    """
    Convert a standard polynomial to a chebyshev polynomial in one dimension.

    Args:
        P (): The standard polynomial to be converted.

    Returns:
        new_conv (): The chebyshev polynomial.

    """
    conv = cheb.poly2cheb(P)
    if conv.size == P.size:
        return conv
    else:
        pad = P.size - conv.size
        new_conv = np.pad(conv, ((0,pad)), 'constant')
        return new_conv

def cheb2poly(T):
    """
    Convert a chebyshev polynomial to a standard polynomial in multiple dimensions.

    """
    dim = len(T.shape)
    A = T.coeff
    for i in range(dim):
        A = np.apply_along_axis(conv_cheb, i, A)
    return MultiPower(A)

def poly2cheb(P):
    """
    Convert a standard polynomial to a chebyshev polynomial in multiple dimensions.
    
    Args:
        P (): The multi-dimensional standard polynomial. (tensor?)

    Returns:
        (MultiCheb): The multi-dimensional chebyshev polynomial.

    """
    dim = len(P.shape)
    A = P.coeff
    for i in range(dim):
        A = np.apply_along_axis(conv_poly, i, A)
    return MultiCheb(A)
