import numpy as np
from scipy.signal import fftconvolve, convolve
import itertools
from groebner.polynomial import Polynomial
from numpy.polynomial import chebyshev as cheb

'''
08/31/17
Author: Rex McArthur
Creates a class of n-dim chebyshev polynomials. Tracks leading term,
coefficents, and inculdes basic operations (+,*,scalar multip, etc.)
Assumes GRevLex ordering, but should be extended.
'''


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

    def __init__(self, coeff, order='degrevlex', lead_term=None, clean_zeros = True):
        super(MultiCheb, self).__init__(coeff, order, lead_term, clean_zeros)


    def __add__(self,other):
        '''
        Here we add an addition method
        '''
        if self.shape != other.shape:
            new_self, new_other = self.match_size(self,other)
        else:
            new_self, new_other = self, other

        return MultiCheb(new_self.coeff + new_other.coeff)

    def __sub__(self,other):
        '''
        Here we subtract the two polys coeffs
        '''
        if self.shape != other.shape:
            new_self, new_other = self.match_size(self,other)
        else:
            new_self, new_other = self, other
        return MultiCheb(new_self.coeff - new_other.coeff)

    def _reverse_axes(self):
        """
        Reverse the axes of the coeff tensor.
        """
        return self.coeff.flatten()[::-1].reshape(self.coeff.shape)

    def match_size(self,a,b):
        '''
        Matches the shape of the polynomials
        '''

        A_shape, B_shape = list(a.shape), list(b.shape)
        A, B = a.coeff, b.coeff
        if len(A_shape) != len(B_shape):
            add_to_shape = 0
            if len(A_shape) < len(B_shape):
                add_to_shape = len(B_shape) - len(A_shape)
                for i in range(add_to_shape):
                    A_shape.insert(0,1)
                a = A.reshape(A_shape)
                a = MultiCheb(a)
            else:
                add_to_shape = len(A_shape) - len(B_shape)
                for i in range(add_to_shape):
                    B_shape.insert(0,1)
                b = B.reshape(B_shape)
                b = MultiCheb(b)

        new_shape = [max(i,j) for i,j in itertools.zip_longest(a.shape, b.shape, fillvalue = 0)] #finds the largest length in each dimmension
        # finds the difference between the largest length and the original shapes in each dimmension.
        add_a = [i-j for i,j in itertools.zip_longest(new_shape, a.shape, fillvalue = 0)]
        add_b = [i-j for i,j in itertools.zip_longest(new_shape, b.shape, fillvalue = 0)]
        #create 2 matrices with the number of rows equal to number of dimmensions and 2 columns
        add_a_list = np.zeros((len(new_shape),2))
        add_b_list = np.zeros((len(new_shape),2))
        #changes the second column to the values of add_a and add_b.
        add_a_list[:,1] = add_a
        add_b_list[:,1] = add_b
        #uses add_a_list and add_b_list to pad each polynomial appropriately.
        a = MultiCheb(np.pad(a.coeff,add_a_list.astype(int),'constant'), clean_zeros = False)
        b = MultiCheb(np.pad(b.coeff,add_b_list.astype(int),'constant'), clean_zeros = False)
        return a,b

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

    def __mul__(self,other):
        '''
        Multiply by convolving intelligently
        Formula T_n(x)T_m(x) = 1/2[T_n+m(x)+T_|n-m|(x)]
        p1 = T_n+m(x)
            Found by convolving the orginal polynomials.
        p2 = T_|n-m|(x)
            Found in 3 steps
                1. Reverse the order of one polynomial.
                2. Convolve reversed polynomial and remaining polynomial.
                3. Fold around middle axis in each dimension.
        '''
        # Check and see if same size
        if self.shape != other.shape:
            new_self, new_other = self.match_size(self,other)
        else:
            new_self, new_other = self, other

        p1 = MultiCheb(convolve(new_self.coeff,new_other.coeff)) #p1 is found by convolving the original polynomials.
        c = new_other._reverse_axes() #reverses order of polynomial.
        p2 = convolve(new_self.coeff,c) #p2 is found by convolving new_self and c.
        shape_of_p2 = p2.shape #the shape is used to find the axis to fold around in each direction.
        dim = p2.ndim #The dimension is needed for array slicing.
        for i in range(dim): #Loop goes through each dimension and folds polynomial in that direction.
            half = shape_of_p2[i]//2 #Take the length of each dimension and find the middle index in that dimmension
            p2 = MultiCheb.fold_for_reg_mult(p2, half, i, dim) #Pass values into function for folding.

        p2 = MultiCheb(p2)
        Pf = (p1+p2)
        return MultiCheb((.5*Pf.coeff), clean_zeros = True)
        #TODO: You can use the lead_term kwarg to save some time

    def fold_in_i_dir(solution_matrix, dim, i, x, fold_idx):
        """
        Folds around a fold_inx and returns new solution.
        solution_matrix is polynomial to be folded
        dim is the number of dimensions of solution_matrix
        i represents the dimensions being folded.
        x is the size of the solution matrix in the dimension being folded
        fold_idx is the index to fold around.
        """
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

    def mon_mult(self, idx):
        for i in range(len(idx)):
            idx_zeros = np.zeros(len(idx),dtype = int)
            idx_zeros[i] = idx[i]
            self = self.mon_mult1(idx_zeros)
        return self

    def mon_mult1(self,idx):
        """
        Takes a polynomial and the index of a monomial and returns the result of the multiplication.
        """
        #This is the cheating convert to power way.
        #power = cheb2poly(self)
        #mult = power.mon_mult(idx)
        #return poly2cheb(mult)
        
        
        
        
        pad_values = list()
        for i in idx: #iterates through monomial and creates a tuple of pad values for each dimension
            pad_dim_i = (i,0)
            #In np.pad each dimension is a tuple of (i,j) where i is how many to pad in front and j is how many to pad after.
            pad_values.append(pad_dim_i)
        p1 = MultiCheb(np.pad(self.coeff, (pad_values), 'constant', constant_values = 0))

        solution_matrix = self.coeff

        largest_idx = [i-1 for i in solution_matrix.shape]
        new_shape = [max(i,j) for i,j in itertools.zip_longest(largest_idx, idx, fillvalue = 0)] #finds the largest length in each dimmension
        add_a = [i-j for i,j in itertools.zip_longest(new_shape, largest_idx, fillvalue = 0)]
        add_a_list = np.zeros((len(new_shape),2))
        #changes the second column to the values of add_a and add_b.
        add_a_list[:,1] = add_a
        #uses add_a_list and add_b_list to pad each polynomial appropriately.
        solution_matrix = np.pad(solution_matrix,add_a_list.astype(int),'constant')

        number_of_dim = solution_matrix.ndim
        shape_of_self = solution_matrix.shape

        #Loop iterates through each dimension of the polynomial and folds in that dimension
        for i in range(number_of_dim):
            solution_matrix = MultiCheb.fold_in_i_dir(solution_matrix, number_of_dim, i, shape_of_self[i], idx[i])

        p2 = MultiCheb(solution_matrix)
        Pf = (p1+p2)
        return MultiCheb(.5*Pf.coeff) #Make

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
