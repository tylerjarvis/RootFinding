# A collection of functions used in the F4 and Macaulay solvers
import numpy as np
from scipy.linalg import lu, qr, solve_triangular
import heapq

class Term(object):
    '''
    Terms are just tuples of exponents with the grevlex ordering
    '''
    def __init__(self,val):
        self.val = tuple(val)

    def __repr__(self):
        return str(self.val) + ' with grevlex order'

    def __lt__(self, other, order = 'grevlex'):
        '''
        Redfine less-than according to grevlex
        '''
        if order == 'grevlex': #Graded Reverse Lexographical Order
            if sum(self.val) < sum(other.val):
                return True
            elif sum(self.val) > sum(other.val):
                return False
            else:
                for i,j in zip(reversed(self.val),reversed(other.val)):
                    if i < j:
                        return False
                    if i > j:
                        return True
                return False
        elif order == 'lexographic': #Lexographical Order
            for i,j in zip(self.val,other.val):
                if i < j:
                    return True
                if i > j:
                    return False
            return False
        elif order == 'grlex': #Graded Lexographical Order
            if sum(self.val) < sum(other.val):
                return True
            elif sum(self.val) > sum(other.val):
                return False
            else:
                for i,j in zip(self.val,other.val):
                    if i < j:
                        return True
                    if i > j:
                        return False
                return False
        elif order == 'eriks': #Make single variable ones higher for Macaualy
            if sum(self.val) < sum(other.val):
                return True
            elif sum(self.val) > sum(other.val):
                return False
            selfSingle = (len(np.where(np.array(self.val) != 0)[0]) == 1)
            otherSingle = (len(np.where(np.array(other.val) != 0)[0]) == 1)
            if otherSingle and not selfSingle:
                return True
            elif selfSingle and not otherSingle:
                return False
            else:
                for i,j in zip(self.val,other.val):
                    if i < j:
                        return True
                    if i > j:
                        return False
                return False

    # Define the other relations in grevlex order

    def __eq__(self, other):
        return self.val == other.val

    def __gt__(self, other):
        return not(self < other or self == other)

    def __ge__(self, other):
        return (self > other or self == other)

    def __le__(self,other):
        return (self < other or self == other)

    #Makes terms hashable so they can go in a set
    def __hash__(self):
        return hash(self.val)

class Term_w_InvertedOrder(Term):
    '''
    Called by MaxHeap object to reverse the ordering for a min heap
    Used exclusively with Terms
    '''
    def __init__(self,term):
        '''
        Takes in a Term.  val is the underlying tuple, term is the underlying term
        '''
        self.val = term.val
        self.term = term

    # Invert the order

    def __lt__(self,other): return self.term > other.term
    def __le__(self,other): return (self.term > other.term or self.term == other.term)
    def __ge__(self,other): return (self.term < other.term or self.term == other.term)
    def __gt__(self,other): return (self.term < other.term)

    def __repr__(self):
        return str(list(self.val)) + ' with inverted grevlex order'


class MaxHeap(object):
    '''
    Implementation of a set max-priority queue--one that only adds
    terms to the queue if they aren't there already

    Incoming and outgoing objects are all Terms (not Term_w_InvertedOrder)
    '''

    def __init__(self):
        self.h = []         # empty heap
        self._set = set()   # empty set (of things already in the heap)

    def heappush(self, x):
        if not x.val in self._set:       # check if already in the set
            x = Term_w_InvertedOrder(x)
            heapq.heappush(self.h,x)     # push with InvertedOrder
            self._set.add(x.val)         # but use the tuple in the set (it is easily hashable)

    def heappop(self):
        term = heapq.heappop(self.h).term   # only keep the original term--without the InvertedOrder
        self._set.discard(term.val)
        return term

    def __getitem__(self, i):
        return self.h[i].term

    def __len__(self):
        return len(self.h)

    def __repr__(self):
        return('A max heap of {} unique terms with the DegRevLex term order.'.format(len(self)))

class MinHeap(MaxHeap):
    '''
    Implementation of a set min-priorioty queue.

    '''

    def heappush(self,x):
        ## Same as MaxHeap push, except that the term order is not inverted
        if not x in self._set:
            heapq.heappush(self.h, x)
            self._set.add(x)
        else:
            pass

    def heappop(self):
        """ Same as MaxHeap pop except that the term itself IS the underlying term.
        """
        term = heapq.heappop(self.h)
        self._set.discard(term.val)
        return term

    def __getitem__(self, i):
        """ Same as MaxHeap getitem except that the term itself IS the underlying term.
        """
        return self.h[i]

    def __repr__(self):
        return('A min heap of {} unique terms with the DegRevLex term order.'.format(len(self)))


def divides(mon1, mon2):
    '''
    parameters
    ----------
    mon1 : tuple
        contains the exponents of the monomial divisor
    mon2 : tuple
        contains the exponents of the monomial dividend

    returns
    -------
    boolean
        true if mon1 divides mon2, false otherwise
    '''
    return all(np.subtract(mon2, mon1) >= 0)

def sorted_polys_monomial(polys):
    '''
    Sorts the polynomials by the number of monomials they have, the ones with the least amount first.

    polys (list-like): !!! Is it a list or could it be any iterable?

    '''

    # A list to contain the number of monomials with non zero coefficients.
    num_monomials = []
    for p in polys:
        # Grab all the coefficients in the tensor that are non zero.
        # !!! Why are we only getting index 0?
        n = len(np.where(p.coeff != 0)[0])
        num_monomials.append(n)

    # Generate a sorted index based on num_monomials.
    # TODO: I'm pretty sure there is a numpy function that does this already.
    argsort_list = sorted(range(len(num_monomials)), key=num_monomials.__getitem__)[::]

    # Sort the polynomials according to the index argsort_list.
    sorted_polys = [polys(i) for i in argsort_list]

    return sorted_polys


def argsort(index_list):
    '''
    Returns an argsort list for the index, as well as sorts the list in place

    !!! This could be combined with sorted_polys_monomial to avoid repetitive code.

    '''

    argsort_list = sorted(range(len(index_list)), key=index_list.__getitem__)[::-1]
    return argsort_list, index_list.sort()[::-1]


def calc_r(m, sorted_polys):
    '''
    Finds the r polynomial that has a leading monomial m.
    Returns the polynomial.

    '''

    for p in sorted_polys:
        l = list(p.lead_term)
        # Check to see if l divides m
        if all([i<=j for i,j in zip(l,m)]) and len(l) == len(m):
            # !!! i-j is used in the divide() method, not j-i. Is this a problem?
            c = [j-i for i,j in zip(l,m)]
            if l != m: # Make sure c isn't all 0
                return p.mon_mult(c)


def row_swap_matrix(matrix):
    '''
    rearange the rows of matrix so it starts close to upper traingular

    '''

    rows, columns = np.where(matrix != 0)
    last_i = -1
    lms = list() # !!! What does lms stand for?
    for i,j in zip(rows,columns):
        if i != last_i:
            lms.append(j)
            last_i = i
    # !!! Repetition of previous code.
    argsort_list = sorted(range(len(lms)), key=lms.__getitem__)[::]
    return matrix[argsort_list]


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

def get_var_list(dim):
    '''Returns a list of the variables [x_1, x_2, ..., x_n] as tuples.'''
    _vars = []
    for i in range(dim):
        var = np.zeros(dim, dtype=int)
        var[i] = 1
        _vars.append(tuple(var))
    return _vars

def clean_zeros_from_matrix(matrix, global_accuracy=1.e-10):
    '''
    Sets all points in the matrix less than the gloabal accuracy to 0.

    '''
    matrix[np.where(np.abs(matrix) < global_accuracy)] = 0
    return matrix

def fullRank(matrix, global_accuracy = 1.e-10):
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
    pass

def inverse_P(p):
    '''
    Takes in the one dimentional array of column switching.
    Returns the one dimentional array of switching it back.

    '''
    # The elementry matrix that flips the columns of given matrix.
    P = np.eye(len(p))[:,p]
    # This finds the index that equals 1 of each row of P.
    #(This is what we want since we want the index of 1 at each column of P.T)
    return np.where(P==1)[1]


def triangular_solve(matrix):
    """
    Reduces the upper block triangular matrix.
    """
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
        order = inverse_P(order_c+order_d)

        # Reverse the columns back.
        solver = solver[:,order]
        # Temporary checker. Plots the non-zero part of the matrix.
        #plt.matshow(~np.isclose(solver,0))

        return solver

    else:
    # The case where the matrix passed in is a square matrix
        return np.eye(m)
