# A collection of functions used in the F4 and Macaulay solvers
import numpy as np
from scipy.linalg import lu, qr, solve_triangular
import heapq

class InstabilityWarning(Warning):
    pass

class TVBError(RuntimeError):
    pass

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
        ''' Same as MaxHeap pop except that the term itself IS the underlying term.
        '''
        term = heapq.heappop(self.h)
        self._set.discard(term.val)
        return term

    def __getitem__(self, i):
        ''' Same as MaxHeap getitem except that the term itself IS the underlying term.
        '''
        return self.h[i]

    def __repr__(self):
        return('A min heap of {} unique terms with the DegRevLex term order.'.format(len(self)))

def argsort_dec(list_):
    '''Sort the given list into decreasing order.

    Parameters
    ----------
    list_ : list
        The list to be sorted.

    Returns
    -------
    argsort_list : list
        A list of the old indexes in their new places. For example, if
        [3,1,4] was sorted to be [4,3,1], then argsort_list would be [2,0,1]
    list_ : list
        The same list as was input, but now in decreasing order.

    '''

    argsort_list = sorted(range(len(list_)), key=list_.__getitem__)[::-1]
    list_.sort()
    return argsort_list, list_[::-1]

def argsort_inc(list_):
    '''Sort the given list into increasing order.

    Parameters
    ----------
    list_ : list
        The list to be sorted.

    Returns
    -------
    argsort_list : list
        A list of the old indexes in their new places. For example, if
        [3,1,4] was sorted to be [1,3,4], then argsort_list would be [1,0,2]
    list_ : list
        The same list as was input, but now in increasing order.

    '''

    argsort_list = sorted(range(len(list_)), key=list_.__getitem__)
    list_.sort()
    return argsort_list, list_

def clean_matrix(matrix, matrix_terms, set_zeros=False, accuracy=1.e-10):
    '''Removes columns in the matrix that are all zero along with associated
    terms in matrix_terms.

    Parameters
    ----------
    matrix : 2D numpy array
        The matrix with rows corresponding to polynomials, columns corresponding
        to monomials, and entries corresponding to coefficients.
    matrix_terms : array-like, contains Term objects
        The column labels for matrix in order.
    set_zeros : bool, optional
        If true, all entries in the matrix that are within accuracy of 0 will
        be set to 0.
    accuracy : float, optional
        How close entries should be to 0 for them to be set to 0 (only applies
        if set_zeros is True).

    Returns
    -------
    matrix : 2D numpy array
        Same matrix as input but with all 0 columns removed.
    matrix_terms : array-like, contains Term objects
        Same as input but with entries corresponding to 0 columns in the matrix
        removed.

    '''

    ##This would replace all small values in the matrix with 0.
    if set_zeros:
        matrix = clean_zeros_from_matrix(matrix, accuracy=accuracy)

    #Removes all 0 monomials
    non_zero_monomial = np.sum(abs(matrix), axis=0) != 0
    matrix = matrix[:,non_zero_monomial] #only keeps the non_zero_monomials
    matrix_terms = matrix_terms[non_zero_monomial]

    return matrix, matrix_terms

def clean_zeros_from_matrix(array, accuracy=1.e-10):
    '''Sets all values in the array less than the given accuracy to 0.

    Parameters
    ----------
    array : numpy array
    accuracy : float, optional
        Values in the matrix less than this will be set to 0.

    Returns
    -------
    array : numpy array
        Same array, but with values less than the given accuracy set to 0.

    '''
    array[(array < accuracy) & (array > -accuracy)] = 0
    return array

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

def inverse_P(P):
    '''The inverse of P, the array with column switching indexes.

    Parameters
    ----------
    P : array-like
        1D array P returned by scipy's QRP decomposition.

    Returns
    -------
    1D numpy array
        The indexes needed to switch the columns back to their original
        positions.

    See Also
    --------
    scipy.linalg.qr : QR decomposition (with pivoting=True).

    '''

    # The elementry matrix that flips the columns of given matrix.
    M_P= np.eye(len(P))[:,P]
    # This finds the index that equals 1 of each row of P.
    #(This is what we want since we want the index of 1 at each column of P.T)
    return np.where(M_P==1)[1]

def lcm(a,b):
    '''Finds the LCM of the two leading terms of polynomials a and b

    Parameters
    ----------
    a, b : polynomial objects

    Returns
    -------
    numpy array
        The lcm of the leading terms of a and b. The usual representation is
        used, i.e., :math:`x^2y^3` is represented as :math:`\mathtt{(2,3)}`

    '''
    return np.maximum(a.lead_term, b.lead_term)

def quotient(a, b):
    '''Finds the quotient of monomials a and b, that is, a / b.

    Parameters
    ----------
    a, b : array-like, the monomials to divide

    Returns
    -------
    list
        The quotient a / b

    '''

    return [i-j for i,j in zip(a, b)]

def rrqr_reduce(matrix, clean = False, global_accuracy = 1.e-10):
    '''
    Reduces the matrix into row echelon form, so each row has a unique leading term.

    Parameters
    ----------
    matrix : (2D numpy array)
        The matrix of interest.
    clean: bool
        Defaults to False. If True then at certain points in the code all the points in the matrix
        that are close to 0 are set to 0.
    global_accuracy: float
        Defaults to 1.e-10. What is determined to be zero when searching for the pivot columns or setting
        things to zero.

    Returns
    -------
    matrix : (2D numpy array)
        The reduced matrix in row echelon form. It should look like this.
        a - - - - - - -
        0 b - - - - - -
        0 0 0 c - - - -
        0 0 0 0 d - - -
        0 0 0 0 0 0 0 e
    '''
    if matrix.shape[0]==0 or matrix.shape[1]==0:
        return matrix
    height = matrix.shape[0]
    A = matrix[:height,:height] #Get the square submatrix
    B = matrix[:,height:] #The rest of the matrix to the right
    Q,R,P = qr(A, pivoting = True) #rrqr reduce it
    PT = inverse_P(P)
    diagonals = np.diagonal(R) #Go along the diagonals to find the rank
    rank = np.sum(np.abs(diagonals)>global_accuracy)
    if rank == height: #full rank, do qr on it
        Q,R = qr(A)
        A = R #qr reduce A
        B = Q.T.dot(B) #Transform B the same way
    else: #not full rank
        A = R[:,PT] #Switch the columns back
        if clean:
            Q[np.where(abs(Q) < global_accuracy)]=0
        B = Q.T.dot(B) #Multiply B by Q transpose
        if clean:
            B[np.where(abs(B) < global_accuracy)]=0
        #sub1 is the top part of the matrix, we will recursively reduce this
        #sub2 is the bottom part of A, we will set this all to 0
        #sub3 is the bottom part of B, we will recursively reduce this.
        #All submatrices are then put back in the matrix and it is returned.
        sub1 = np.hstack((A[:rank,],B[:rank,])) #Takes the top parts of A and B
        result = rrqr_reduce(sub1) #Reduces it
        A[:rank,] = result[:,:height] #Puts the A part back in A
        B[:rank,] = result[:,height:] #And the B part back in B

        sub2 = A[rank:,]
        zeros = np.zeros_like(sub2)
        A[rank:,] = np.zeros_like(sub2)

        sub3 = B[rank:,]
        B[rank:,] = rrqr_reduce(sub3)

    reduced_matrix = np.hstack((A,B))
    return reduced_matrix

def rrqr_reduce2(matrix, clean = True, global_accuracy = 1.e-10):
    '''
    Reduces the matrix into row echelon form, so each row has a unique leading term.
    Note that it preforms the same function as rrqr_reduce, currently I'm not sure which is better.

    Parameters
    ----------
    matrix : (2D numpy array)
        The matrix of interest.
    clean: bool
        Defaults to True. If True then at certain points in the code all the points in the matrix
        that are close to 0 are set to 0.
    global_accuracy: float
        Defaults to 1.e-10. What is determined to be zero when searching for the pivot columns or setting
        things to zero.

    Returns
    -------
    matrix : (2D numpy array)
        The reduced matrix in row echelon form. It should look like this.
        a - - - - - - -
        0 b - - - - - -
        0 0 0 c - - - -
        0 0 0 0 d - - -
        0 0 0 0 0 0 0 e
    '''
    if matrix.shape[0] <= 1 or matrix.shape[0]==1 or  matrix.shape[1]==0:
        return matrix
    height = matrix.shape[0]
    A = matrix[:height,:height] #Get the square submatrix
    B = matrix[:,height:] #The rest of the matrix to the right
    independentRows, dependentRows, Q = row_linear_dependencies(A, accuracy = global_accuracy)
    nullSpaceSize = len(dependentRows)
    if nullSpaceSize == 0: #A is full rank
        Q,R = qr(matrix)
        return clean_zeros_from_matrix(R)
    else: #A is not full rank
        #sub1 is the independentRows of the matrix, we will recursively reduce this
        #sub2 is the dependentRows of A, we will set this all to 0
        #sub3 is the dependentRows of Q.T@B, we will recursively reduce this.
        #We then return sub1 stacked on top of sub2+sub3
        if clean:
            Q[np.where(abs(Q) < global_accuracy)]=0
        bottom = matrix[dependentRows]
        BCopy = B.copy()
        sub3 = bottom[:,height:]
        sub3 = Q.T[-nullSpaceSize:]@BCopy
        if clean:
            sub3 = clean_zeros_from_matrix(sub3)
        sub3 = rrqr_reduce2(sub3)

        sub1 = matrix[independentRows]
        sub1 = rrqr_reduce2(sub1)

        sub2 = bottom[:,:height]
        sub2[:] = np.zeros_like(sub2)

        reduced_matrix = np.vstack((sub1,np.hstack((sub2,sub3))))
        if clean:
            return clean_zeros_from_matrix(reduced_matrix)
        else:
            return reduced_matrix

def sorted_polys_coeff(polys):
    '''Sorts the polynomials by how much bigger the leading coefficient is than
    the rest of the coeff matrix.

    Parameters
    ----------
    polys : array-like
        Contains polynomial objects to sort.

    Returns
    -------
    sorted_polys : list
        The polynomial objects in order of lead coefficient to everything else
        ratio.

    '''

    # The lead_coeff to other stuff ratio.
    lead_coeffs = [abs(poly.lead_coeff)/np.sum(np.abs(poly.coeff)) for poly in polys]

    argsort_list = argsort_dec(lead_coeffs)[0]
    sorted_polys = [polys[i] for i in argsort_list]

    return sorted_polys

def sorted_polys_monomial(polys):
    '''Sorts the polynomials by the number of monomials they have, the ones
    with the least amount first.

    Parameters
    ----------
    polys : array-like, contains polynomial objects !!! Is it a list or could it be any iterable?
        Polynomials to be sorted

    Returns
    -------
    sorted_polys : list
        Polynomials in order.

    '''

    # A list to contain the number of monomials with non zero coefficients.
    num_monomials = []
    for poly in polys:
        # This gets the length of the list of first indexes, since
        # that is number of non-zero coefficients in the coefficient array.
        # See documentation for np.where
        num_monomials.append(len(np.where(poly.coeff != 0)[0]))

    # Generate a sorted index based on num_monomials.
    # TODO: I'm pretty sure there is a numpy function that does this already.
    argsort_list = argsort_inc(num_monomials)[0]

    # Sort the polynomials according to the index argsort_list.
    sorted_polys = [polys[i] for i in argsort_list]

    return sorted_polys

def row_swap_matrix(matrix):
    '''Rearrange the rows of matrix so it is close to upper traingular.

    Parameters
    ----------
    matrix : 2D numpy array
        The matrix whose rows need to be switched

    Returns
    -------
    2D numpy array
        The same matrix but with the rows changed so it is close to upper
        triangular

    Examples
    --------
    >>> utils.row_swap_matrix(np.array([[0,2,0,2],[0,1,3,0],[1,2,3,4]]))
    array([[1, 2, 3, 4],
           [0, 2, 0, 2],
           [0, 1, 3, 0]])

    '''

    rows, columns = np.where(matrix != 0)
    last_i = -1
    leading_mon_columns = list()
    for i,j in zip(rows,columns):
        if i != last_i:
            leading_mon_columns.append(j)
            last_i = i

    argsort_list = argsort_inc(leading_mon_columns)[0]
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

def row_linear_dependencies(matrix, accuracy=1.e-10):
    '''
    Uses rank revealing QR to determine which rows of the given matrix are
    linearly independent and which ones are linearly dependent. (This
    function needs a name change).

    Parameters
    ----------
    matrix : (2D numpy array)
        The matrix of interest.

    Returns
    -------
    independentRows : (list)
        The indexes of the rows that are linearly independent
    dependentRows : (list)
        The indexes of the rows that can be removed without affecting the rank
        (which are the linearly dependent rows).
    Q : (2D numpy array)
        The Q matrix used in RRQR reduction in finding the rank.
    '''

    height = matrix.shape[0]
    Q,R,P = qr(matrix, pivoting = True)
    diagonals = np.diagonal(R) #Go along the diagonals to find the rank
    rank = np.sum(np.abs(diagonals)>accuracy)
    numMissing = height - rank
    if numMissing == 0: # Full Rank. All rows independent
        return [i for i in range(height)],[],None
    else:
        # Find the rows we can take out. These are ones that are non-zero in
        # the last rows of Q transpose, since QT*A=R.
        # To find multiple, we find the pivot columns of Q.T
        QMatrix = Q.T[-numMissing:]
        Q1,R1,P1 = qr(QMatrix, pivoting = True)
        independentRows = P1[R1.shape[0]:] #Other Columns
        dependentRows = P1[:R1.shape[0]] #Pivot Columns
        return independentRows, dependentRows, Q

def get_var_list(dim):
    _vars = [] # list of the variables: [x_1, x_2, ..., x_n]
    for i in range(dim):
        var = np.zeros(dim, dtype=int)
        var[i] = 1
        _vars.append(tuple(var))
    return _vars

def triangular_solve(matrix, matrix_terms = None, reorder = True):
    """
    Takes a matrix that is in row echelon form and reduces it into row reduced echelon form.

    Parameters
    ----------
    matrix : 2D numpy array
        The matrix of interest.
    matrix_terms : An numpy array.
        The i'th row matrix_terms is the term in the i'th column of the matrix.
    reorder : bool
        If reorder is True then the matrix is reordered after triangular solve to put it in it's
        initial order. Otherwise it is returned so the first part of the matrix is the identity matrix.
        The matrix_terms are reordered accordingly.

    Returns
    -------
    matrix : 2D numpy array
        The matrix is row reduced echelon form if reorder it True, ordered with the pivot columns in
        the fron otherwise.

    Optional Return
    ---------------
    matrix_terms : An numpy array.
        Only returned if reorder is False. The reordered matrix_terms. If reorder is True they will
        have not been affected.
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
        #order = inverse_P(order_c+order_d)

        # Reverse the columns back.
        if reorder:
            solver1 = np.empty_like(solver)
            solver1[:,order_c+order_d] = solver
            #solver = solver[:,order]
            return solver1
        else:
            matrix_terms = matrix_terms[order_c+order_d]
            return solver, matrix_terms

    else:
    # The case where the matrix passed in is a square matrix
        return np.eye(m)

def first_x(string):
    '''
    Finds the first position of an 'x' in a string. If there is not x it returns the length
    of the string.

    Parameters
    ----------
    string : str
        The string of interest.
    Returns
    -------
    i : int
        The position in the string of the first 'x' character. If 'x' does not appear in the string
        the return value is the length of the string.

    '''
    for i in range(len(string)):
        if string[i] == 'x':
            return i
    return len(string)

def is_number(string):
    '''
    Checks is a string can be converted to a number.
    Parameters
    ----------
    string : str
        The string of interest.
    Returns
    -------
    value : bool
        Whether or not the string is a valid number.
    '''
    try:
        float(string)
        return True
    except ValueError:
        return False

def makePolyCoeffMatrix(inputString):
    '''
    Takes a string input of a polynomaial and returns the coefficient matrix for it. Usefull for making things of high
    degree of dimension so you don't have to make it by hand.

    All strings must be of the following syntax. Ex. '3x0^2+2.1x1^2*x2+-14.73x0*x2^3'

    1. There can be no spaces.
    2. All monomials must be seperated by a '+'. If the coefficient of the monomial is negative then the '-' sign
       should come after the '+'. This is not needed for the first monomial.
    3. All variables inside a monomial are seperated by a '*'.
    4. The power of a variable in a monomial is given folowing a '^' sign.
    '''
    matrixSpots = list()
    coefficients = list()
    for monomial in inputString.split('+'):
        coefficientString = monomial[:first_x(monomial)]
        if coefficientString == '-':
            coefficient = -1
        elif coefficientString == '':
            coefficient = 1
        else:
            coefficient = float(coefficientString)
        mons = monomial[first_x(monomial):].split('*')
        matrixSpot = np.zeros(1, dtype = int)
        for mon in mons:
            stuff = mon.split('^')
            if len(stuff) == 1:
                power = 1
            else:
                power = int(stuff[1])
            if stuff[0] == '':
                varDegree = -1
            else:
                varDegree = int(stuff[0][1:])
            if varDegree != -1:
                if len(matrixSpot) <= varDegree:
                    matrixSpot = np.append(matrixSpot, np.zeros(varDegree - len(matrixSpot)+1, dtype = int))
                matrixSpot[varDegree] = power
        matrixSpots.append(matrixSpot)
        coefficients.append(coefficient)

    #Pad the matrix spots so they are all the same length.
    length = max(len(matrixSpot) for matrixSpot in matrixSpots)
    for i in range(len(matrixSpots)):
        matrixSpot = matrixSpots[i]
        if len(matrixSpot) < length:
            matrixSpot = np.append(matrixSpot, np.zeros(length - len(matrixSpot), dtype = int))
            matrixSpots[i] = matrixSpot
    matrixSize = np.maximum.reduce([matrixSpot for matrixSpot in matrixSpots])
    matrixSize = matrixSize + np.ones_like(matrixSize)
    matrix = np.zeros(matrixSize)
    for i in range(len(matrixSpots)):
        matrixSpot = matrixSpots[i]
        coefficient = coefficients[i]
        matrix[tuple(matrixSpot)] = coefficient
    return matrix

def sort_matrix(matrix, matrix_terms):
    '''Sort matrix columns by some term order.

    Sorts the matrix columns into whichever order is defined on the term objects
    in matrix_terms (usually grevlex).

    Parameters
    ----------
    matrix : 2D numpy array
        The matrix with rows corresponding to polynomials, columns
        corresponding to monomials, and each entry is a coefficient.
    matrix_terms : array-like, contains Term objects
        Contains the monomial labels for the matrix columns in order, i.e., if
        the first column of matrix corresponds to the monomial x^2, then
        matrix_terms[0] is Term(x^2).

    Returns
    -------
    ordered_matrix : 2D numpy array
        The same matrix as was input, but now with the columns switched so they
        are in order.
    matrix_terms : array-like, contains Term objects
        Same as the input, but now ordered.

    '''

    argsort_list, matrix_terms = argsort_dec(matrix_terms)
    ordered_matrix = matrix[:,argsort_list]
    return ordered_matrix, matrix_terms
