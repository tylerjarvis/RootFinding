import numpy as np
import itertools
from scipy.linalg import qr, solve_triangular, qr_multiply
from numalgsolve.polynomial import Polynomial, MultiCheb, MultiPower
from numalgsolve.utils import row_swap_matrix, MacaulayError, slice_top, mon_combos, \
                              num_mons_full, memoized_all_permutations, mons_ordered, \
                              all_permutations_cheb

def add_polys(degree, poly, poly_coeff_list):
    """Adds polynomials to a Macaulay Matrix.

    This function is called on one polynomial and adds all monomial multiples of
     it to the matrix.

    Parameters
    ----------
    degree : int
        The degree of the Macaulay Matrix
    poly : Polynomial
        One of the polynomials used to make the matrix.
    poly_coeff_list : list
        A list of all the current polynomials in the matrix.
    Returns
    -------
    poly_coeff_list : list
        The original list of polynomials in the matrix with the new monomial
        multiplications of poly added.
    """

    poly_coeff_list.append(poly.coeff)
    deg = degree - poly.degree
    dim = poly.dim

    mons = mon_combos([0]*dim,deg)

    for mon in mons[1:]: #skips the first all 0 mon
        poly_coeff_list.append(poly.mon_mult(mon, returnType = 'Matrix'))
    return poly_coeff_list

def find_degree(poly_list, verbose=False):
    '''Finds the appropriate degree for the Macaulay Matrix.

    Parameters
    --------
    poly_list: list
        The polynomials used to construct the matrix.

    Returns
    -----------
    find_degree : int
        The degree of the Macaulay Matrix.

    '''
    if verbose:
        print('Degree of Macaulay Matrix:', sum(poly.degree for poly in poly_list) - len(poly_list) + 1)
    return sum(poly.degree for poly in poly_list) - len(poly_list) + 1

def rrqr_reduceMacaulay(matrix, matrix_terms, cuts, number_of_roots, accuracy = 1.e-10):
    ''' Reduces a Macaulay matrix, BYU style.

    The matrix is split into the shape
    A B C
    D E F
    Where A is square and contains all the highest terms, and C contains all the x,y,z etc. terms. The lengths
    are determined by the matrix_shape_stuff tuple. First A and D are reduced using rrqr without pivoting, and then the rest of
    the matrix is multiplied by Q.T to change it accordingly. Then E is reduced by rrqr with pivoting, the rows of B are shifted
    accordingly, and F is multipled by Q.T to change it accordingly. This is all done in place to save memory.

    Parameters
    ----------
    matrix : numpy array.
        The Macaulay matrix, sorted in BYU style.
    matrix_terms: numpy array
        Each row of the array contains a term in the matrix. The i'th row corresponds to
        the i'th column in the matrix.
    cuts : tuple
        When the matrix is reduced it is split into 3 parts with restricted pivoting. These numbers indicate
        where those cuts happen.
    Returns
    -------
    matrix : numpy array
        The reduced matrix.
    matrix_terms: numpy array
        The resorted matrix_terms.
    '''    
    #print("Starting matrix.shape:\n", matrix.shape)
    #RRQR reduces A and D without pivoting sticking the result in it's place.
    Q1,matrix[:,:cuts[0]] = qr(matrix[:,:cuts[0]])
    
    #check if there are zeros along the diagonal of R1
    if any(np.isclose(np.diag(matrix[:,:cuts[0]]),0, atol=accuracy)):
        raise MacaulayError("R1 IS NOT FULL RANK")

    #Looks like 0 but not, add to the rank.
    #still_good = np.sum(np.abs(matrix[:,:cuts[0]].diagonal()) < accuracy)
    #if abs(matrix[:,:cuts[0]].diagonal()[-1]) < accuracy:
    #    print(matrix[:,:cuts[0]].diagonal())
    #    raise MacaulayError("HIGHEST NOT FULL RANK")

    #Multiplying the rest of the matrix by Q.T
    matrix[:,cuts[0]:] = Q1.T@matrix[:,cuts[0]:]
    Q1 = 0 #Get rid of Q1 for memory purposes.

    #RRQR reduces E sticking the result in it's place.
    Q,matrix[cuts[0]:,cuts[0]:cuts[1]],P = qr(matrix[cuts[0]:,cuts[0]:cuts[1]], pivoting = True)

    #Multiplies F by Q.T.
    matrix[cuts[0]:,cuts[1]:] = Q.T@matrix[cuts[0]:,cuts[1]:]
    Q = 0 #Get rid of Q for memory purposes.

    #Shifts the columns of B
    matrix[:cuts[0],cuts[0]:cuts[1]] = matrix[:cuts[0],cuts[0]:cuts[1]][:,P]

    #Checks for 0 rows and gets rid of them.
    #rank = np.sum(np.abs(matrix.diagonal())>accuracy) + still_good
    #matrix = matrix[:rank]

    #eliminates rows we don't care about-- those at the bottom of the matrix
    #since the top corner is a square identity matrix, useful_rows + number_of_roots is the width of the Macaulay matrix
    matrix = row_swap_matrix(matrix)
    for row in matrix[::-1]:
        if np.allclose(row, 0):
            matrix = matrix[:-1]
        else:
            break
    #print("Final matrix.shape:\n", matrix.shape)
    #useful_rows = matrix.shape[1] - number_of_roots
    #matrix = matrix[:useful_rows,:]

    #set very small values in the matrix to zero before backsolving
    matrix[np.isclose(matrix, 0, atol=accuracy)] = 0

    #Resorts the matrix_terms.
    matrix_terms[cuts[0]:cuts[1]] = matrix_terms[cuts[0]:cuts[1]][P]
    #print("Macaulay1Rank:", np.sum(np.abs(matrix.diagonal())>accuracy))    
    
    return matrix, matrix_terms

def rrqr_reduceMacaulay2(matrix, matrix_terms, cuts, number_of_roots, accuracy = 1.e-10):
    ''' Reduces a Macaulay matrix, BYU style

    This function does the same thing as rrqr_reduceMacaulay but uses
    qr_multiply instead of qr and a multiplication
    to make the function faster and more memory efficient.

    This function only works properly if the bottom left (D) part of the matrix is zero

    Parameters
    ----------
    matrix : numpy array.
        The Macaulay matrix, sorted in BYU style.
    matrix_terms: numpy array
        Each row of the array contains a term in the matrix. The i'th row corresponds to
        the i'th column in the matrix.
    cuts : tuple
        When the matrix is reduced it is split into 3 parts with restricted pivoting. These numbers indicate
        where those cuts happen.
    accuracy : float
        What is determined to be 0.
    Returns
    -------
    matrix : numpy array
        The reduced matrix.
    matrix_terms: numpy array
        The resorted matrix_terms.
    '''
    #print("Starting matrix.shape:\n", matrix.shape)
    #RRQR reduces A and D without pivoting sticking the result in it's place.
    C1,matrix[:cuts[0],:cuts[0]] = qr_multiply(matrix[:,:cuts[0]], matrix[:,cuts[0]:].T, mode = 'right')
    matrix[:cuts[0],cuts[0]:] = C1.T
    C1 = 0

    #check if there are zeros along the diagonal of R1
    if any(np.isclose(np.diag(matrix[:,:cuts[0]]),0, atol=accuracy)):
        raise MacaulayError("R1 IS NOT FULL RANK")

    #if abs(matrix[:,:cuts[0]].diagonal()[-1]) < accuracy:
    #    raise MacaulayError("HIGHEST NOT FULL RANK")

    #set small values to zero before backsolving
    matrix[np.isclose(matrix, 0, atol=accuracy)] = 0

    matrix[:cuts[0],cuts[0]:] = solve_triangular(matrix[:cuts[0],:cuts[0]],matrix[:cuts[0],cuts[0]:])
    matrix[:cuts[0],:cuts[0]] = np.eye(cuts[0])
    matrix[cuts[0]:,cuts[0]:] -= (matrix[cuts[0]:,:cuts[0]])@matrix[:cuts[0],cuts[0]:] #?

    C,R,P = qr_multiply(matrix[cuts[0]:,cuts[0]:cuts[1]], matrix[cuts[0]:,cuts[1]:].T, mode = 'right', pivoting = True)

    matrix = matrix[:R.shape[0]+cuts[0]]
    #matrix[cuts[0]:,:cuts[0]] = np.zeros_like(matrix[cuts[0]:,:cuts[0]])
    matrix[cuts[0]:,cuts[0]:cuts[0]+R.shape[1]] = R
    matrix[cuts[0]:,cuts[0]+R.shape[1]:] = C.T
    C,R = 0,0

    #Shifts the columns of B.
    matrix[:cuts[0],cuts[0]:cuts[1]] = matrix[:cuts[0],cuts[0]:cuts[1]][:,P]
    matrix_terms[cuts[0]:cuts[1]] = matrix_terms[cuts[0]:cuts[1]][P]
    P = 0

    # Check if there are no solutions
    #rank = np.sum(np.abs(matrix.diagonal())>accuracy)

    # extra_block = matrix[rank:, -matrix_shape_stuff[2]:]
    # Q,R = qr(extra_block)
    # if np.sum(np.abs(R.diagonal())>accuracy) == matrix_shape_stuff[2]:
    #     raise ValueError("The system given has no roots.")

    #Get rid of 0 rows at the bottom.
    #matrix = matrix[:rank]

    #eliminates rows we don't care about-- those at the bottom of the matrix
    #since the top corner is a square identity matrix, always_useful_rows + number_of_roots is the width of the Macaulay matrix
    always_useful_rows = matrix.shape[1] - number_of_roots
    #matrix = matrix[:useful_rows,:]

    #set small values in the matrix to zero now, after the QR reduction
    matrix[np.isclose(matrix, 0, atol=accuracy)] = 0
    #eliminate zero rows from the bottom of the matrix. Zero rows above
    #nonzero elements are not eliminated. This saves time since Macaulay matrices
    #we deal with are only zero at the very bottom
    matrix = row_swap_matrix(matrix)
    for row in matrix[::-1]:
        if np.allclose(row, 0):
            matrix = matrix[:-1]
        else:
            break

    return matrix, matrix_terms

def rrqr_reduceMacaulayFullRank(matrix, matrix_terms, cuts, accuracy = 1.e-10):
    ''' Reduces a Macaulay matrix, BYU style.

    This function does the same thing as rrqr_reduceMacaulay2 but only works if the matrix is full rank AND if
    the top left corner (the square of side length cut[0]) is invertible.
    In this case it is faster.

    Parameters
    ----------
    matrix : numpy array.
        The Macaulay matrix, sorted in BYU style.
    matrix_terms: numpy array
        Each row of the array contains a term in the matrix. The i'th row corresponds to
        the i'th column in the matrix.
    cuts : tuple
        When the matrix is reduced it is split into 3 parts with restricted pivoting. These numbers indicate
        where those cuts happen.
    accuracy : float
        What is determined to be 0.
    Returns
    -------
    matrix : numpy array
        The reduced matrix.
    matrix_terms: numpy array
        The resorted matrix_terms.
    '''
    C1,matrix[:cuts[0],:cuts[0]] = qr_multiply(matrix[:cuts[0],:cuts[0]],\
                                                  matrix[:cuts[0],cuts[0]:].T, mode = 'right')
    matrix[:cuts[0],cuts[0]:] = C1.T
    C1 = 0

    #check if there are zeros along the diagonal of R1
    if any(np.isclose(np.diag(matrix[:,:cuts[0]]),0, atol=accuracy)):
        raise MacaulayError("R1 IS NOT FULL RANK")

    #if abs(matrix[:,:cuts[0]].diagonal()[-1]) < accuracy:
    #    raise MacaulayError("HIGHEST NOT FULL RANK")

    C,matrix[cuts[0]:,cuts[0]:cuts[1]],P = qr_multiply(matrix[cuts[0]:,cuts[0]:cuts[1]],\
                                                       matrix[cuts[0]:,cuts[1]:].T, mode = 'right', pivoting = True)

    matrix[cuts[0]:,cuts[1]:] = C.T
    C = 0

    #Shifts the columns of B.
    matrix[:cuts[0],cuts[0]:cuts[1]] = matrix[:cuts[0],cuts[0]:cuts[1]][:,P]
    matrix_terms[cuts[0]:cuts[1]] = matrix_terms[cuts[0]:cuts[1]][P]
    P = 0
    return matrix, matrix_terms

def checkEqual(lst):
    '''Helper function for createMatrixFast. Checks if each element in a list is the same.

    Parameters
    ----------
    lst : list
        The list of interest.
    Returns
    -------
    checkEqual : bool
        True if each element in the list is the same. False otherwise.
    '''
    return lst.count(lst[0]) == len(lst)

def get_ranges(nums):
    '''Helper function for createMatrixFast. Finds where to slice the different parts of the matrix into.

    This is in an effort to avoid row_swap_matrix which can be slow. Instead, as we are buiding the part of the
    matrix corresponding to each polynomial seperately, this tells us where each part should go in the whole matrix.

    Parameters
    ----------
    nums : list
        The Macualay matrix degree minus the polynomial degrees for for each polynomial.
    Returns
    -------
    ranges : list
        The rows in the Macaulay Matrix that the given polynomail will be sliced into.
    '''
    ranges = []
    for i in nums:
        ranges.append(np.array([],dtype=int))
    start = 0
    count = 0
    n = len(nums)
    for num in nums:
        spot = count
        for r in ranges[count:]:
            r = np.hstack((r,np.arange(start,start+(n-count)*(num-len(r)),n-count)))
            ranges[spot] = r
            start+=1
            spot += 1
        start = ranges[-1][-1]+1
        count+=1
    return ranges

def createMatrixFast(polys, degree, dim):
    ''' Builds a Macaulay matrix using fast construction.

    Parameters
    ----------
    poly_coeffs : list.
        Contains numpy arrays that hold the coefficients of the polynomials to be put in the matrix.
    degree : int
        The degree of the Macaulay Matrix
    dim : int
        The dimension of the polynomials going into the matrix.
    Returns
    -------
    matrix : 2D numpy array
        The Macaulay matrix.
    matrix_terms : numpy array
        The ith row is the term represented by the ith column of the matrix.
    cuts : tuple
        When the matrix is reduced it is split into 3 parts with restricted pivoting. These numbers indicate
        where those cuts happen.
    '''
    bigShape = [degree+1]*dim

    matrix_terms, cuts = sorted_matrix_terms(degree, dim)
    columns = len(matrix_terms)

    range_split = [num_mons_full(degree-poly.degree,dim) for poly in polys]
    rows = np.sum(range_split)
    ranges = get_ranges(range_split)    #How to slice the poly into the matrix rows.
    matrix = np.zeros((rows,columns))
    curr = 0

    #Get the slices needed to pull the matrix_terms from the coeff matrix.
    matrix_term_indexes = list()
    for row in matrix_terms.T:
        matrix_term_indexes.append(row)

    permutations = None
    currentDegree = 2
    #Adds the poly_coeffs to flat_polys, using added_zeros to make sure every term is in there.
    added_zeros = np.zeros(bigShape)

    for poly,matrix_range in zip(polys,ranges):
        slices = slice_top(poly.coeff)
        added_zeros[slices] = poly.coeff
        array = added_zeros[matrix_term_indexes]
        added_zeros[slices] = np.zeros_like(poly.coeff)

        permutations = memoized_all_permutations(degree - poly.degree, dim, degree, permutations, currentDegree)
        currentDegree = degree - poly.degree
        permList = list(permutations.values())

        temp = array[np.reshape(permList, (len(permList), columns))[::-1]]
        matrix[matrix_range] = temp

    if matrix_shape_stuff[0] > matrix.shape[0]: #The matrix isn't tall enough, these can't all be pivot columns.
        raise MacaulayError("HIGHEST NOT FULL RANK. TRY HIGHER DEGREE")
    #Sorts the rows of the matrix so it is close to upper triangular.
    if not checkEqual([poly.degree for poly in polys]): #Will need some switching possibly if some degrees are different.
        matrix = row_swap_matrix(matrix)
    return matrix, matrix_terms, cuts

def construction(polys, degree, dim):
    ''' Builds a Macaulay matrix using fast construction in the Chebyshev basis.

    Parameters
    ----------
    polys : list.
        Contains numpy arrays that hold the coefficients of the polynomials to be put in the matrix.
    degree : int
        The degree of the Macaulay Matrix
    dim : int
        The dimension of the polynomials going into the matrix.
    Returns
    -------
    matrix : 2D numpy array
        The Macaulay matrix.
    matrix_terms : numpy array
        The ith row is the term represented by the ith column of the matrix.
    cuts : tuple
        When the matrix is reduced it is split into 3 parts with restricted pivoting. These numbers indicate
        where those cuts happen.
    '''
    bigShape = [degree+1]*dim
    matrix_terms, cuts = sorted_matrix_terms(degree, dim)
    #print(matrix_shape_stuff)
    matrix_term_indexes = list()
    for row in matrix_terms.T:
        matrix_term_indexes.append(row)

    permutations = all_permutations_cheb(degree - np.min([poly.degree for poly in polys]), dim, degree)
    #print(permutations)
    added_zeros = np.zeros(bigShape)
    flat_polys = list()
    i = 0;
    for poly in polys:
        slices = slice_top(poly.coeff)
        added_zeros[slices] = poly.coeff
        array = added_zeros[matrix_term_indexes]
        added_zeros[slices] = np.zeros_like(poly.coeff)
        #print(array)

        #flat_polys.append(array[np.vstack(permutations.values())])
        degreeNeeded = degree - poly.degree
        mons = mons_ordered(dim,degreeNeeded)
        mons = np.pad(mons, (0,1), 'constant', constant_values = i)
        i += 1
        flat_polys.append(array)
        for mon in mons[1:-1]:
            result = np.copy(array)
            for i in range(dim):
                if mon[i] != 0:

                    mult = [0]*dim
                    mult[i] = mon[i]
                    result = np.sum(result[permutations[tuple(mult)]], axis = 0)
            flat_polys.append(result)
    #print(flat_polys)
    matrix = np.vstack(flat_polys)
    if matrix_shape_stuff[0] > matrix.shape[0]: #The matrix isn't tall enough, these can't all be pivot columns.
        raise MacaulayError("HIGHEST NOT FULL RANK. TRY HIGHER DEGREE")
    matrix = row_swap_matrix(matrix)
    #print(matrix)
    return matrix, matrix_terms, cuts
