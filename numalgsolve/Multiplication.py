import numpy as np
import itertools
from numalgsolve.polynomial import MultiCheb, MultiPower, is_power
from numalgsolve.TVBCore import rrqr_reduceTelenVanBarel2, find_degree, add_polys
from numalgsolve.utils import row_swap_matrix, TVBError, slice_top, get_var_list, mon_combos, mon_combosHighest, sort_polys_by_degree, deg_d_polys, all_permutations, num_mons_full, memoized_all_permutations, mons_ordered, all_permutations_cheb, num_mons
from scipy.linalg import solve_triangular, eig

def multiplication(polys):
    '''
    Finds the roots of the given list of multidimensional polynomials using a multiplication matrix.

    Parameters
    ----------
    polys : list of polynomial objects
        Polynomials to find the common roots of.
    returns
    -------
    roots : numpy array
        The common roots of the polynomials. Each row is a root.
    '''
    poly_type = is_power(polys, return_string = True)
    dim = polys[0].dim

    m_f, var_dict = TVBMultMatrix(polys, poly_type)

    # both TVBMultMatrix and groebnerMultMatrix will return m_f as
    # -1 if the ideal is not zero dimensional or if there are no roots
    if type(m_f) == int:
        return -1

    # Get list of indexes of single variables and store vars that were not
    # in the vector space basis.
    var_spots = list()
    spot = np.zeros(dim)
    for i in range(dim):
        spot[i] = 1
        var_spots.append(var_dict[tuple(spot)])
        spot[i] = 0

    # Get left eigenvectors

    vals,vecs = np.linalg.eig(m_f.T)

    zeros_spot = var_dict[tuple(0 for i in range(dim))]

    vecs = vecs[:,np.abs(vecs[zeros_spot]) > 1.e-10]

    roots = vecs[var_spots]/vecs[zeros_spot]
    return roots.T

def TVBMultMatrix(polys, poly_type):
    '''
    Finds the multiplication matrix using the reduced Macaulay matrix from the
    TVB method.

    Parameters
    ----------
    polys : array-like
        The polynomials to find the common zeros of
    poly_type : string
        The type of the polynomials in polys

    Returns
    -------
    multiplicationMatrix : 2D numpy array
        The multiplication matrix for a random polynomial f
    var_dict : dictionary
        Maps each variable to its position in the vector space basis
    '''
    basisDict, VB, degree = TelenVanBarel(polys)

    dim = max(f.dim for f in polys)

    # Get random polynomial f
    f = _random_poly(poly_type, dim)[0]

    #Dictionary of terms in the vector basis their spots in the matrix.
    VBdict = {}
    spot = 0
    for row in VB:
        VBdict[tuple(row)] = spot
        spot+=1

    # Build multiplication matrix m_f
    mMatrix = np.zeros((len(VB), len(VB)))
    for i in range(VB.shape[0]):
        f_coeff = f.mon_mult(VB[i], returnType = 'Matrix')
        for term in zip(*np.where(f_coeff != 0)):
            if term in VBdict:
                mMatrix[VBdict[term]][i] += f_coeff[term]
            else:
                mMatrix[:,i] -= f_coeff[term]*basisDict[term]

    # Construct var_dict
    var_dict = {}
    for i in range(len(VB)):
        mon = VB[i]
        if np.sum(mon) == 1 or np.sum(mon) == 0:
            var_dict[tuple(mon)] = i

    return mMatrix, var_dict

def TelenVanBarel(initial_poly_list, accuracy = 1.e-10):
    """Uses Telen and VanBarels matrix reduction method to find a vector basis for the system of polynomials.

    Parameters
    --------
    initial_poly_list: list
        The polynomials in the system we are solving.
    run_checks : bool
        If True, checks will be run to make sure TVB works and if it doesn't an S-polynomial will be searched
        for to fix it.
    accuracy: float
        How small we want a number to be before assuming it is zero.

    Returns
    -----------
    basisDict : dict
        A dictionary of terms not in the vector basis a matrixes of things in the vector basis that the term
        can be reduced to.
    VB : numpy array
        The terms in the vector basis, each row being a term.
    degree : int
        The degree of the Macaualy matrix that was constructed.
    """
    power = is_power(initial_poly_list)
    dim = initial_poly_list[0].dim
    poly_coeff_list = []
    degree = find_degree(initial_poly_list)

    #This sorting is required for fast matrix construction. Ascending should be False.
    initial_poly_list = sort_polys_by_degree(initial_poly_list, ascending = False)

    """This is the first construction option, simple monomial multiplication."""
    for poly in initial_poly_list:
        poly_coeff_list = add_polys(degree, poly, poly_coeff_list)
    """This is the second construction option, it uses the fancy triangle method that is faster but less stable."""
    #for deg in reversed(range(min([poly.degree for poly in initial_poly_list]), degree+1)):
    #    poly_coeff_list += deg_d_polys(initial_poly_list, deg, dim)

    #Creates the matrix for either of the above two methods. Comment out if using the third method.
    matrix, matrix_terms, cuts = create_matrix(poly_coeff_list, degree, dim)

    #print(matrix.shape)

    """This is the thrid matrix construction option, it uses the permutation arrays."""
    #if power:
    #    matrix, matrix_terms, cuts = createMatrixFast(initial_poly_list, degree, dim)
    #else:
    #    matrix, matrix_terms, cuts = construction(initial_poly_list, degree, dim)

    matrix, matrix_terms = rrqr_reduceTelenVanBarel2(matrix, matrix_terms, cuts, accuracy = accuracy)
    #matrix, matrix_terms = rrqr_reduceTelenVanBarelFullRank(matrix, matrix_terms, cuts, accuracy = accuracy)

    height = matrix.shape[0]
    matrix[:,height:] = solve_triangular(matrix[:,:height],matrix[:,height:])
    matrix[:,:height] = np.eye(height)

    #return np.vstack((matrix[:,height:].T,np.eye(height))), matrix_terms

    VB = matrix_terms[height:]

    #plt.plot(matrix_terms[:,0],matrix_terms[:,1],'kx')
    #plt.plot(VB[:,0],VB[:,1],'r.')

    basisDict = makeBasisDict(matrix, matrix_terms, VB, power)

    return basisDict, VB, degree

def makeBasisDict(matrix, matrix_terms, VB, power):
    '''Calculates and returns the basisDict.

    This is a dictionary of the terms on the diagonal of the reduced TVB matrix to the terms in the Vector Basis.
    It is used to create the multiplication matrix in root_finder.

    Parameters
    --------
    matrix: numpy array
        The reduced TVB matrix.
    matrix_terms : numpy array
        The terms in the matrix. The i'th row is the term represented by the i'th column of the matrix.
    VB : numpy array
        Each row is a term in the vector basis.
    power : bool
        If True, the initial polynomials were MultiPower. If False, they were MultiCheb.

    Returns
    -----------
    basisDict : dict
        Maps terms on the diagonal of the reduced TVB matrix (tuples) to numpy arrays that represent the
        terms reduction into the Vector Basis.
    '''
    basisDict = {}

    VBSet = set()
    for i in VB:
        VBSet.add(tuple(i))

    if power: #We don't actually need most of the rows, so we only get the ones we need.
        neededSpots = set()
        for term, mon in itertools.product(VB,get_var_list(VB.shape[1])):
            if tuple(term+mon) not in VBSet:
                neededSpots.add(tuple(term+mon))

    #spots = list()
    #for dim in range(VB.shape[1]):
    #    spots.append(VB.T[dim])

    for i in range(matrix.shape[0]):
        term = tuple(matrix_terms[i])
        if power and term not in neededSpots:
            continue
        basisDict[term] = matrix[i][matrix.shape[0]:]

    return basisDict

def create_matrix(poly_coeffs, degree, dim):
    ''' Builds a Telen Van Barel matrix.

    Parameters
    ----------
    poly_coeffs : list.
        Contains numpy arrays that hold the coefficients of the polynomials to be put in the matrix.
    degree : int
        The degree of the TVB Matrix
    dim : int
        The dimension of the polynomials going into the matrix.
    Returns
    -------
    matrix : 2D numpy array
        The Telen Van Barel matrix.
    matrix_terms : numpy array
        The ith row is the term represented by the ith column of the matrix.
    cuts : tuple
        When the matrix is reduced it is split into 3 parts with restricted pivoting. These numbers indicate
        where those cuts happen.
    '''
    bigShape = [degree+1]*dim
    matrix_terms, cuts = sorted_matrix_terms(degree, dim)

    #Get the slices needed to pull the matrix_terms from the coeff matrix.
    matrix_term_indexes = list()
    for row in matrix_terms.T:
        matrix_term_indexes.append(row)

    #Adds the poly_coeffs to flat_polys, using added_zeros to make sure every term is in there.
    added_zeros = np.zeros(bigShape)
    flat_polys = list()
    for coeff in poly_coeffs:
        slices = slice_top(coeff)
        added_zeros[slices] = coeff
        flat_polys.append(added_zeros[matrix_term_indexes])
        added_zeros[slices] = np.zeros_like(coeff)
        coeff = 0
    poly_coeffs = 0

    #Make the matrix. Reshape is faster than stacking.
    matrix = np.reshape(flat_polys, (len(flat_polys),len(matrix_terms)))

    #if cuts[0] > matrix.shape[0]: #The matrix isn't tall enough, these can't all be pivot columns.
    #    raise TVBError("HIGHEST NOT FULL RANK. TRY HIGHER DEGREE")

    #Sorts the rows of the matrix so it is close to upper triangular.
    matrix = row_swap_matrix(matrix)
    return matrix, matrix_terms, cuts

def sorted_matrix_terms(degree, dim):
    '''Finds the matrix_terms sorted in the term order needed for TelenVanBarel reduction.
    So the highest terms come first,the x,y,z etc monomials last.
    Parameters
    ----------
    degree : int
        The degree of the TVB Matrix
    dim : int
        The dimension of the polynomials going into the matrix.
    Returns
    -------
    sorted_matrix_terms : numpy array
        The sorted matrix_terms. The ith row is the term represented by the ith column of the matrix.
    cuts : tuple
        When the matrix is reduced it is split into 3 parts with restricted pivoting. These numbers indicate
        where those cuts happen.
    '''
    highest_mons = mon_combosHighest([0]*dim,degree)[::-1]

    other_mons = list()
    d = degree - 1
    while d > 1:
        other_mons += mon_combosHighest([0]*dim,d)[::-1]
        d -= 1

    xs_mons = mon_combos([0]*dim,1)[::-1]
    sorted_matrix_terms = np.reshape(highest_mons+other_mons+xs_mons, (len(highest_mons+other_mons+xs_mons),dim))
    return sorted_matrix_terms, tuple([len(highest_mons),len(highest_mons)+len(other_mons)])

def _random_poly(_type, dim):
    '''
    Generates a random polynomial that has the form
    c_1x_1 + c_2x_2 + ... + c_nx_n where n = dim and each c_i is a randomly
    chosen integer between 0 and 1000.

    Parameters
    ----------
    _type : string
        Type of Polynomial to generate. "MultiCheb" or "MultiPower".
    dim : int
        Degree of polynomial to generate (?).

    Returns
    -------
    Polynomial
        Randomly generated Polynomial.
    '''
    _vars = get_var_list(dim)

    random_poly_shape = [2 for i in range(dim)]

    random_poly_coeff = np.zeros(tuple(random_poly_shape), dtype=int)
    for var in _vars:
        random_poly_coeff[var] = np.random.randint(1000)

    if _type == 'MultiCheb':
        return MultiCheb(random_poly_coeff), _vars
    else:
        return MultiPower(random_poly_coeff), _vars
