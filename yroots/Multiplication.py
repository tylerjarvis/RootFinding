import numpy as np
import itertools
from scipy.linalg import solve_triangular, eig, schur
from yroots.LinearProjection import nullspace
from yroots.polynomial import MultiCheb, MultiPower, is_power
from yroots.MacaulayReduce import reduce_macaulay, find_degree, \
                              add_polys
from yroots.utils import row_swap_matrix, MacaulayError, slice_top, get_var_list, \
                              mon_combos, mon_combosHighest, sort_polys_by_degree, \
                              deg_d_polys, all_permutations_cheb, ConditioningError, newton_polish
import warnings
from scipy.stats import ortho_group

def multiplication(polys, max_cond_num, verbose=False, return_all_roots=True):
    '''
    Finds the roots of the given list of multidimensional polynomials using a multiplication matrix.

    Parameters
    ----------
    polys : list of polynomial objects
        Polynomials to find the common roots of.
    max_cond_num : float
        The maximum condition number of the Macaulay Matrix Reduction
    verbose : bool
        Prints information about how the roots are computed.
    return_all_roots : bool
        If True returns all the roots, otherwise just the ones in the unit box.
    returns
    -------
    roots : numpy array
        The common roots of the polynomials. Each row is a root.
    Raises
    ------
    ConditioningError if MSMultMatrix(...) raises a ConditioningError.
    '''
    #We don't want to use Linear Projection right now
#    polys, transform, is_projected = polys, lambda x:x, False

    if len(polys) == 1:
        from yroots.OneDimension import solve
        return transform(solve(polys[0], MSmatrix=0))
    poly_type = is_power(polys, return_string = True)
    dim = polys[0].dim

    #By Bezout's Theorem. Useful for making sure that the reduced Macaulay Matrix is as we expect
    degrees = [poly.degree for poly in polys]
    max_number_of_roots = np.prod(degrees)

    matrix, matrix_terms, cut, A, Pc = build_macaulay(polys, max_cond_num, verbose)

    try:
        E,Q = reduce_macaulay(matrix,cut,max_cond_num)
    except ConditioningError as e:
        raise e

    if poly_type == "MultiCheb":
        M = ms_matrices_cheb(E,Q,matrix_terms,dim)
    else:
        M = ms_matrices(E,Q,matrix_terms,dim)

    roots = msroots(M)

    if A:
        n = A.shape[0]
        tmp = np.empty((roots.shape[0],n),dtype='complex')
        tmp[Pc[n:]] = roots
        tmp[Pc[:n]] = (-A[:,n:-1]@(roots.T)-A[:,-1]).T
        roots = tmp

    #Check if too many roots
    assert roots.shape[0] <= max_number_of_roots,"Found too many roots,{}/{}/{}:{}".format(roots.shape,max_number_of_roots, degrees,roots)
    if return_all_roots:
        return roots
    else:
        # only return roots in the unit complex hyperbox
        return roots[np.all(np.abs(roots) <= 1,axis = 0)]

def indexarray(matrix_terms,m,var):
    mults = matrix_terms[m:].copy()
    mults[:,var] += 1
    return np.argmin(np.abs(mults[:,np.newaxis] - matrix_terms[np.newaxis]).sum(axis=-1),axis=1)

def indexarray_cheb(matrix_terms,m,var):
    up = matrix_terms[m:].copy()
    up[:,var] += 1
    down = matrix_terms[m:].copy()
    down[:,var] -= 1
    down[down[:,var]==-1,var] += 2
    arr1 = np.argmin(np.abs(up[:,np.newaxis] - matrix_terms[np.newaxis]).sum(axis=-1),axis=1)
    arr2 = np.argmin(np.abs(down[:,np.newaxis] - matrix_terms[np.newaxis]).sum(axis=-1),axis=1)
    return arr1,arr2

def ms_matrices(E,Q,matrix_terms,dim):
    n = Q.shape[1]
    m = E.shape[0]
    M = np.empty((n,n,dim))
    A = np.hstack((-E.T,Q.T))
    for i in range(dim):
        arr = indexarray(matrix_terms,m,i)
        M[...,i] = A[:,arr]@Q
    return M

def ms_matrices_cheb(E,Q,matrix_terms,dim):
    n = Q.shape[1]
    m = E.shape[0]
    M = np.empty((n,n,dim))
    A = np.hstack((-E.T,Q.T))
    for i in range(dim):
        arr1,arr2 = indexarray_cheb(matrix_terms,m,i)
        M[...,i] = .5*(A[:,arr1]+A[:,arr2])@Q
    return M

def sort_eigs(eigs,diag):
    """Sorts the eigs array to match the order on the diagonal
    of the Schur factorization"""
    n = diag.shape[0]
    lst = list(np.arange(n))
    w = np.empty_like(eigs)
    for eig in eigs:
        i = lst[np.argmin(np.abs(diag[lst]-eig))]
        w[i] = eig
        lst.remove(i)
    return w

def msroots(M):
    # perform a random rotation
    dim = M.shape[-1]
    Q = ortho_group.rvs(dim)
    M = (Q@M[...,np.newaxis])[...,0]
    eigs = np.empty((dim,M.shape[0]),dtype='complex')
    T,Z = schur(M[...,0],output='complex')
    eigs[0] = sort_eigs(eig(M[...,0],right=False),np.diag(T))
    for i in range(1,dim):
        T = (Z.conj().T)@(M[...,i])@Z
        eigs[i] = sort_eigs(eig(M[...,i],right=False),np.diag(T))
    return (Q.T@eigs).T

def MSMultMatrix(polys, poly_type, max_cond_num, macaulay_zero_tol, verbose=False, MSmatrix=0):
    '''
    Finds the multiplication matrix using the reduced Macaulay matrix.

    Parameters
    ----------
    polys : array-like
        The polynomials to find the common zeros of
    poly_type : string
        The type of the polynomials in polys
    verbose : bool
        Prints information about how the roots are computed.
    MSmatrix : int
        Controls which Moller-Stetter matrix is constructed. The options are:
            0 (default) -- The Moller-Stetter matrix of a random polynomial
            Some positive integer i < dimension -- The Moller-Stetter matrix of x_i
    max_cond_num : float
        The maximum condition number of the Macaulay Matrix Reduction
    macaulay_zero_tol : float
        What is considered 0 in the macaulay matrix reduction.
    Returns
    -------
    multiplicationMatrix : 2D numpy array
        The multiplication matrix for a random polynomial f
    var_dict : dictionary
        Maps each variable to its position in the vector space basis
    basisDict : dict
        A dictionary of terms not in the vector basis a matrixes of things in the vector basis that the term
        can be reduced to.
    VB : numpy array
        The terms in the vector basis, each row being a term.

    Raises
    ------
    ConditioningError if MacaulayReduction(...) raises a ConditioningError.
    '''
    try:
        basisDict, VB, varsRemoved = MacaulayReduction(polys, max_cond_num=max_cond_num, macaulay_zero_tol=macaulay_zero_tol, verbose=verbose)
    except ConditioningError as e:
        raise e

    dim = max(f.dim for f in polys)

    # Get the polynomial to make the MS matrix of
    if MSmatrix==0: #random poly
        f = _random_poly(poly_type, dim)[0]
    else: #multiply by x_i where i is determined by MSmatrix
        xi_ind = np.zeros(dim, dtype=int)
        xi_ind[MSmatrix-1] = 1
        coef = np.zeros((2,)*dim)
        coef[tuple(xi_ind)] = 1
        if poly_type == "MultiPower":
            f = MultiPower(np.array(coef))
        elif poly_type == "MultiCheb":
            f = MultiCheb(np.array(coef))
        else:
            raise ValueError()

    if verbose:
        print("\nCoefficients of polynomial whose Moller-Stetter matrix we construt\n", f.coeff)

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

    return mMatrix, var_dict, basisDict, VB

def build_macaulay(initial_poly_list, max_cond_num, verbose=False):
    """Reduces the Macaulay matrix to find a vector basis for the system of polynomials.

    Parameters
    --------
    initial_poly_list: list
        The polynomials in the system we are solving.
    max_cond_num : float
        The maximum condition number of the Macaulay Matrix Reduction
    verbose : bool
        Prints information about how the roots are computed.
    Returns
    -----------
    matrix : ndarray
        The Macaulay matrix
    matrix_terms : numpy array
        Array containing the monomial column labels
    cut : int
        Where to cut the Macaulay matrix for the highest-degree monomials
    varsToRemove : list
        The variables to remove from the basis because we have linear polysnomials

    Raises
    ------
    ConditioningError if rrqr_reduceMacaulay(...) raises a ConditioningError.
    """
    power = is_power(initial_poly_list)
    dim = initial_poly_list[0].dim
    poly_coeff_list = []
    degree = find_degree(initial_poly_list)

    linear_polys = [poly for poly in initial_poly_list if poly.degree == 1]
    nonlinear_polys = [poly for poly in initial_poly_list if poly.degree != 1]
    #Choose which variables to remove if things are linear, and add linear polys to matrix
    if len(linear_polys) == 1: #one linear
        varsToRemove = [np.argmax(np.abs(linear_polys[0].coeff[get_var_list(dim)]))]
        poly_coeff_list = add_polys(degree, linear_polys[0], poly_coeff_list)
    elif len(linear_polys) > 1: #multiple linear
        #get the row rededuced linear coefficients
        A,Pc = nullspace(linear_polys)
        varsToRemove = Pc[:len(A)].copy()
        #add to macaulay matrix
        for row in A:
            #reconstruct a polynomial for each row
            coeff = np.zeros([2]*dim)
            coeff[get_var_list(dim)] = row[:-1]
            coeff[tuple([0]*dim)] = row[-1]
            if not ower:
                poly = MultiCheb(coeff)
            else:
                poly = MultiPower(coeff)
            poly_coeff_list = add_polys(degree, poly, poly_coeff_list)
    else: #no linear
        A,Pc = None,None
        varsToRemove = []

    #add nonlinear polys to poly_coeff_list
    for poly in nonlinear_polys:
        poly_coeff_list = add_polys(degree, poly, poly_coeff_list)

    #Creates the matrix
    return (*create_matrix(poly_coeff_list, degree, dim, varsToRemove), A, Pc)

def makeBasisDict(matrix, matrix_terms, VB, power):
    '''Calculates and returns the basisDict.

    This is a dictionary of the terms on the diagonal of the reduced Macaulay matrix to the terms in the Vector Basis.
    It is used to create the multiplication matrix in root_finder.

    Parameters
    --------
    matrix: numpy array
        The reduced Macaulay matrix.
    matrix_terms : numpy array
        The terms in the matrix. The i'th row is the term represented by the i'th column of the matrix.
    VB : numpy array
        Each row is a term in the vector basis.
    power : bool
        If True, the initial polynomials were MultiPower. If False, they were MultiCheb.

    Returns
    -----------
    basisDict : dict
        Maps terms on the diagonal of the reduced Macaulay matrix (tuples) to numpy arrays that represent the
        terms reduction into the Vector Basis.
    '''
    basisDict = {}

    VBSet = set()
    for i in VB:
        VBSet.add(tuple(i))

    #We don't actually need most of the rows, so we only get the ones we need
    if power:
        neededSpots = set()
        for term, mon in itertools.product(VB,get_var_list(VB.shape[1])):
            if tuple(term+mon) not in VBSet:
                neededSpots.add(tuple(term+mon))

    for i in range(matrix.shape[0]):
        term = tuple(matrix_terms[i])
        if power and term not in neededSpots:
            continue
        basisDict[term] = matrix[i][matrix.shape[0]:]

    return basisDict

def create_matrix(poly_coeffs, degree, dim, varsToRemove):
    ''' Builds a Macaulay matrix.

    Parameters
    ----------
    poly_coeffs : list.
        Contains numpy arrays that hold the coefficients of the polynomials to be put in the matrix.
    degree : int
        The degree of the Macaulay Matrix
    dim : int
        The dimension of the polynomials going into the matrix.
    varsToRemove : list
        The variables to remove from the basis because we have linear polysnomials
    Returns
    -------
    matrix : 2D numpy array
        The Macaulay matrix.
    matrix_terms : numpy array
        The ith row is the term represented by the ith column of the matrix.
    cut : int
        Number of monomials of highest degree
    '''
    bigShape = [degree+1]*dim

    matrix_terms, cut = sorted_matrix_terms(degree, dim, varsToRemove)

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
        flat_polys.append(added_zeros[tuple(matrix_term_indexes)])
        added_zeros[slices] = np.zeros_like(coeff)
    del poly_coeffs

    #Make the matrix. Reshape is faster than stacking.
    matrix = np.reshape(flat_polys, (len(flat_polys),len(matrix_terms)))

    #Sorts the rows of the matrix so it is close to upper triangular.
    matrix = row_swap_matrix(matrix)
    return matrix, matrix_terms, cut

def sorted_matrix_terms(degree, dim, varsToRemove):
    '''Finds the matrix_terms sorted in the term order needed for Macaulay reduction.
    So the highest terms come first,the x,y,z etc monomials last.
    Parameters
    ----------
    degree : int
        The degree of the Macaulay Matrix
    dim : int
        The dimension of the polynomials going into the matrix.
    varsToRemove : list
        The variables to remove from the basis because we have linear polysnomials
    Returns
    -------
    sorted_matrix_terms : numpy array
        The sorted matrix_terms. The ith row is the term represented by the ith column of the matrix.
    cuts : int
        Number of monomials of highest degree
    '''
    highest_mons = mon_combosHighest([0]*dim,degree)[::-1]

    other_mons = list()
    d = degree - 1
    while d > 1:
        other_mons += mon_combosHighest([0]*dim,d)[::-1]
        d -= 1

    #extra-small monomials: 1,x,y, etc.
    xs_mons = mon_combos([0]*dim,1)[::-1]

    #trivial case
    if degree == 1:
        matrix_terms = np.reshape(xs_mons, (len(xs_mons),dim))
        cuts = 0
    #normal case
    else:
        matrix_terms = np.reshape(highest_mons+other_mons+xs_mons, (len(highest_mons+other_mons+xs_mons),dim))
        cut = len(highest_mons)

    # for var in varsToRemove:
    #     B = matrix_terms[cuts[0]:]
    #     mask = B[:,var] != 0
    #     matrix_terms[cuts[0]:] = np.vstack([B[mask], B[~mask]])
    #     cuts = tuple([cuts[0] + np.sum(mask), cuts[1]+1])

    return matrix_terms, cut

def _random_poly(_type, dim):
    '''
    Generates a random linear polynomial that has the form
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

    # random_poly_coeff = np.zeros(tuple(random_poly_shape), dtype=int)
    # for var in _vars:
    #     random_poly_coeff[var] = np.random.randint(1000)

    random_poly_coeff = np.zeros(tuple(random_poly_shape), dtype=float)
    #np.random.seed(42)

    coeffs = np.random.rand(dim)
    coeffs /= np.linalg.norm(coeffs)
    for i,var in enumerate(_vars):
        random_poly_coeff[var] = coeffs[i]

    if _type == 'MultiCheb':
        return MultiCheb(random_poly_coeff), _vars
    else:
        return MultiPower(random_poly_coeff), _vars
