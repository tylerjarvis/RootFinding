import numpy as np
import itertools
from scipy.linalg import solve_triangular, eig, schur
from yroots.LinearProjection import nullspace
from yroots.polynomial import MultiCheb, MultiPower, is_power
from yroots.MacaulayReduce import reduce_macaulay_qrt, find_degree, \
                              add_polys, reduce_macaulay_tvb, reduce_macaulay_svd
from yroots.utils import row_swap_matrix, MacaulayError, slice_top, get_var_list, \
                              mon_combos, mon_combosHighest, sort_polys_by_degree, \
                              deg_d_polys, all_permutations_cheb,\
                              newton_polish, condeigs, solve_linear, memoize
import warnings
from scipy.stats import ortho_group

def multiplication(polys, max_cond_num, verbose=False, return_all_roots=True,method='svd'):
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
    '''
    #We don't want to use Linear Projection right now
#    polys, transform, is_projected = polys, lambda x:x, False

    if len(polys) == 1:
        from yroots.OneDimension import solve
        return transform(solve(polys[0], MSmatrix=0))
    poly_type = is_power(polys, return_string = True)
    dim = polys[0].dim

    #By Bezout's Theorem. Useful for making sure that the reduced Macaulay Matrix is as we expect
    bezout_bound = np.prod([poly.degree for poly in polys])

    matrix, matrix_terms, cut = build_macaulay(polys, verbose)

    roots = np.array([])

    # If cut is zero, then all the polynomials are linear and we solve
    # using solve_linear.
    if cut == 0:
        roots, cond = solve_linear([p.coeff for p in polys])
        # Make sure roots is a 2D array.
        roots = np.array([roots])
    else:
        # Attempt to reduce the Macaulay matrix
        if method == 'svd':
            res = reduce_macaulay_svd(matrix,cut,bezout_bound,max_cond_num)
            if res[0] is None:
                return res
            E,Q = res
        elif method == 'qrt':
            res = reduce_macaulay_qrt(matrix,cut,bezout_bound,max_cond_num)
            if res[0] is None:
                return res
            E,Q = res
        elif method == 'tvb':
            res = reduce_macaulay_tvb(matrix,cut,bezout_bound,max_cond_num)
            if res[0] is None:
                return res
            E,Q = res
        else:
            raise ValueError("Method must be one of 'svd','qrt' or 'tvb'")

        # Construct the Möller-Stetter matrices
        # M is a 3d array containing the multiplication-by-x_i matrix in M[...,i]
        if poly_type == "MultiCheb":
            if method == 'qrt' or method == 'svd':
                M = ms_matrices_cheb(E,Q,matrix_terms,dim)
            elif method == 'tvb':
                M = ms_matrices_p_cheb(E,Q,matrix_terms,dim,cut)

        else:
            if method == 'qrt' or method == 'svd':
                M = ms_matrices(E,Q,matrix_terms,dim)
            elif method == 'tvb':
                M = ms_matrices_p(E,Q,matrix_terms,dim,cut)

        # Compute the roots using eigenvalues of the Möller-Stetter matrices
        roots = msroots(M)

    if return_all_roots:
        return roots
    else:
        # only return roots in the unit complex hyperbox
        return roots[[np.all(np.abs(root) <= 1) for root in roots]]

def indexarray(matrix_terms,m,var):
    """Compute the array mapping monomials under multiplication by x_var

    Parameters
    ----------
    matrix_terms : 2d integer ndarray
        Array containing the monomials in order. matrix_terms[i] is the array
        containing the exponent for each variable in the ith multivariate
        monomial
    m : int
        Number of monomials of highest degree, i.e. those that do not need to be
        multiplied
    var : int
        Variable to multiply by: x_0,...,x_(dim-1)

    Returns
    -------
    arr : 1d integer ndarray
        Array containing the indices of the lower-degree monomials after multiplication
        by x_var
    """
    mults = matrix_terms[m:].copy()
    mults[:,var] += 1
    return np.argmin(np.abs(mults[:,np.newaxis] - matrix_terms[np.newaxis]).sum(axis=-1),axis=1)

def indexarray_cheb(matrix_terms,m,var):
    """Compute the array mapping Chebyshev monomials under multiplication by x_var:

        T_1*T_0 = T_1
        T_1*T_n = .5(T_(n+1)+ T_(n-1))

    Parameters
    ----------
    matrix_terms : 2d integer ndarray
        Array containing the monomials in order. matrix_terms[i] is the array
        containing the degree for each univariate Chebyshev monomial in the ith
        multivariate monomial
    m : int
        Number of monomials of highest degree, i.e. those that do not need to be
        multiplied
    var : int
        Variable to multiply by: x_0,...,x_(dim-1)

    Returns
    -------
    arr1 : 1d integer ndarray
        Array containing the indices of T_(n+1)
    arr2 : 1d
        Array containing the indices of T_(n-1)
    """
    up = matrix_terms[m:].copy()
    up[:,var] += 1
    down = matrix_terms[m:].copy()
    down[:,var] -= 1
    down[down[:,var]==-1,var] += 2
    arr1 = np.argmin(np.abs(up[:,np.newaxis] - matrix_terms[np.newaxis]).sum(axis=-1),axis=1)
    arr2 = np.argmin(np.abs(down[:,np.newaxis] - matrix_terms[np.newaxis]).sum(axis=-1),axis=1)
    return arr1,arr2

def ms_matrices(E,Q,matrix_terms,dim):
    """Compute the Möller-Stetter matrices in the monomial basis

    Parameters
    ----------
    E : (m,k) ndarray
        Columns of the reduced Macaulay matrix corresponding to the quotient basis
    Q : (l,n) 2d ndarray
        Matrix whose columns give the quotient basis in terms of the monomial basis
    matrix_terms : 2d ndarray
        Array with ordered monomial basis
    dim : int
        Number of variables

    Returns
    -------
    M : (n,n,dim) ndarray
        Array containing the nxn Möller-Stetter matrices, where the matrix
        corresponding to multiplication by x_i is M[...,i]
    """
    n = Q.shape[1]
    m = E.shape[0]
    M = np.empty((n,n,dim))
    A = np.hstack((-E.T,Q.T))
    for i in range(dim):
        arr = indexarray(matrix_terms,m,i)
        M[...,i] = A[:,arr]@Q
    return M

def ms_matrices_cheb(E,Q,matrix_terms,dim):
    """Compute the Möller-Stetter matrices in the Chebyshev basis

    Parameters
    ----------
    E : (m,k) ndarray
        Columns of the reduced Macaulay matrix corresponding to the quotient basis
    Q : (l,n) 2d ndarray
        Matrix whose columns give the quotient basis in terms of the Chebyshev basis
    matrix_terms : 2d ndarray
        Array with ordered Chebyshev basis
    dim : int
        Number of variables

    Returns
    -------
    M : (n,n,dim) ndarray
        Array containing the nxn Möller-Stetter matrices, where the matrix
        corresponding to multiplication by x_i is M[...,i]
    """
    n = Q.shape[1]
    m = E.shape[0]
    M = np.empty((n,n,dim))
    A = np.hstack((-E.T,Q.T))
    for i in range(dim):
        arr1,arr2 = indexarray_cheb(matrix_terms,m,i)
        M[...,i] = .5*(A[:,arr1]+A[:,arr2])@Q
    return M

def ms_matrices_p(E,P,matrix_terms,dim,cut):
    r,n = E.shape
    matrix_terms[cut:] = matrix_terms[cut:][P]
    M = np.empty((n,n,dim))
    A = np.hstack((-E.T,np.eye(n)))
    for i in range(dim):
        arr = indexarray(matrix_terms,r,i)
        M[...,i] = A[:,arr]
    return M

def ms_matrices_p_cheb(E,P,matrix_terms,dim,cut):
    """ Compute the Möller-Stetter matrices in the Chebyshev basis in the
        Telen-Van Barel method.

    Parameters
    ----------
    E : (m,k) ndarray
        Columns of the reduced Macaulay matrix corresponding to the quotient basis
    P : (,l) ndarray
        Array of pivots returned in QR with pivoting, used to permute the columns.
    matrix_terms : 2d ndarray
        Array with ordered Chebyshev basis
    dim : int
        Number of variables

    Returns
    -------
    M : (n,n,dim) ndarray
        Array containing the nxn Möller-Stetter matrices, where the matrix
        corresponding to multiplication by x_i is M[...,i]
    """
    r,n = E.shape
    matrix_terms[cut:] = matrix_terms[cut:][P]
    M = np.empty((n,n,dim))
    A = np.hstack((-E.T,np.eye(n)))
    for i in range(dim):
        arr1,arr2 = indexarray_cheb(matrix_terms,r,i)
        M[...,i] = .5*(A[:,arr1]+A[:,arr2])
    return M

def sort_eigs(eigs,diag):
    """Sorts the eigs array to match the order on the diagonal
    of the Schur factorization

    Parameters
    ----------
    eigs : 1d ndarray
        Array of unsorted eigenvalues
    diag : 1d complex ndarray
        Array containing the diagonal of the approximate Schur factorization

    Returns
    -------
    w : 1d ndarray
        Eigenvalues from eigs sorted to match the order in diag
    """
    n = diag.shape[0]
    lst = list(range(n))
    arr = []
    for eig in eigs:
        i = lst[np.argmin(np.abs(diag[lst]-eig))]
        arr.append(i)
        lst.remove(i)
    return np.argsort(arr)

@memoize
def get_Q_c(dim):
    """Generates a once-chosen random orthogonal matrix and a random linear combination
    for use in the simultaneous eigenvalue compution.

    Parameters
    ----------
    dim : int
        Dimension of the system

    Returns
    -------
    Q : (dim,dim) ndarray
        Random orthogonal rotation
    c : (dim,) ndarray
        Random linear combination
    """
    np.random.seed(103)
    Q = ortho_group.rvs(dim)
    c = np.random.randn(dim)
    return Q,c

def msroots(M):
    """Computes the roots to a system via the eigenvalues of the Möller-Stetter
    matrices. Implicitly performs a random rotation of the coordinate system
    to avoid repeated eigenvalues arising from special structure in the underlying
    polynomial system. Approximates the joint eigenvalue problem using a Schur
    factorization of a linear combination of the matrices.

    Parameters
    ----------
    M : (n,n,dim) ndarray
        Array containing the nxn Möller-Stetter matrices, where the matrix
        corresponding to multiplication by x_i is M[...,i]

    Returns
    -------
    roots : (n,dim) ndarray
        Array containing the approximate roots of the system, where each row
        is a root.
    """
    dim = M.shape[-1]

    # perform a random rotation with a random orthogonal Q
    Q,c = get_Q_c(dim)
    M = (Q@M[...,np.newaxis])[...,0]

    eigs = np.empty((dim,M.shape[0]),dtype='complex')
    # Compute the matrix U that triangularizes a random linear combination
    U = schur((M*c).sum(axis=-1),output='complex')[1]

    for i in range(0,dim):
        T = (U.conj().T)@(M[...,i])@U
        w = eig(M[...,i],right=False)
        arr = sort_eigs(w,np.diag(T))
        eigs[i] = w[arr]

    # Rotate back before returning, transposing to match expected shape
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

def build_macaulay(initial_poly_list, verbose=False):
    """Constructs the unreduced Macaulay matrix. Removes linear polynomials by
    substituting in for a number of variables equal to the number of linear
    polynomials.

    Parameters
    --------
    initial_poly_list: list
        The polynomials in the system we are solving.
    verbose : bool
        Prints information about how the roots are computed.
    Returns
    -----------
    matrix : 2d ndarray
        The Macaulay matrix
    matrix_terms : 2d integer ndarray
        Array containing the ordered basis, where the ith row contains the
        exponent/degree of the ith basis monomial
    cut : int
        Where to cut the Macaulay matrix for the highest-degree monomials
    varsToRemove : list
        The variables removed with removing linear polynomials
    A : 2d ndarray
        A matrix giving the linear relations between the removed variables and
        the remaining variables
    Pc : 1d integer ndarray
        Array containing the order of the variables as the appear in the columns
        of A
    """
    power = is_power(initial_poly_list)
    dim = initial_poly_list[0].dim
    poly_coeff_list = []
    degree = find_degree(initial_poly_list)

    # linear_polys = [poly for poly in initial_poly_list if poly.degree == 1]
    # nonlinear_polys = [poly for poly in initial_poly_list if poly.degree != 1]
    # #Choose which variables to remove if things are linear, and add linear polys to matrix
    # if len(linear_polys) >= 1: #Linear polys involved
    #     #get the row rededuced linear coefficients
    #     A,Pc = nullspace(linear_polys)
    #     varsToRemove = Pc[:len(A)].copy()
    #     #add to macaulay matrix
    #     for row in A:
    #         #reconstruct a polynomial for each row
    #         coeff = np.zeros([2]*dim)
    #         coeff[tuple(get_var_list(dim))] = row[:-1]
    #         coeff[tuple([0]*dim)] = row[-1]
    #         if not power:
    #             poly = MultiCheb(coeff)
    #         else:
    #             poly = MultiPower(coeff)
    #         poly_coeff_list = add_polys(degree, poly, poly_coeff_list)
    # else: #no linear
    #     A,Pc = None,None
    #     varsToRemove = []

    #add nonlinear polys to poly_coeff_list
    for poly in initial_poly_list:#nonlinear_polys:
        poly_coeff_list = add_polys(degree, poly, poly_coeff_list)

    #Creates the matrix
    # return (*create_matrix(poly_coeff_list, degree, dim, varsToRemove), A, Pc)
    return create_matrix(poly_coeff_list, degree, dim)#, varsToRemove)

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

def create_matrix(poly_coeffs, degree, dim):#, varsToRemove):
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

    matrix_terms, cut = sorted_matrix_terms(degree, dim)#, varsToRemove)

    #Get the slices needed to pull the matrix_terms from the coeff matrix.
    matrix_term_indexes = list()
    for row in matrix_terms.T:
        matrix_term_indexes.append(row)

    #Adds the poly_coeffs to flat_polys, using added_zeros to make sure every term is in there.
    added_zeros = np.zeros(bigShape)
    flat_polys = list()
    for coeff in poly_coeffs:
        slices = slice_top(coeff.shape)
        added_zeros[slices] = coeff
        flat_polys.append(added_zeros[tuple(matrix_term_indexes)])
        added_zeros[slices] = np.zeros_like(coeff)
    del poly_coeffs

    #Make the matrix. Reshape is faster than stacking.
    matrix = np.reshape(flat_polys, (len(flat_polys),len(matrix_terms)))

    #Sorts the rows of the matrix so it is close to upper triangular.
    matrix = row_swap_matrix(matrix)
    return matrix, matrix_terms, cut

def sorted_matrix_terms(degree, dim):#, varsToRemove):
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
        cuts = len(highest_mons)

    # for var in varsToRemove:
    #     B = matrix_terms[cuts[0]:]
    #     mask = B[:,var] != 0
    #     matrix_terms[cuts[0]:] = np.vstack([B[mask], B[~mask]])
    #     cuts = tuple([cuts[0] + np.sum(mask), cuts[1]+1])

    return matrix_terms, cuts

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
