import numpy as np
from numpy.fft import fftn
from yroots.utils import get_var_list, mon_combos, mon_combos_limited_wrap
from yroots.polynomial import Polynomial, MultiCheb
from scipy.linalg import qr
from scipy import sparse as sp
from itertools import product

def cheb_remove_linear(polys, approx_tol, solve_tol, transform_in=None):
    """This function recursively removes linear polynomials from a list by
    applying the project_down function once for each linear polynomial.
    This function assumes these polynomials had the zeros removed already, so that #TODO what zeros?
    it can rely on dimensions to detect linear polynomials.

    Parameters
    ----------
    polys : list of polynomial objects
        Polynomials to find the common roots of.
    approx_tol: float
        A tolerance to pass into the trim_coeff.
    solve_tol : float
        A tolerance to pass into the trim_coeff.
    transform_in : function
        only intended for use in recursion

    Returns
    -------
    polys : list of polynomial objects
        Polynomials to find the common roots of.
    transform : function
        A function mapping the roots of the output system to the roots of the
        original system.
    projected : bool
        True is projection was performed, False if no projection
    """
    assert len(polys) > 0
    transform_in = transform_in or (lambda x:x)
    if len(polys) == 1:
        if polys[0].shape[0] <= 2:
            raise ValueError("All of the polynomials were linear.")
        else:
            return polys, transform_in
    for poly in polys:
        max_deg = max(poly.shape) - 1
        if max_deg < 2:
            polys_copy = polys[:]
            polys_copy.remove(poly)
            new_polys, transform2 = project_down(polys_copy, poly.coeff, approx_tol, solve_tol)
            new_polys = list(map(MultiCheb, new_polys))
            transform3 = lambda x: transform_in(transform2(x))
            ret_polys, transform4, _ = remove_linear(new_polys, approx_tol, solve_tol, transform3)
            return ret_polys, lambda x: transform4(x), True
    else:
        return polys, transform_in, False

def project_down(polys, linear, approx_tol, solve_tol):
    """This function reduces the dimension of a polynomial system when one of
    functions is linear. For polynomials in n variables, it uses an affine
    transformation that maps the (n-1) dimensional hyper-square to cover the
    intersection between the variety of the linear polynomial and the
    n dimensional hyper-square. Then it performs chebyshev interpolation of the
    functions onto this intersection.

    Parameters
    ----------
        polys : list of polynomials
            The polynomials to be projected in a lower dimension by
            interpolated.
        linear : numpy array
            The coefficents of the linear function.

    Returns
    -------
        proj_polys : list
            The projected polynomials.
        T : function from R^(n-1) -> R^n
            This function maps the roots of the projected system to the roots of
            the original system.
    """
    assert all([issubclass(type(p),Polynomial) for p in polys])
    assert isinstance(linear, np.ndarray)

    #  Affine transformation
    #
    # T(x) = Ax + v
    #
    # Maps the (n-1)-dimensional hypercube to the parallelepiped

    p0, edge_vectors = bounding_parallelepiped(linear)
    A = edge_vectors/2
    v = p0 + np.mean(edge_vectors, axis=1)
    T = lambda x: np.dot(A,x.T).T + v

    proj_poly_coeff = []
    for p in polys:
        proj_poly_coeff.append(proj_approximate_nd(p, T))

    if len(polys) > 1:
        from yroots.subdivision import trim_coeffs
        return trim_coeffs(proj_poly_coeff, approx_tol, solve_tol)[0], T
    else:
        return proj_poly_coeff, T

def proj_approximate_nd(f, transform):
    """Finds the chebyshev approximation of an n-dimensional function on the
    affine transformation of hypercube.

    Parameters
    ----------
    f : function from R^n -> R
        The function to project by interpolating.
    transform : function from R^(n-1) -> R^n
        The affine function mapping the hypercube to the desired space.

    Returns
    -------
    coeffs : numpy array
        The coefficient of the chebyshev interpolating polynomial.
    """
    from yroots.subdivision import chebyshev_block_copy
    dim = f.dim
    proj_dim = dim-1
    deg = f.degree
    degs = np.array([deg]*proj_dim)

    cheb_values = np.cos(np.arange(deg+1)*np.pi/deg)
    cheb_grids = np.meshgrid(*([cheb_values]*proj_dim), indexing='ij')

    flatten = lambda x: x.flatten()
    cheb_points = transform(np.column_stack(tuple(map(flatten, cheb_grids))))
    values_block = f(cheb_points).reshape(*([deg+1]*proj_dim))
    values = chebyshev_block_copy(values_block)
    coeffs = np.real(fftn(values/np.product(degs)))

    for i in range(proj_dim):
        #construct slices for the first and degs[i] entry in each dimension
        idx0 = [slice(None)] * proj_dim
        idx0[i] = 0

        idx_deg = [slice(None)] * proj_dim
        idx_deg[i] = degs[i]

        #halve the coefficients in each slice
        coeffs[tuple(idx0)] /= 2
        coeffs[tuple(idx_deg)] /= 2

    slices = []
    for i in range(proj_dim):
        slices.append(slice(0,degs[i]+1))

    return coeffs[tuple(slices)]

def bounding_parallelepiped(linear): #TODO what if things are outside of the evaluable domain?
    """
    A helper function for projecting polynomials. It accepts a linear
    polynomial and return vectors describing an (n-1)-dimensional parallelepiped
    that covers the intersection between the linear polynomial (it's variety)
    and the n-dimensional hypercube.

    Note: The parallelepiped can be described using just one vertex, and (n-1)
    vectors, each of dimension n.

    Second Note: This first attempt is very simple, and can be greatly improved
    by creating a parallelepiped that much more closely surrounds the points.
    Currently, it just makes an nd-rectangle.

    Parameters
    ----------
        linear : numpy array
            The coefficients of the linear function.

    Returns
    -------
        p0 : numpy array
            One vertex of the parallelepiped.
        edges : numpy array
            Array of vectors describing the edges of the parallelepiped, from p0.
    """

    dim = linear.ndim
    coord = np.ones((dim-1,2))
    coord[:,0] = -1
    const = linear[tuple([0]*dim)]
    coeff = np.zeros(dim)
    # flatten the linear coefficients
    for i,idx in enumerate(get_var_list(dim)):
        coeff[i] = linear[idx]

    # get the intersection points with the hypercube edges
    lower = -np.ones(dim)
    upper = np.ones(dim)
    vert = []
    for i in range(dim):
        pts = np.array([(pt[:i]+ (0,) + pt[i:]) for pt in product(*coord)])
        val = -const
        for j,c in enumerate(coeff):
            if i==j:
                continue
            val = val - c*pts[:,j]
        if not np.isclose(coeff[i], 0):
            pts[:,i] = val/coeff[i]
        else:
            pts[:,i] = np.nan
        mask = np.all(lower <= pts, axis=1) & np.all(pts <= upper, axis=1)
        if np.any(mask):
            vert.append(pts[mask])

    # what to do if no intersections
    if len(vert) == 0:
        p0 = -const/np.dot(coeff, coeff)*coeff
        Q, R = np.linalg.qr(np.column_stack([coeff, np.eye(dim)[:,:dim-1]]))
        edges = Q[:,1:]
        return p0, edges

    # do the thing
    vert = np.unique(np.vstack(vert), axis=0)
    v0 = vert[0]
    vert_shift = vert - v0
    Q, vert_flat, _ = qr(vert_shift.T, pivoting=True)
    vert_flat = vert_flat[:-1] # remove flattened dimension
    min_vals = np.min(vert_flat, axis=1)
    max_vals = np.max(vert_flat, axis=1)

    p0 = Q[:,:-1].dot(min_vals) + v0
    edges = Q[:,:-1].dot(np.diag(max_vals-min_vals))
    return p0, edges

def remove_affine_constraint(polys):
    """This function removes linear polynomials from a list by
    picking the variable in each linear polynomial with the coefficient closet
    to one. This function assumes that the polynomials are in Chebyshev form.

    Parameters
    ----------
    polys : list of polynomial objects
        Polynomials to find the common roots of.

    Returns
    -------
    polys : list of polynomial objects
        Polynomials to find the common roots of.
    transform : function
        A function mapping the roots of the output system to the roots of the
        original system.
    """
    #detect which polynomials are linear
    linear = np.where(polydegs == 1)[0]
    nonlinear = np.where(polydegs != 1)[0]

    #remove each linear polynomial
    #for hyperplane in linear:

    #for now, just one linear polynomial
    newpolys = []
    removing_var, lin_combo = pick_removing_var(polys[linear].coeff)
    new_dim = polys[0].dim-1
    #get the projection of the coefficient matrix of each nonlinear polynomial
    for poly_idx in nonlinear:
        coeff = transform_coeff_matrix(polys[poly_idx].coeff, lin_combo, removing_var)
        newpolys.append(MultiCheb(coeff))

    return newpolys, lin_combo, removing_var

def pick_removing_var(coeff):
    """Picks which variable to remove using an affine constraint
    0 = b + c0*x0 + c1*x1 + ... + cn*xn.

    Parameters
    ----------
    coeff : ndarray
        coefficient tensor of the affine constraint/hyperplane

    Returns
    -------
    removing_var : int
        Variable xi to remove
    transform : function
        A function mapping the roots of the output system to the roots of the
        original system.
    lin_combo : ndarray (new_dim,)
        Expression for xi. First entry is the constant term,
        then all of the coefficients of the variables (skipping i).
        In other words, lin_combo is [-b/ci,-c0/ci,-c1/ci,...,-cn/ci]
    """
    olddim = len(coeff.shape)
    #pick linear term with coeff closest to 1 as the removing_var
    lin_combo = coeff[slice(get_var_list(1, olddim, coeff.shape))] #TODO: TEST
    removing_var = np.argmin(np.abs(lin_combo - 1))

    #Turn the affine terms into a representation of
    # xi = b + c0*x0 + c1*x1 + ... + cn*xn.
    lin_combo = list(lin_combo)
    removing_var_coeff = lin_combo.pop(removing_var)
    const_term = coeff[tuple([0]*olddim)] #add in the constant term
    lin_combo = -np.array([const_term] + lin_combo)/removing_var_coeff

    return removing_var, lin_combo

def get_spot(combo,term,Td_idx):
    """Helper function for transform_coeff_matrix. Determines where to update
    the new coefficient tensor for certain products of chebyshev monomials.
    For example, in 2-dimensions, we compute how Td(xi) is represented as a sum
    of Ti(y)Tj(z), but then in order to multiply together Td(xi)Tk(y)Tl(z)
    we have to compute  Ti(y)Tj(z)*Tk(y)Tl(z). Because
    Ti(y)Tk(y) = Ti+k(y)/2 + T|i-k|(y)/2, we have to update 2^new_dim spots
    each time.

    Parameters
    ----------
    combo : list of bools, length new_dim
        which of the 2^new_dim spots we're going to update. For each dimension,
        True indicates that we're updating Ti+k(y)/2 and False indicates that
        we're updating T|i-k|(y)/2.
    term : list or tuple
        which terms in the old coefficient tensor we're multiplying by.
        In the example, this would be Tk(y)Tl(z), represented as [k,l]
    Td_idx : list or tuple
        which terms in the expression of Td(xi) we're multiplying by.
        In the example, this would be Td(xi) = Ti(y)Tj(z), represented as [i,j]

    Returns
    -------
    spot : list of bools
        index in the new coefficient tensor to update

    """
    spot = []
    for l,is_plus in enumerate(combo):
        if is_plus:
            spot.append(term[l] + Td_idx[l])
        else:
            spot.append(abs(term[l] - Td_idx[l]))
    return tuple(spot)

def transform_coeff_matrix(oldcoeff,lin_combo,removing_var):
    """Transforms a coefficient tensor into a lower dimensional coefficient tensor.

    Parameters
    ----------
    oldcoeff : ndarray
        coefficient tensor of the old polynomial
    lin_combo : ndarray (new_dim,)
        Expression for xi. First entry is the constant term,
        then all of the coefficients of the variables (skipping i).
        In other words, lin_combo is [-b/ci,-c0/ci,-c1/ci,...,-cn/ci]
    removing_var : int
        Variable xi to remove

    Returns
    -------
    newcoeff : ndarray
        coefficient tensor of the new polynomial
    """
    #make matrix for storing the new coefficient
    #also get the max degree of removing_var in the oldcoeff
    starter_slice_shape = list(oldcoeff.shape)
    max_deg_removing_var = starter_slice_shape.pop(removing_var)
    starter_slice_to = tuple([slice(0,n,None) for n in starter_slice_shape])
    newcoeff_shape = [deg-1 + max_deg_removing_var for deg in starter_slice_shape]
    newcoeff = np.zeros(newcoeff_shape)
    #find expressions for Td(xi) for each d up to max_deg_removing_var
    Td,new_dim = get_Td_expressions(lin_combo,max_deg_removing_var)

    #we start off with the face that was not changed by mutliplication
    getAll = slice(None,None,None)
    starter_slice_from = tuple([getAll]*(removing_var)+[slice(1)]+[getAll]*(len(oldcoeff.shape)-removing_var-1))
    newcoeff[starter_slice_to] = oldcoeff[starter_slice_from].reshape(starter_slice_shape)
    # other_nonzero_terms_slice = tuple([slice(None,None,None)]*(removing_var-1)+[slice(1,None,None)]+[slice(None,None,None)]*(len(oldcoeff.shape)-removing_var))
    for term in zip(*np.where(oldcoeff!=0)):
        #we've already accounted for that face
        if term[removing_var] == 0:
            continue
        #take off the removing_var degree of the term
        coeff = oldcoeff[term]
        term = list(term)
        d = term.pop(removing_var)
        for Td_idx in mon_combos([0]*new_dim,d):
            increment = Td[d][tuple(Td_idx)] * coeff / 2**new_dim
            #True means i+k, false means abs(i-k)
            for combo in product(*[[True,False]]*new_dim): #TODO: faster sparsity stuff there
                spot = get_spot(combo,term,Td_idx)
                newcoeff[spot] += increment
    return newcoeff

def get_Td_expressions(lin_combo,maxdeg):
    """This function expresses Td(xi) as a Chebyshev-form polynomial
    in the other variables given that 0 = b + c0*x0 + c1*x1 + ... + cn*xn.

    Parameters
    ----------
    lin_combo : ndarray (new_dim,)
        Expression for xi. First entry is the constant term,
        then all of the coefficients of the variables (skipping i).
        In other words, lin_combo is [-b/ci,-c0/ci,-c1/ci,...,-cn/ci]
    maxdeg : int
        Maximum degree to compute Td(xi) for

    Returns
    -------
    Td : dictionary of coefficient arrays
        Td[d] is the coefficient array of Td(xi) in terms of x1, ..., xn
        (skipping i).
    """
    new_dim = len(lin_combo) - 1

    Td = {}

    #T0(xi)
    Td[0] = np.zeros([maxdeg+1]*new_dim)
    Td[0][tuple([0]*new_dim)] = 1

    #T1(xi)
    Td[1] = np.zeros([maxdeg+1]*new_dim)
    Td[1][tuple([0]*new_dim)] = lin_combo[0]
    varlist = get_var_list(new_dim)
    for i,var in enumerate(varlist):
        Td[1][var] = lin_combo[i+1]

    for n in range(1,maxdeg):
        #Td+1(xi) = -Td-1(lin_combo) + lin_combo[0] * Td(lin_combo)
        # + lin_combo[1] x2 * Td(lin_combo) + lin_combo[2] x3 * Td(lin_combo) + ...
        Td[n+1] = -Td[n-1] + 2*lin_combo[0] * Td[n]

        #breaks into cases for array slicing purposes
        if new_dim > 1:
            for j,var in enumerate(varlist):
                #slicers in the jth direction for moving coefficient tensors around
                sliceupdown_from = []
                sliceup_to = []
                slicedown_to = []
                sliceface_from = []
                sliceface_to = []
                for k in range(new_dim):
                    if j == k:
                        sliceupdown_from.append(slice(1,-1,None))
                        sliceup_to.append(slice(2,None,None))
                        slicedown_to.append(slice(None,-2,None))
                        sliceface_from.append(slice(1))
                        sliceface_to.append(slice(1,2,None))
                    else:
                        sliceupdown_from.append(slice(None,None,None))
                        sliceup_to.append(slice(None,None,None))
                        slicedown_to.append(slice(None,None,None))
                        sliceface_from.append(slice(None,None,None))
                        sliceface_to.append(slice(None,None,None))
                Td[n+1][tuple(sliceup_to)] += lin_combo[j+1]*Td[n][tuple(sliceupdown_from)]
                Td[n+1][tuple(slicedown_to)] += lin_combo[j+1]*Td[n][tuple(sliceupdown_from)]
                Td[n+1][tuple(sliceface_to)] += 2*lin_combo[j+1]*Td[n][tuple(sliceface_from)]
        else:
            Td[n+1][1:] += 2*lin_combo[1]*Td[n][:-1]

    return Td, new_dim

def rref(A):
    """Reduce the square matrix A to REF with full pivoting.
    Parameters:
        A ((n,n) ndarray): The matrix to be reduced.
    Returns:
        ((n,n) ndarray): The RREF of A.
        ((n,) ndarray): The row pivoting array.
        ((n,) ndarray): The column pivoting array.
    """
    A = np.array(A, dtype=np.float, copy=True)
    m,n = A.shape
    Pr = np.arange(m)
    Pc = np.arange(n)
    for j in range(m):
        row,col = np.where(np.abs(A[j:,j:-1])==np.abs(A[j:,j:-1]).max())
        k,l = row[0]+j,col[0]+j
        A[j],A[k] = A[k],A[j]
        Pr[j],Pr[k] = Pr[k],Pr[j]
        A[:,j],A[:,l] = A[:,l],A[:,j]
        Pc[j],Pc[l] = Pc[l],Pc[j]
        for i in range(m):
            if i != j:
                A[i,j:] -= A[j,j:] * A[i,j] / A[j,j]
        A[j] = A[j]/A[j,j]
    return A,Pr,Pc
