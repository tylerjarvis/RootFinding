import numpy as np
from numpy.fft import fftn
from yroots.utils import get_var_list, mon_combos
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

# def remove_affine_constraint(polys):
#     """This function removes linear polynomials from a list by
#     picking the variable in each linear polynomial with the coefficient closet
#     to one. This function assumes that the polynomials are in Chebyshev form.
#
#     Parameters
#     ----------
#     polys : list of polynomial objects
#         Polynomials to find the common roots of.
#
#     Returns
#     -------
#     polys : list of polynomial objects
#         Polynomials to find the common roots of.
#     transform : function
#         A function mapping the roots of the output system to the roots of the
#         original system.
#     """
#     #detect which polynomials are linear
#     linear = np.where(polydegs == 1)[0]
#     nonlinear = np.where(polydegs != 1)[0]
#     power = is_power[polys[0]]
#
#
#     #remove each linear polynomial
#     #for hyperplane in linear:
#
#     #for now, just one linear polynomial
#     newpolys = []
#     removing_var = get_removing_var(hyperplane.coeff)
#     new_dim = polys[0].dim-1
#     transform, lin_combo = get_transform(hyperplane.coeff, removing_var)
#     #find the maximum degree of the polynomial in terms of removing_var
#     maxdeg = np.max([poly.coeff.shape[removing_var] for poly in polys])
#     Td = get_Td_expressions(lin_combo,maxdeg)
#     #get the projection of the coefficient matrix of each nonlinear polynomial
#     for poly in nonlinear:
#         coeff = transform_coeff_matrix(poly.coeff, lin_combo, removing_var)
#         newpolys.append(MultiCheb(coeff))
#
#     return newpolys

# def get_removing_var():

# def get_transform():

def transform_coeff_matrix(coeff,Td,removing_var):
    #matrix for storing the new coefficient
    newcoeff_shape = list(coeff.shape)
    del newcoeff_shape[removing_var]
    newcoeff = np.zeros(newcoeff_shape)
    #True means i+k, false means abs(i-k)
    for i in product(*[[True,False]]*4):
        print(bin(i))

def get_Td_expresssions(lin_combo,maxdeg):
    """This function expresses Td(xi) as a Chebyshev-form polynomial
    in the other variables given that xi = b + c0*x0 + c1*x1 + ... + cn*xn.

    Parameters
    ----------
    lin_combo : list of floats
        Expression for xi. First entry should be the constant term,
        then all of the coefficients of the variables (skipping i).
        In other words, lin_combo = [b,c0,c1,...cn]
    maxdeg : int
        Maximum degree to compute Td(xi) in

    Returns
    -------
    Td : dictionary of coefficient arrays
        Td[d] is the coefficient array of Td(xi) in terms of x1, ..., xn
        (skipping i).
    """
    new_dim = len(lin_combo) - 1

    Td = {}

    #T0
    Td[0] = np.zeros([maxdeg+1]*new_dim)
    Td[0][tuple([0]*new_dim)] += 1

    #T1
    Td[1] = np.zeros([maxdeg+1]*new_dim)
    Td[1][tuple([0]*new_dim)] += lin_combo[0]
    for i,var in enumerate(get_var_list(new_dim)):
        Td[1][var] += lin_combo[i+1]

    for n in range(1,maxdeg):
        #Td+1 = Td-1(lin_combo) + lin_combo[0] * Td(lin_combo)
        # + lin_combo[1] x2 * Td(lin_combo) + lin_combo[2] x3 * Td(lin_combo) + ...
        Td[n+1] = Td[n-1] + lin_combo[0] * Td[n]

        #breaks into cases for array slicing purposes
        if new_dim > 1:
            for j,var in enumerate(get_var_list(new_dim)):
                #slicers for [1:] and [:-1] in the jth direction
                slice1 = []
                slice_neg1 = []
                for k in range(new_dim):
                    if j == k:
                        slice1.append(slice(1,None,None))
                        slice_neg1.append(slice(None,-1,None))
                    else:
                        slice1.append(slice(None,None,None))
                        slice_neg1.append(slice(None,None,None))
                Td[n+1][tuple(slices1)] += lin_combo[j+1]*Td[n][tuple(slices_neg1)]
        else:
            Td[n+1][1:] += lin_combo[1]*Td[n][:-1]

    return Td
