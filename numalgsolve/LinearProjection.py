from itertools import product

import numpy as np
from numpy.fft.fftpack import fftn
from numalgsolve.utils import get_var_list
from numalgsolve.polynomial import MultiCheb, MultiPower, Polynomial
from numalgsolve.subdivision import interval_approximate_nd, trim_coeff,\
                                    chebyshev_block_copy
from scipy.linalg import qr

def project_down(polys, linear):
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

    return proj_poly_coeff, T


def proj_approximate_nd(f,transform):
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
    dim = f.dim
    proj_dim = dim-1
    deg = f.degree
    degs = np.array([deg]*proj_dim)

    # assert hasattr(f,"evaluate_grid")
    # dang, we don't get to use evaluate_grid here

    cheb_values = np.cos(np.arange(deg+1)*np.pi/deg)
    cheb_grids = np.meshgrid(*([cheb_values]*proj_dim), indexing='ij')

    flatten = lambda x: x.flatten()
    cheb_points = transform(np.column_stack(map(flatten, cheb_grids)))
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

    return trim_coeff(coeffs[tuple(slices)])

def bounding_parallelepiped(linear):
    """
    A helper function for projecting polynomials. It accepts a linear
    polynomial and return vectors describing an (n-1)-dimensional parallelepiped
    that covers the intersection between the linear polynomial (it's variety)
    and the n-dimensional hypercube.

    Note: The parallelepiped can be described using just one vertex, and (n-1)
    vectors, each of dimension n.

    Second Note: This first attempt is very simple, and can be greatly improved
    by creating a parallelepiped that much more closely surrounds the points.
    Currently, it just makes a rectangle.

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
        for pt in product(*coord):
            val = -const
            skipped = 0
            for j,c in enumerate(coeff):
                if i==j:
                    skipped = 1
                    continue
                val -= c*pt[j-skipped]
            one_point = list(pt)
            if not np.isclose(coeff[i], 0):
                one_point.insert(i,val/coeff[i])
            one_point = np.array(one_point)
            if np.all(lower <= one_point) and np.all(one_point <= upper):
                vert.append(one_point)

    # what to do if no intersections
    if len(vert) == 0:
        p0 = -const/np.dot(coeff, coeff)*coeff
        Q, R = np.linalg.qr(np.column_stack([coeff, np.eye(dim)[:,:dim-1]]))
        edges = Q[:,1:]
        return p0, edges
        # raise Exception("What do I do!?")

    # do the thing
    vert = np.unique(np.array(vert), axis=0)
    v0 = vert[0]
    vert_shift = vert - v0
    Q, vert_flat, _ = qr(vert_shift.T, pivoting=True)
    vert_flat = vert_flat[:-1] # remove flattened dimension
    min_vals = np.min(vert_flat, axis=1)
    max_vals = np.max(vert_flat, axis=1)

    p0 = Q[:,:-1].dot(min_vals) + v0
    edges = Q[:,:-1].dot(np.diag(max_vals-min_vals))
    return p0, edges
