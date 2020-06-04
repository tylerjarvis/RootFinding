"""
Computes the growth factors of random quadratics in dimensions
2-10
"""
from yroots.utils import condeigs
from yroots.polyroots import solve
from yroots.Multiplication import *
import yroots as yr
import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
import sys
from matplotlib import ticker as mticker

def msmatpolys(polys, max_cond_num=1.e6, verbose=False, return_all_roots=True,method='svd'):
    '''
    Returns the MS matrices of the system of polynomials

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
    M : (deg,deg,dim) ndarray
        The moller stetter matrices of the system
    Raises
    ------
    ConditioningError if reduce_macaulay() raises a ConditioningError.
    TooManyRoots if the macaulay matrix returns more roots than the Bezout bound.
    '''
    #We don't want to use Linear Projection right now

    if len(polys) == 1:
        from yroots.OneDimension import solve
        return transform(solve(polys[0], MSmatrix=0))
    poly_type = is_power(polys, return_string = True)
    dim = polys[0].dim

    #By Bezout's Theorem. Useful for making sure that the reduced Macaulay Matrix is as we expect
    degrees = [poly.degree for poly in polys]
    max_number_of_roots = np.prod(degrees)

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
        if method == 'qrt':
            try:
                E,Q,cond,cond_back = reduce_macaulay_qrt(matrix,cut,max_cond_num)
            except ConditioningError as e:
                raise e
        elif method == 'tvb':
            try:
                E,Q,cond,cond_back = reduce_macaulay_tvb(matrix,cut,max_cond_num)
            except ConditioningError as e:
                raise e
        elif method == 'svd':
            try:
                E,Q,cond,cond_back = reduce_macaulay_svd(matrix,cut,max_cond_num)
            except ConditioningError as e:
                raise e

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
    return M

def growthfactor(polys,dim):
    """Computes the growth factors of a system of polynomails.

    Parameters
    ----------
    polys : list of polynomial objects
        Polynomial system
    dim : int
        dimension of the polynomials
    returns
    -------
    growth factors: (dim, num_roots) array
        Array of growth factors. The [i,j] spot is  the growth factor for
        the i'th coordinate of the j'th root.
    """
    roots = solve(polys,verbose=True)
    #run multiplication but just get the tensor of MS matrices
    M = msmatpolys(polys)
    eig_conds = np.empty((dim,len(roots)))
    for d in range(dim):
        M_ = M[...,d]
        vals, vecR = la.eig(M_)
        eig_conds[d] = condeigs(M_,vals,vecR)
        arr = sort_eigs(vals,roots[:,d])
        vals = vals[arr]
        eig_conds[d] = eig_conds[d][arr]
    factors = np.empty(len(roots))
    #compute the condition numbers of the roots
    root_conds = np.empty(len(roots))
    for i,root in enumerate(roots):
        J = np.empty((dim,dim),dtype='complex')
        for j,poly in enumerate(polys):
            J[j] = poly.grad(root)
        S = la.svd(J,compute_uv=False)
        root_cond = S[-1]
        root_conds[i] = root_cond
    #compute the growth factors
    factors = eig_conds / root_conds
    return factors

def get_growth_factors(coeffs):
    """Computes the growth factors of a bunch of systems of polynomails.

    Parameters
    ----------
    coeffs : (N,dim,deg,deg,...) array
        Coefficient tensors of N test systems. Each test system should have dim
        polynomial systems of degree deg
    returns
    -------
    growth factors: (N, dim, deg^dim) array
        Array of growth factors. The [k,i,j] spot is  the growth factor for
        the i'th coordinate of the j'th root of the k'th system
    """
    N,dim = coeffs.shape[:2]
    deg = 2
    gfs = np.zeros((N,dim,deg**dim))
    print(gfs.shape)
    for i,system in enumerate(coeffs):
        polys = [yr.MultiPower(c) for c in system]
        gf = growthfactor(polys,dim)
        #only records if has the right number of roots_sort
        #TODO: why do some systems not have enough roots?
        if gf.shape[1] == deg**dim:
            gfs[i] = gf
        if i%10 == 9:
            print(i+1,'done')
    return gfs

def plot(gf_data,digits_lost=False,figsize=(6,4),dpi=200):
    """
    Plots the growth factors of .

    Parameters
    ----------
    gf_data : list of arrays
        arrays of growth factors, which must be flattened
    digits_lost : bool
        whether the y-axis should be a log scale of the growth factors
        or a linear scale of the digits lost
    figsize : tuple of floats
        figure size
    dpi : int
        dpi of the image
    """
    dims = 2+np.arange(len(gf_data))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize,dpi=dpi)

    gfs_log10 = [np.log10(g) for g in gf_data]
    #log before plot
    ax.violinplot(gfs_log10,
                  positions=dims,
                  widths=.9,
                  points=1000,
                  showextrema=False)
    maxs = [np.max(g) for g in gfs_log10]
    mins = [np.min(g) for g in gfs_log10]
    alpha = .25
    ax.hlines(maxs,dims-.02,dims+.02,lw=1,alpha=alpha)
    ax.hlines(mins,dims-.02,dims+.02,lw=1,alpha=alpha)
    ax.vlines(dims,mins,maxs,lw=.5,linestyles='dashed',alpha=alpha)
    box_props = dict(facecolor='w')
    median_props = dict(color='r')
    ax.boxplot(gfs_log10,positions=dims,
               vert=True,
               showfliers=False,
               patch_artist=True,
               boxprops=box_props,
               medianprops=median_props)
    ax.plot(dims,dims,c='g',label=r'Devestating Example, $\epsilon=.1$')
    max_y_lim = max(dims)+1
    ax.set_title('Growth Factors of Quadratic Polynomial Systems')
    if digits_lost:
        ax.set_ylim(-1,max_y_lim)
        ax.set_ylabel('Digits Lost')
    else:
        ax.set_ylabel('Growth Factor')
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
        ax.yaxis.set_ticks([np.log10(x) for p in range(-1,max_y_lim)
                               for x in np.linspace(10**p, 10**(p+1), 10)], minor=True)
    ax.set_xlabel('Dimension')
    ax.legend()
    ax.set_title('Growth Factors of Quadratic Polynomial Systems')
    plt.show()

if __name__ == "__main__":
    dims = sys.argv[1:]
    for dim in dims:
        coeffs = np.load(f'random_tests/coeffs/dim{dim}_deg2_randn.npy')
        gfs = get_growth_factors(coeffs)
        not_full_roots = np.unique(np.where(gfs == 0)[0])
        np.save(f'growth_factors/gfs_deg2_dim{dim}.npy',gfs)
        np.save(f'growth_factors/not_full_roots_deg2_dim{dim}.npy',not_full_roots)
