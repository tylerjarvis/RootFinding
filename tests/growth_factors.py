"""
Computes the growth factors of random quadratics in dimensions
2-10
"""
from devastating_example_test_scripts import *
from yroots.utils import condeigs, newton_polish
from yroots.polyroots import solve
from yroots.Multiplication import *
import yroots as yr
import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
import sys
from matplotlib import ticker as mticker
from scipy.stats import linregress
from matplotlib.patches import Patch

macheps = 2.220446049250313e-16

def devestating_growth_factors(dims,eps,kind,newton,N=50,just_dev_root=True,seed=468,perturb_eps=0,save=True):
    """Computes the growth factors of a system of polynomails.

    Parameters
    ----------
    dims : list of ints
        The dimensions to test
    eps : float
        epsilon value for the devestating example
    kind : string
        the type of devestating example system. One of 'power', 'spower',
        'cheb', and 'chebs'.
    newton : bool
        whether or not to newton polish the roots
    N : int or list
        number of tests to run in each dimension
    just_dev_root : bool
        If true, only returns growth factors for the devestating root.
        Otherwise, returns growth factors for all roots.
    seed : int
        random seed to use in generating the systems
    perturb_eps : float
        the amount by which to perturb the system
    save : bool
        whether to save and return the results or just return them
    returns
    -------
    growth factors: (dim, N, num_roots) or (dim, N) array
        Array of growth factors. The [i,j] spot is  the growth factor for
        the i'th coordinate in the j'th test system.
    """
    print('Devestating Example in dimensions',dims)
    np.random.seed(seed)
    if isinstance(N,int):
        N = [N]*len(dims)
    gfs = dict()
    if kind in ['power','cheb']: shifted = False
    else: shifted = True
    for n,dim in zip(N,dims):
        if save:
            if newton: folder = 'growth_factors/dev/newton/dim{}/'.format(dim)
            else:      folder = 'growth_factors/dev/nopol/dim{}/'.format(dim)
        print(dim)
        gf = []
        for _ in range(n):
            #get a random devestating example
            polys = randpoly(dim,eps,kind)
            if perturb_eps > 0:
                perturb(polys,perturb_eps)
            growth_factor = growthfactor(polys,dim,newton,dev=just_dev_root,shifted=shifted)
            print(_+1,'done')
            gf.append(growth_factor)
            if save:
                np.save(folder+'deg2_sys{}.npy'.format(_),gf)
                print(_+1,'saved')
        gfs[dim] = np.array(gf)
        if save: np.save(folder+'deg2.npy',gfs[dim])
    return gfs

def growthfactor(polys,dim,newton,dev=False,shifted=None):
    """Computes the growth factors of a system of polynomails.

    Parameters
    ----------
    polys : list of polynomial objects
        Polynomial system
    dim : int
        dimension of the polynomials
    newton : bool
        whether or not to newton polish the roots
    dev : bool
        whether or not we are computing the growth factor for a devestating
        example system, in which case we want to use the root at the origin
    shifted : bool
        for devestating systems, whether the system is
    returns
    -------
    growth factors: (dim, num_roots) array
        Array of growth factors. The [i,j] spot is  the growth factor for
        the i'th coordinate of the j'th root.
    """
    roots,M = solve(polys)
    if newton:
        dist_between_roots = la.norm(roots[:,np.newaxis]-roots,axis=2)
        smallest_dist_between_roots = np.min(dist_between_roots[np.nonzero(dist_between_roots)])
        newroots = np.array([newton_polish(polys,root,tol=10*macheps) for root in roots])
        max_diff = np.max(np.abs(newroots-roots))
        roots = newroots
    #find the growth factors for all the roots
    if not dev:
        eig_conds = np.empty((dim,len(roots)))
        for d in range(dim):
            M_ = M[...,d]
            #TODO think about that more carefully... polish numerator too or can we just use the polished roots?
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
            root_cond = 1/S[-1]
            root_conds[i] = root_cond
        #compute the growth factors
        factors = eig_conds / root_conds
        if newton: return factors, max_diff, smallest_dist_between_roots
        else: return factors
    #only find the growth factor for the root at the origin
    else:
        #find the root at the origin
        if shifted:
            dev_root = np.ones(dim)
        else:
            dev_root = np.zeros(dim)
        idx = np.argmin(la.norm(roots-dev_root,axis=1))

        eig_conds = np.empty(dim)
        for d in range(dim):
            M_ = M[...,d]
            vals, vecR = la.eig(M_)
            eig_conds_curr = condeigs(M_,vals,vecR)
            arr = sort_eigs(vals,roots[:,d])
            vals = vals[arr]
            eig_conds[d] = eig_conds_curr[arr][idx]
        factors = np.empty(len(roots))
        #compute the condition numbers of the roots
        J = np.empty((dim,dim),dtype='complex')
        for j,poly in enumerate(polys):
            J[j] = poly.grad(roots[idx])
        S = la.svd(J,compute_uv=False)
        root_cond = 1/S[-1]
        #compute the growth factors
        factors = eig_conds / root_cond
        if newton: return factors, max_diff, smallest_dist_between_roots
        else: return factors

def get_growth_factors(coeffs, newton, save=True):
    """Computes the growth factors of a bunch of systems of polynomails.

    Parameters
    ----------
    coeffs : (N,dim,deg,deg,...) array
        Coefficient tensors of N test systems. Each test system should have dim
        polynomial systems of degree deg
    newton : bool
        whether or not to newton polish the roots
    save : bool
        whether or not to save and return the results or just return them
    returns
    -------
    growth factors: (N, dim, deg^dim) array
        Array of growth factors. The [k,i,j] spot is  the growth factor for
        the i'th coordinate of the j'th root of the k'th system
    """
    N,dim = coeffs.shape[:2]
    deg = 2
    print((N,dim,deg**dim))
    not_full_roots = np.zeros(N,dtype=bool)
    gfs = [0]*N
    if save:
        if newton: folder = 'growth_factors/rand/newton/dim{}/'.format(dim)
        else:      folder = 'growth_factors/rand/nopol/dim{}/'.format(dim)
    for i,system in enumerate(coeffs):
        polys = [yr.MultiPower(c) for c in system]
        gf = growthfactor(polys,dim,newton)
        if newton:

            gf,max_diff,smallest_dist_between_roots = gf
            if not 10*max_diff < smallest_dist_between_roots:
                print('**Potentially converging roots with polishing**')
                print('\tNewton changed roots by at most: {}'.format(max_diff))
                print('\tDist between root was at least:  {}'.format(smallest_dist_between_roots))
        #only records if has the right number of roots_sort
        #TODO: why do some systems not have enough roots?
        print(i+1,'done')
        if gf.shape[1] == deg**dim:
            gfs[i] = gf
            if save: np.save(folder+'deg2_sys{}.npy'.format(i),gf)
        else:
            not_full_roots[i] = True
            if save: np.save(folder+'not_full_roots_deg2.npy',not_full_roots)
        if save: print(i+1,'saved')
    #final save at the end
    if save: np.save(folder+'deg2_res.npy',gfs)
    print('saved all results')
    return gfs

def plot(datasets,labels=None,filename='growth_factor_plot',digits_lost=False,figsize=(6,4),dpi=400,best_fit=True):
    """
    Plots growth factor data.

    Parameters
    ----------
    datasets : list of dictionaries
        Growth factor datasets to plot. Each dataset dictionary should be
        formatted to map dimension to an array of growth factors
    digits_lost : bool
        whether the y-axis should be a log scale of the growth factors
        or a linear scale of the digits lost
    figsize : tuple of floats
        figure size
    dpi : int
        dpi of the image
    """
    dims = 2+np.arange(np.max([len(data.keys()) for data in datasets]))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize,dpi=dpi)
    ax.yaxis.grid(color='gray',alpha=.15,linewidth=1,which='major')
    def plot_dataset(data,color):
        pos = 2+np.arange(len(data))
        #log before plot
        data_log10 = [np.log10(data[d].flatten()) for d in data.keys()]
        #violins
        parts = ax.violinplot(data_log10,
                      positions=pos,
                      widths=.8,
                      points=1000,
                      showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(.3)
        #boxplots
        maxs = [np.max(g) for g in data_log10]
        mins = [np.min(g) for g in data_log10]
        ax.hlines(maxs,pos-.02,pos+.02,lw=1)
        ax.hlines(mins,pos-.02,pos+.02,lw=1)
        ax.vlines(pos,mins,maxs,lw=.5,linestyles='dashed')
        box_props = dict(facecolor='w')
        median_props = dict(color=color)
        ax.boxplot(data_log10,positions=pos,
                   vert=True,
                   showfliers=False,
                   patch_artist=True,
                   boxprops=box_props,
                   widths=.35,
                   medianprops=median_props)
        if best_fit:
            points = np.array([[d,val] for i,d in enumerate(data.keys()) for val in data_log10[i]])
            slope, intercept = linregress(points)[:2]
            print(slope, intercept)
            ax.plot(pos,pos*slope+intercept,c=color)
    for i,dataset in enumerate(datasets):
        plot_dataset(dataset,f'C{i}')
    # ax.plot(dims,dims-1,c='C1',label=r'Theoretical Devestating Example, $\epsilon=.1$')
    # ax.plot(dims,2*(dims-1),c='C2',label=r'Theoretical Devestating Example, $\epsilon=.01$')
    # ax.plot(dims,3*(dims-1),c='C3',label=r'Theoretical Devestating Example, $\epsilon=.001$')
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
    legend_elements = [Patch(facecolor=f'C{i}') for i in range(len(datasets))]
    ax.legend(legend_elements,labels)
    ax.set_title('Growth Factors of Quadratic Polynomial Systems')
    plt.savefig(fname=filename+'.pdf',bbox_inches='tight',dpi=dpi,format='pdf')
    plt.show()

if __name__ == "__main__":
    #INPUT FORMAT test_type--dev or rand; newton_polish--newton or nopol; dims-- dimensions to run in
    input = sys.argv[1:]
    test = input[0]
    if input[1] == 'newton':
        newton = True
    elif input[1] == 'nopol':
        newton=False
    else:
        raise ValueError("2nd input must be one of 'newton' for polishing or 'nopol' for no polishing")
    dims = [int(i) for i in input[2:]]
    if test == 'rand':
        for dim in dims:
            coeffs = np.load('random_tests/coeffs/dim{}_deg2_randn.npy'.format(dim))
            gfs = get_growth_factors(coeffs, newton)
    elif test == 'dev':
        eps = .1
        kind = 'power'
        N = 50
        devestating_growth_factors(dims,eps,kind,newton,N=N)
    else:
        raise ValueError("1st input must be one of 'rand' for random polys 'dev' for devestating example")
