"""
Computes the conditioning ratios of random quadratics in dimensions
2-10
"""
from .devastating_example_test_scripts import *
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

def devestating_conditioning_ratios(dims,eps,kind,newton,N=50,just_dev_root=True,
                                seed=468,perturb_eps=0,save=True,verbose=0):
    """Computes the conditioning ratios of a system of polynomails.

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
        If true, only returns conditioning ratios for the devestating root.
        Otherwise, returns conditioning ratios for all roots.
    seed : int
        random seed to use in generating the systems
    perturb_eps : float
        the amount by which to perturb the system
    save : bool
        whether to save and return the results or just return them
    verbose : int (default 0)
        the level of verbosity
    returns
    -------
    conditioning ratios: (dim, N, num_roots) or (dim, N) array
        Array of conditioning ratios. The [i,j] spot is  the conditioning ratio for
        the i'th coordinate in the j'th test system.
    """
    if verbose>0:print('Devestating Example in dimensions',dims)
    np.random.seed(seed)
    if isinstance(N,int):
        N = [N]*len(dims)
    crs = dict() #conditioning ratios dictionary
    if kind in ['power','cheb']: shifted = False
    else: shifted = True
    for n,dim in zip(N,dims):
        if save:
            if newton: folder = 'conditioning_ratios/dev/newton/dim{}/'.format(dim)
            else:      folder = 'conditioning_ratios/dev/nopol/dim{}/'.format(dim)
        if verbose>0:print('Dimension', dim)
        cr = []
        for _ in range(n):
            #get a random devestating example
            polys = randpoly(dim,eps,kind)
            if verbose>2: print('System Coeffs',*[p.coeff for p in polys],sep='\n')
            if perturb_eps > 0:
                polys = perturb(polys,perturb_eps)
            conditioning_ratio = conditioningratio(polys,dim,newton,dev=just_dev_root,shifted=shifted,verbose=verbose>1)
            if newton:
                conditioning_ratio, max_diff, smallest_dist_between_roots = conditioning_ratio
                if 10*max_diff >= smallest_dist_between_roots:
                    print('**Potentially converging roots with polishing**')
                    print('\tNewton changed roots by at most: {}'.format(max_diff))
                    print('\tDist between root was at least:  {}'.format(smallest_dist_between_roots))
            if verbose>0:print(_+1,'done')
            cr.append(conditioning_ratio)
            if save:
                np.save(folder+'deg2_sys{}.npy'.format(_),cr)
                if verbose>0:print(_+1,'saved')
        crs[dim] = np.array(cr)
        if save: np.save(folder+'deg2.npy',crs[dim])
    return crs

def conditioningratio(polys,dim,newton,dev=False,shifted=None,root=None,verbose=False):
    """Computes the conditioning ratios of a system of polynomails.

    Parameters
    ----------
    polys : list of polynomial objects
        Polynomial system
    dim : int
        dimension of the polynomials
    newton : bool
        whether or not to newton polish the roots
    dev : bool
        whether or not we are computing the conditioning ratio for a devestating
        example system, in which case we want to use the root at the origin
    shifted : bool
        for devestating systems, whether the system is
    root : 1d nparray
        optional parameter for when you know the actual
        root you want to find the conditioning ratio of
    returns
    -------
    conditioning ratios: (dim, num_roots) array
        Array of conditioning ratios. The [i,j] spot is  the conditioning ratio for
        the i'th coordinate of the j'th root.
    """
    roots,M = solve(polys,max_cond_num=np.inf,verbose=verbose)
    if newton:
        dist_between_roots = la.norm(roots[:,np.newaxis]-roots,axis=2)
        smallest_dist_between_roots = np.min(dist_between_roots[np.nonzero(dist_between_roots)])
        newroots = np.array([newton_polish(polys,root,tol=10*macheps) for root in roots])
        max_diff = np.max(np.abs(newroots-roots))
        roots = newroots
    #find the conditioning ratios for all the roots
    if root is not None:
        #find the root
        idx = np.argmin(la.norm(roots-root,axis=1))

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
            J[j] = poly.grad(root)
        S = la.svd(J,compute_uv=False)
        root_cond = 1/S[-1]
        # print(np.log10(root_cond))
        # print(np.log10(eig_conds))
        #compute the conditioning ratios
        factors = eig_conds / root_cond
        if newton: return factors, max_diff, smallest_dist_between_roots
        else: return factors
    elif not dev:
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
        #compute the conditioning ratios
        factors = eig_conds / root_conds
        if newton: return factors, max_diff, smallest_dist_between_roots
        else: return factors
    #only find the conditioning ratio for the root at the origin
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
        #compute the conditioning ratios
        factors = eig_conds / root_cond
        if newton: return factors, max_diff, smallest_dist_between_roots
        else: return factors

def get_conditioning_ratios(coeffs, newton, save=True):
    """Computes the conditioning ratios of a bunch of systems of polynomails.

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
    conditioning ratios: (N, dim, deg^dim) array
        Array of conditioning ratios. The [k,i,j] spot is  the conditioning ratio for
        the i'th coordinate of the j'th root of the k'th system
    """
    N,dim = coeffs.shape[:2]
    deg = 2
    print((N,dim,deg**dim))
    not_full_roots = np.zeros(N,dtype=bool)
    crs = [0]*N
    if save:
        if newton: folder = 'conditioning_ratios/rand/newton/dim{}/'.format(dim)
        else:      folder = 'conditioning_ratios/rand/nopol/dim{}/'.format(dim)
    for i,system in enumerate(coeffs):
        polys = [yr.MultiPower(c) for c in system]
        cr = conditioningratio(polys,dim,newton)
        if newton:

            cr,max_diff,smallest_dist_between_roots = cr
            if not 10*max_diff < smallest_dist_between_roots:
                print('**Potentially converging roots with polishing**')
                print('\tNewton changed roots by at most: {}'.format(max_diff))
                print('\tDist between root was at least:  {}'.format(smallest_dist_between_roots))
        #only records if has the right number of roots_sort
        #TODO: why do some systems not have enough roots?
        print(i+1,'done')
        if cr.shape[1] == deg**dim:
            crs[i] = cr
            if save: np.save(folder+'deg2_sys{}.npy'.format(i),cr)
        else:
            not_full_roots[i] = True
            if save: np.save(folder+'not_full_roots_deg2.npy',not_full_roots)
        if save: print(i+1,'saved')
    #final save at the end
    if save:
        np.save(folder+'deg2_res.npy',crs)
        print('saved all results')
    return crs

'''functions to generate random systems that almost have double roots.

get_scalar, get_coeff and get_MultiPower can be used to find a hyperellipse/hyperbola
with pre-chosen roots.

the rest of the functions use specially chosen roots to generate examples.
'''
def get_scalar(center,roots):
    'solves for the scalars in the conic equation. see conditioning_ratios.ipynb for details'
    dim = roots.shape[1]
    return la.solve((roots - center)**2,np.ones(dim))

def get_coeff(center,roots):
    """
    finds the coefficient tensor of the hyperellipses/hyperbolas with specified center
    and roots
    """
    scalar = get_scalar(center,roots)
    dim = len(center)
    coeff = np.zeros([3]*dim)
    spot = [0]*dim
    coeff[tuple(spot)] = np.sum(scalar*center**2)-1
    for var,c,s in zip(range(dim),center,scalar):
        #linear
        spot[var] = 1
        coeff[tuple(spot)] = -2*s*c
        spot[var] = 2
        coeff[tuple(spot)] = s
        spot[var] = 0
    return coeff

def get_MultiPower(center,roots):
    """
    creates a MultiPower object of a hyperellipse/hyperbola with a specified center and roots
    """
    return yr.MultiPower(get_coeff(center,roots))

def gen_almost_multiple_roots(dim,seed,delta,verbose=False):
    """
    Generates an n-dimensional hyperellipse/hyperbola with random seed 'seed.'
    The first root is *almost* multiplicity 'dim.' Specifically, the first root is chosen,
    and then dim-1 perturbations of that root are forced to also be roots. Those perturbations
    are chosen using a normal distribution in each coordinate with mean 0 and standard deviation delta.
    """
    np.random.seed(seed)
    centers = np.random.randn(dim,dim)
    if verbose: print('Centers:',centers,sep='\n')
    root = np.random.randn(dim)
    if verbose: print('Root:',root,sep='\n')
    dirs = np.random.randn(dim,dim)*delta
    dirs[0] = 0
    if verbose: print('Directions:',dirs,sep='\n')
    roots = root+dirs
    if verbose: print('Roots:',roots,sep='\n')
    return roots,[get_MultiPower(c,roots) for c in centers]

def gen_almost_double_roots(dim,seed,delta,verbose=False):
    """
    Generates an n-dimensional hyperellipse/hyperbola with random seed 'seed.'
    The first root is *almost* a double root. Specifically, the first root is chosen,
    and then a perturbation of that root is forced to also be a root. The perturbation
    is chosen using a normal distribution in each coordinate with mean 0 and standard deviation delta.
    There are also dim-2 other randomly pre-chosen chosen real roots. To see what they are, usee verbose=True.
    """
    np.random.seed(seed)
    centers = np.random.randn(dim,dim)
    if verbose: print('Centers:',centers,sep='\n')
    root = np.random.randn(1,dim)
    if verbose: print('Root:',root,sep='\n')
    direction = np.random.randn(1,dim)*delta
    if verbose: print('Perturbation:',direction,sep='\n')
    root2 = root+direction
    if verbose: print('Almost Double Root:',root2,sep='\n')
    if dim > 2:
        other_roots = np.random.randn(dim-2,dim)
        if verbose: print('Other Roots:',other_roots,sep='\n')
        roots = np.vstack((root,root2,other_roots))
    else:
        roots = np.vstack((root,root2))
    if verbose: print('Total Roots:',roots,sep='\n')
    return roots,[get_MultiPower(c,roots) for c in centers]

def gen_rand_hyperellipses(dim,seed,delta,verbose=False):
    """
    Generates an n-dimensional hyperellipse/hyperbola with random seed 'seed.'
    There are dim randomly pre-chosen real roots. To see what they are, usee verbose=True.
    """
    np.random.seed(seed)
    centers = np.random.randn(dim,dim)
    if verbose: print('Centers:',centers,sep='\n')
    roots = np.random.randn(dim,dim)
    if verbose: print('Roots:',roots,sep='\n')
    return roots,[get_MultiPower(c,roots) for c in centers]

def get_data(delta,gen_func,seeds = {2:range(300),3:range(300),4:range(300)}):
    """
    Computes the conditioning ratio of the first generated root of systems generated with gen_func(dim,seed,delta) for each
    seed in the seeds dictionary.
    Seeds is assumed to be a dictionary where the keys are the dimensions you want to test in, and the values
    are an iterable of random seeds to generate random systems with.
    """
    dims = seeds.keys()
    data = {d:[] for d in dims}
    for dim in dims:
        print(dim)
        for n in seeds[dim]:
            roots,polys = gen_func(dim=dim,seed=n,delta=delta)
            data[dim].extend(conditioningratio(polys,dim,newton=False,root=roots[0]))
        data[dim] = np.array(data[dim]).flatten()
    return data

def plot(datasets,labels=None,subplots=None,title=None,filename='conditioning_ratio_plot',digits_lost=False,figsize=(6,4),dpi=400,best_fit=True):
    """
    Plots conditioning ratio data.

    Parameters
    ----------
    datasets : list of dictionaries
        Growth factor datasets to plot. Each dataset dictionary should be
        formatted to map dimension to an array of conditioning ratios
    digits_lost : bool
        whether the y-axis should be a log scale of the conditioning ratios
        or a linear scale of the digits lost
    figsize : tuple of floats
        figure size
    dpi : int
        dpi of the image
    """
    if subplots is None: fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize,dpi=dpi)
    else: fig, ax = plt.subplots(nrows=subplots[0], ncols=subplots[1], figsize=figsize,dpi=dpi,sharey=True,sharex=True)
    def plot_dataset(ax,data,color,label=None):
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
            if label is not None:
                print(label)
            print('Slope:',slope, '\nIntercept:',intercept,end='\n\n')
            ax.plot(pos,pos*slope+intercept,c=color)
    if subplots is None:
        dims = 2+np.arange(np.max([len(data.keys()) for data in datasets]))
        ax.yaxis.grid(color='gray',alpha=.15,linewidth=1,which='major')
        if labels is None:
            for i,dataset in enumerate(datasets):
                plot_dataset(ax,dataset,f'C{i}')
        else:
            for i,dataset in enumerate(datasets):
                plot_dataset(ax,dataset,f'C{i}',labels[i])
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
        if title is None:
            ax.set_title('Growth Factors of Quadratic Polynomial Systems')
        else:
            ax.set_title(title)
    else:
        for ax_,datasets_axis,title_axis,labels_axis in zip(ax,datasets,title,labels):
            ax_.yaxis.grid(color='gray',alpha=.15,linewidth=1,which='major')
            if labels is None:
                for i,dataset in enumerate(datasets_axis):
                    plot_dataset(ax_,dataset,f'C{i}')
            else:
                for i,dataset in enumerate(datasets_axis):
                    plot_dataset(ax_,dataset,f'C{i}',labels_axis[i])
            ax_.set_title('Growth Factors of Quadratic Polynomial Systems')
            ax_.set_xlabel('Dimension')
            legend_elements = [Patch(facecolor=f'C{i}') for i in range(len(datasets_axis))]
            ax_.legend(legend_elements,labels_axis)
            if title is None:
                ax_.set_title('Growth Factors of Quadratic Polynomial Systems')
            else:
                ax_.set_title(title_axis)
        if title is not None: plt.suptitle(title[-1])
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        dims = 2+np.arange(np.max([len(data.keys()) for data in datasets[0]]))
        max_y_lim = 9
        if digits_lost:
            ax[0].set_ylim(-1,max_y_lim)
            ax[0].set_ylabel('Digits Lost')
        else:
            ax[0].set_ylabel('Growth Factor')
            ax[0].yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
            ax[0].yaxis.set_ticks([np.log10(x) for p in range(-1,max_y_lim)
                                   for x in np.linspace(10**p, 10**(p+1), 10)], minor=True)
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
            crs = get_conditioning_ratios(coeffs, newton)
    elif test == 'dev':
        eps = .1
        kind = 'power'
        N = 50
        devestating_conditioning_ratios(dims,eps,kind,newton,N=N)
    else:
        raise ValueError("1st input must be one of 'rand' for random polys 'dev' for devestating example")
