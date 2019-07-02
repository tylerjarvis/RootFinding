import numpy as np
from yroots.polynomial import MultiCheb, MultiPower
from yroots.OneDimension import multPower, multCheb, divPower, divCheb
#from yroots.TVBMethod import solve as TVBsolve
from yroots.polyroots import solve
from yroots.Division import division
from yroots.Multiplication import multiplication
from numpy.polynomial.polynomial import polyfromroots, polyroots
from numpy.polynomial.chebyshev import chebfromroots, chebroots
from matplotlib import pyplot as plt
from itertools import product
import argparse
import warnings

class Solver:
    def __init__(self, solver, name, basis, eigvals_avail, defaults_kwargs={}):
        self.solver = solver
        self.name = name
        self.basis = basis
        self.eigvals = eigvals_avail
        self.defaults_kwargs = defaults_kwargs

    def __call__(self, polys, *args, **kwargs):
        return self.solver(polys, *args, **self.defaults_kwargs, **kwargs)

    def __str__(self):
        return self.name

class OneDSolver(Solver):
    def __init__(self, solver, name, basis, eigvals_avail):
        super().__init__(solver, name, basis, eigvals_avail)

    def __call__(self, poly, *args):
        if self.eigvals:
            return self.solver(poly.coeff, *args)
        else:
            return self.solver(poly.coeff)

multPower_s = OneDSolver(multPower, "Mult Power", 'power', True)
divPower_s = OneDSolver(divPower, "Div Power", 'power', True)
numpy_s = OneDSolver(polyroots, "Numpy Power", 'power', False)

multCheb_s = OneDSolver(multCheb, "Mult Cheb", 'cheb', True)
divCheb_s = OneDSolver(divCheb, "Div Cheb", 'cheb', True)
numpyCheb_s = OneDSolver(chebroots, "Numpy Cheb", 'cheb', False)

multiplication_s = Solver(multiplication, "Multiplication", "both", True, defaults_kwargs={'MSmatrix':1})
multrand_s = Solver(multiplication, "Multiplication Random", "both", True, defaults_kwargs={'MSmatrix':0})
division_s = Solver(division, "Division", "both", True)

all_solvers = [multPower_s, divPower_s, numpy_s,
               multCheb_s, divCheb_s, numpyCheb_s,
               multiplication_s, multrand_s, division_s]

def create_roots_graph(args, results):
    nrows = len(results)
    ncols = len(next(iter(results.values())))
    if not args.coeffs: ncols-=1  #get sub dictionary length, but ignore the roots item

    plt.figure(figsize=(2*ncols,2*nrows+0.5))
    for i,(radius, (sub_results, residuals)) in enumerate(results.items()):
        radius = float(radius)
        if not args.coeffs:
            roots = sub_results['roots']
            del sub_results['roots']
        for j,(method, roots_approx) in enumerate(sub_results.items()):
            ax = plt.subplot(nrows,ncols,i*ncols+(j+1))
            if args.dimension == 1:
                if not args.coeffs: plt.plot(roots.real, roots.imag, 'r+', ms = 7)
                plt.plot(roots_approx.real, roots_approx.imag, 'bo', ms=3)
            else:
                # colors = np.zeros((len(roots_approx), 3), dtype=np.int)
                if not args.coeffs:
                    # colors[:,:2] = (254/np.max(roots.imag) * (roots.imag - np.min(roots.imag))).astype(np.int)
                    plt.plot(roots[:,0], roots[:,1], 'r+', ms = 4)#, c=np.abs(roots[:,0].imag))
                # colors[:,:2] = (254/np.max(roots_approx.imag) * (roots_approx.imag - np.min(roots_approx.imag))).astype(np.int)
                plt.plot(roots_approx[:,0], roots_approx[:,1], 'bo', ms=2)
            r = args.radius*1.1
            plt.xlim(-r,r)#-radius,radius)
            plt.ylim(-r,r)#-radius,radius)
            if j > 0:
                ax.set_yticklabels([])
            else:
                if args.dimension==1:
                    plt.ylabel('imag')
                else:
                    plt.ylabel('real')
            if i < nrows-1:
                ax.set_xticklabels([])
            else:
                plt.xlabel('real')
            if i==0:plt.title(method)

    plt.tight_layout()
    plt.show()

def create_stability_graph():
    raise NotImplementedError()

def calculate_residual(fs, z):
    """See definition 2 of
    A Stabilized Normal Form Algorithm for
    Generic Systems of Polynomial Equations
    by Simon Telen and Marc Van Barel
    """
    rs = []
    n = len(fs)
    for f_i in fs:
        f_i_abs = type(f_i)(np.abs(f_i.coeff))
        r_i = (np.abs(f_i(z)) / (f_i_abs(np.abs(z))+1))
        rs.append(r_i)
    return sum(rs)/n

def maximal_residual(polys, roots):
    residuals = []
    polys = polys if isinstance(polys,list) else [polys]
    for root in roots:
        residuals.append(calculate_residual(polys, root))
    return max(residuals)

def logplot(*vals):
    cnt = len(vals)
    plt.figure(figsize=(2.5*cnt+0.5,3.5))
    for i in range(cnt):
        plt.subplot(1,cnt,i+1)
        z = np.clip(np.log10(np.abs(vals[i])),-20,1)
        plt.imshow(z, vmin=-20,vmax=1)
    # plt.colorbar()
    plt.show()

def run_one_dimension(args, radius, eigvals):
    num_points = args.num_points
    eps = args.eps
    power = args.power
    real = args.real
    by_coeffs = args.coeffs

    root_pts = {}
    residuals = {}
    if by_coeffs:
        coeffs = (np.random.random(num_points+1)*2 - 1)*radius
        powerpoly = MultiPower(coeffs)
        chebpoly = MultiCheb(coeffs)
    else:
        r = np.sqrt(np.random.random(num_points))*radius + eps
        angles = 2*np.pi*np.random.random(num_points)
        if power and not real:
            roots = r*np.exp(angles*1j)
        else:
            roots = 2*r-radius
        root_pts = {'roots':roots}

        powerpoly = MultiPower(polyfromroots(roots))
        chebpoly = MultiCheb(chebfromroots(roots))
    n = 1000
    x = np.linspace(-1,1,n)
    X,Y = np.meshgrid(x,1j*x)

    for solver in all_solvers:
        if isinstance(solver,OneDSolver):
            if (solver.basis == 'cheb' and args.cheb) and ((not eigvals) or solver.eigvals):
                name = str(solver)
                root_pts[name] = solver(chebpoly, eigvals)
                residuals[name] = maximal_residual(chebpoly, root_pts[name])
            if (solver.basis == 'power' and args.power) and ((not eigvals) or solver.eigvals):
                name = str(solver)
                root_pts[name] = solver(powerpoly, eigvals)
                residuals[name] = maximal_residual(powerpoly, root_pts[name])


    if args.hist:
        evaluations = {}
        for k,v in root_pts.items():
            if k == 'roots': continue
            poly = powerpoly if 'power' in k else chebpoly
            evaluations[k] = np.abs(poly(root_pts[k]))
        ncols = len(evaluations)
        fig, ax = plt.subplots(1,ncols, sharey=True, figsize=(12,4))
        minimal = -20
        maximal = 1
        for i,(k,v) in enumerate(evaluations.items()):
            ax[i].hist(np.clip(np.log10(v), minimal, maximal),range=(minimal,maximal), bins=40)
            ax[i].set_xlabel(r'$log_{10}(p(r_i))$')
            ax[i].set_title(k)
        plt.suptitle("Eigenvalues" if eigvals else "Eigenvectors")
        plt.show()

    return root_pts, residuals

def run_n_dimension(args, radius, eigvals):
    num_points = args.num_points
    eps = args.eps
    power = args.power
    real = args.real
    by_coeffs = args.coeffs
    dim = args.dimension

    root_pts = {}
    residuals = {}
    powerpolys = []
    chebpolys = []
    if by_coeffs:
        for i in range(dim):
            from yroots.polynomial import getPoly
            powerpolys.append(getPoly(num_points, dim, power=True))
            chebpolys.append(getPoly(num_points, dim, power=False))
    else:
        r = np.random.random((num_points, dim))*radius + eps
        roots = 2*r-radius

        root_pts = {'roots':np.array(list(product(*np.rot90(roots))))}

        for i in range(dim):
            coeffs = np.zeros((num_points+1,)*dim)
            idx = [slice(None),]*dim
            idx[i] = 0
            coeffs[tuple(idx)] = polyfromroots(roots[:,i])
            lt = [0]*dim
            lt[i] = num_points
            powerpolys.append(MultiPower(coeffs))#, lead_term=lt, clean_zeros=False))

            coeffs[tuple(idx)] = chebfromroots(roots[:,i])
            chebpolys.append(MultiCheb(coeffs))#, lead_term=lt, clean_zeros=False))
            # plt.subplot(121);plt.imshow(coeffs);plt.subplot(122);plt.imshow(chebpolys[0].coeff);plt.show()


    for solver in all_solvers:
        if not isinstance(solver,OneDSolver) and solver.basis in ['power','both']:
            # if ((not eigvals) or solver.eigvals):
            name = str(solver) + ' Power'
            root_pts[name] = solver(powerpolys)
            residuals[name] = maximal_residual(powerpolys, root_pts[name])


    for solver in all_solvers:
        if not isinstance(solver,OneDSolver) and solver.basis in ['cheb','both']:
            # if ((not eigvals) or solver.eigvals):
            name = str(solver) + ' Cheb'
            root_pts[name] = solver(chebpolys)
            residuals[name] = maximal_residual(chebpolys, root_pts[name])

    if args.hist:
        evaluations = {}
        for k,v in root_pts.items():
            if k == 'roots': continue
            # evaluations[k] = []
            polys = powerpolys if 'Power' in k else chebpolys
            # for poly in polys:
            evaluations[k] = sum(np.abs(poly(root_pts[k])) for poly in polys)
        ncols = len(evaluations)
        # plt.figure(figsize=(12,6))
        fig, ax = plt.subplots(1,ncols, sharey=True, figsize=(12,4))
        minimal = -15
        maximal = 1
        for i,(k,v) in enumerate(evaluations.items()):
            ax[i].hist(np.clip(np.log10(v), minimal, maximal),range=(minimal,maximal), bins=40)
            ax[i].set_xlabel(r'$log_{10}(p(r_i))$')
            ax[i].set_title(k)
        plt.suptitle("Eigenvalues" if eigvals else "Eigenvectors")
        plt.show()

    return root_pts, residuals

def run_roots_testing(args):

    radius = args.radius
    dim = args.dimension
    results = {}
    eigvals_options = [True] if args.eig=='val' else [False]
    eigvals_options = [True,False] if args.eig=='both' else eigvals_options

    seed = np.random.randint(0,100000)
    if dim == 1:
        for option in eigvals_options:
            np.random.seed(seed)
            results[option] = run_one_dimension(args, radius, option)
    else:
        for option in eigvals_options:
            np.random.seed(seed)
            results[option] = run_n_dimension(args, radius, option)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Stability Test Options")
    parser.add_argument('-g', '--graph', action='store_true', help='Graph Zeros')
    parser.add_argument('-d', '--dimension', type=int, default=1, help='Polynomial dimension')
    parser.add_argument('-n', '--num_points', type=int, default=50, help='Number of complex roots, minimum of 2')
    parser.add_argument('--real', action='store_true', help='Use just real points')
    parser.add_argument('-r', '--radius', type=float, default=1, help='The largest radius for the points')
    parser.add_argument('-e', '--eps', type=float, default=1e-8, help='Minimum distance from 0')
    parser.add_argument('-W', '--nowarn', action='store_true', help='Turn off warnings')
    parser.add_argument('--eig', type=str, default='val', choices=['val','vec','both'], help='Choose between eigenvalues and eigenvectors')
    parser.add_argument('-p', '--power', action='store_true', help='Check the power methods using complex points')
    parser.add_argument('-c', '--cheb', action='store_true', help='Check the chebyshev methods using real points')
    parser.add_argument('--coeffs', action='store_true', help='Choose random coefficients instead of roots.')
    parser.add_argument('--hist', action='store_true', help='Plot histogram of polynomial evaluations at roots.')

    args = parser.parse_args()

    #assert only power or cheb
    # if args.power and args.cheb:
    #     raise ValueError("Choose either power or chebyshev basis, but not both.")

    if not (args.power or args.cheb):
        args.power = True

    if args.nowarn:
        warnings.filterwarnings('ignore')

    if args.num_points <= 0: raise ValueError("Not enough points")
    if args.radius <= 0: raise ValueError("Max radius must be positive")

    results = run_roots_testing(args)
    if args.graph:
        create_roots_graph(args, results)

    print("With eigvals")
    try:
        for key,val in results[True][1].items():
            print("{:>30} {:.3e}".format(key, val))
    except:
        pass

    print("Without eigvals")
    try:
        for key,val in results[False][1].items():
            print("{:>30} {:.3e}".format(key, val))
    except:
        pass
