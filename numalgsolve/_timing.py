import numpy as np
from numalgsolve.OneDimension import multPowerR, multChebR
from numalgsolve.polynomial import MultiCheb, MultiPower, getPoly
from numalgsolve.polyroots import solve as prsolve
from numalgsolve.subdivision import solve as subsolve
import matplotlib.pyplot as plt
import argparse
import cProfile, pstats, io
import time
import warnings

def _multPowerR(poly):
    multPowerR(poly[0].coeff)

def _multChebR(poly):
    multChebR(poly[0].coeff)

def _div(poly):
    prsolve(poly, 'div')

def _mult(poly):
    prsolve(poly, 'mult')

def _nproots(poly):
    np.roots(poly[0].coeff)

def _npcheb(poly):
    np.polynomial.chebyshev.chebroots(poly[0].coeff)

# One Dimension
def timer(solver, dim, power):
    """Timing a specific root solving method for different polynomial sizes.

    Parameters
    ----------
    solver : function
        The root solving function to test
    power : boolean
        True indicates using power basis, False for chebyshev basis

    Returns
    -------
    degrees : list
        list of polynomial degrees that were timed
    times : list
        list of average times for the solver based on degree
    """
    times = []
    max_degree = {1:250, 2:20, 3:7, 4:4, 5:3} #keys by dimensions
    interval = {1:30, 2:3, 3:1, 4:1, 5:1}
    min_degree = {1:10, 2:2, 3:2, 4:2, 5:2}
    degrees = list(range(min_degree[dim],max_degree[dim]+1,interval[dim]))
    for deg in degrees:
        np.random.seed(121*deg)
        tot_time = 0
        #print(deg)
        for _ in range(args.trials):
            polys = [getPoly(deg, dim=dim, power=power) for _ in range(dim)]
            start = time.clock()
            solver(polys)
            tot_time += time.clock()-start
        times.append(tot_time/args.trials)
    return degrees, times

def run_timer(args):
    """Run timing to compare roots solving methods of division, multiplication,
    and numpy on power and chebyshev polynomials.

    Parameters
    ----------
    args : parsed arguments
        Currently, this is unused in this function.

    Returns
    -------
    results : dictionary of lists
        Each list in results is the timing results of that problem type on
        various sizes of polynomial.
    """
    results = {}

    degrees, times = timer(_div, args.dim, power=True)
    results['degrees'] = degrees
    results['div power'] = times
    print('Finished trials for division power')

    degrees, times = timer(_mult, args.dim, power=True)
    results['mult power'] = times
    print('Finished trials for multiplication power')

    degrees, times = timer(_div, args.dim, power=False)
    results['div cheb'] = times
    print('Finished trials for division chebyshev')

    degrees, times = timer(_mult, args.dim, power=False)
    results['mult cheb'] = times
    print('Finished trials for multiplication chebyshev')

    if args.dim == 1:
        degrees, times = timer(_multPowerR, args.dim, power=True)
        results['multR power'] = times
        print('Finished trials for rotated multiplication power')

        degrees, times = timer(_multChebR, args.dim, power=True)
        results['multR cheb'] = times
        print('Finished trials for rotated multiplication chebyshev')

        degrees, times = timer(_nproots, args.dim, power=True)
        results['numpy power'] = times
        print('Finished trials for numpy power')

        degrees, times = timer(_npcheb, args.dim, power=False)
        results['numpy cheb'] = times
        print('Finished trials for numpy chebyshev')

    return results

def create_graph(results, args):
    degrees = results['degrees']
    xmax = int(1.05*max(degrees))

    ymax = 1.05*max([max(v) for k,v in results.items() if k != 'degrees'])
    ymax = max(ymax, 0.1)
    plt.figure(figsize=(11,5))
    plt.subplot(121)
    for key,times in results.items():
        if 'power' not in key: continue
        plt.plot(degrees,  times, label=key.split()[0])

    plt.xlim(0,xmax)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Polynomial Degree", fontsize=14)
    plt.ylabel(f"Average Time over {args.trials} Trials (seconds)", fontsize=14)
    plt.ylim(0,ymax)
    plt.yticks(fontsize=12)
    plt.legend(loc='upper left')
    plt.title("Power Basis Solve Times", fontsize=16)

    ax = plt.subplot(122)

    for key,times in results.items():
        if 'cheb' not in key: continue
        plt.plot(degrees,  times, label=key.split()[0])

    plt.xlim(0,xmax)
    plt.xlabel("Polynomial Degree", fontsize=14)
    plt.xticks(fontsize=12)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    plt.yticks(fontsize=12)
    plt.ylim(0,ymax)
    plt.legend(loc='upper left')
    plt.title("Chebyshev Basis Solve Times", fontsize=16)

    if args.save:
        ext = '' if args.save.endswith('.pdf') else '.pdf'
        plt.savefig(args.save+ext,bbox_inches="tight")

    if args.display:
        plt.show()


def run_single_problem(args):
    """
    Run a root solver on a single problem, using the arguments providedself.

    Parameters
    ----------
    args : parsed arguments
        Determines solver type and polynomial type and size.
    """
    np.random.seed(0)

    polys = [getPoly(args.deg, args.dim, args.power) for _ in range(args.dim)]

    prof = cProfile.Profile()

    prof.enable()
    prsolve(polys, args.method)
    prof.disable()

    s = io.StringIO()
    ps = pstats.Stats(prof, stream=s).strip_dirs().sort_stats(args.sort)

    if args.verbosity < 5:
        ps.print_stats(args.verbosity*20 - 15)
    else:
        ps.print_stats()

    print(s.getvalue())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Timing Options")

    #both
    parser.add_argument('--dim', type=int, default=2, choices=[1,2,3,4,5], help='Dimension of polynomials')
    parser.add_argument('-W', '--nowarn', action='store_true', help='Turn off warnings')

    # graph options
    parser.add_argument('-t', '--trials', type=int, default=10, help='Numbers of trials for graph results')
    parser.add_argument('-d', '--display', action='store_true', help='Display timing graphs')
    parser.add_argument('-s', '--save', type=str, default='', help='Save timing graphs to file')

    # single timing options
    parser.add_argument('-v', '--verbosity', type=int, default=1, choices=[0,1,2,3,4,5], help='Level of detail on single timing. Level 0 will not run single timing.')
    parser.add_argument('--deg', type=int, default=0, help='Degree of polynomial on single timing')
    parser.add_argument('-p', '--power', action='store_true', help='Use power basis polynomial on single timing')
    parser.add_argument('-c', '--cheb', action='store_true', help='Use chebyshev basis polynomial on single timing')
    parser.add_argument('-m', '--method', default='mult', choices=['mult','div'], help='Method to use on single timing')
    parser.add_argument('--sort', default='tottime', choices=['cumulative','tottime'], help='Method for sorting functions in time results')


    args = parser.parse_args()

    if args.trials < 1:
        raise ValueError("trials must be a positive integer")

    #assert only power or cheb
    if args.power and args.cheb:
        raise ValueError("Choose either power or chebyshev basis, but not both.")

    if not (args.power or args.cheb):
        args.power = True

    if args.deg < 0:
        raise ValueError("Degree must be a positive integer")

    if args.deg == 0:
        degree_dimension = {1:100, 2:25, 3:7, 4:4, 5:3}
        args.deg = degree_dimension[args.dim]

    if args.nowarn:
        warnings.filterwarnings('ignore')

    # run small computations for graphing
    if args.display or args.save:
        results = run_timer(args)
        create_graph(results, args)

    # run one large computation
    if args.verbosity > 0:
        print('\n'+"-"*40)
        print("Polynomial Basis : {}".format("Power" if args.power else "Chebyshev"))
        print("Dimension        : {}".format(args.dim))
        print("Degree           : {}".format(args.deg))
        print("Solver Method    : {}".format("Multiplication" if args.method=='mult' else 'Division'))
        print("-"*40+'\n')
        run_single_problem(args)
