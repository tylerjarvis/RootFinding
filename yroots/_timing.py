import numpy as np
from yroots.OneDimension import multPower, multCheb, divCheb, divPower
from yroots.polynomial import MultiCheb, MultiPower, getPoly
from yroots.polyroots import solve as prsolve
from yroots.subdivision import solve as subsolve
try:
    from yroots.TVBMethod import solve as TVBsolve
    TVB_avail = True
except:
    TVB_avail = False
import matplotlib.pyplot as plt
import argparse
import cProfile, pstats, io
import time
import os
import pickle
import warnings

def _div(poly):
    prsolve(poly, MSmatrix=-1)

def _mult(poly):
    prsolve(poly, MSmatrix=0)

def _nproots(poly):
    np.roots(poly[0].coeff)

def _npcheb(poly):
    np.polynomial.chebyshev.chebroots(poly[0].coeff)

if TVB_avail:
    def _TVB(poly):
        TVBsolve(poly)

def bertini(polys):
    def mononmial_from_exp(exponents, var_chars):
        """
        Examples
        --------
        >>> expo = (1,2,3)
        >>> chars = 'xyz'
        >>> mon = mononmial_from_exp(expo, chars)
        >>> mon
        'x^1*y^2*z^3'
        """
        s = ''
        for i,exp in enumerate(exponents):
            if exp > 0:
                s += '{}^{}*'.format(var_chars[i], exp)
        return s.rstrip('*')

    def coeff_to_str(coeff, var_chars):
        s = ''
        for pos, val in np.ndenumerate(coeff):
            if val != 0:
                s += '{:.4f}*{}'.format(val, mononmial_from_exp(pos,var_chars)).rstrip('*') + '+'
        return s.rstrip('+')

    if len(polys) > 3:
        import string
        var_chars = string.ascii_lowercase[:len(polys)]
    else:
        var_chars = 'xyz'[:len(polys)]
        # "CONFIG\n"
        #   "TRACKTYPE: 1;\n"
        #   "END;\n"

    header = ("INPUT\n"
              "variable_group {};\n".format(', '.join(var_chars))+
              "function {};\n".format(', '.join('f'+str(i) for i in range(len(polys))))
              )
    body = ''
    for i,poly in enumerate(polys):
        body += "f{} = {};\n".format(i, coeff_to_str(poly.coeff, var_chars))

    footer = "END;"

    # print(header+body+footer)
    with open('input', 'w') as f:
        f.write(header+body+footer)

    from subprocess import call
    # call(['./bertini/bertini.exe'])
    # call(['./bertini/bertini-serial'])
    # call(['./bertini/bertini-run-parallel'])
    call(['./bertini/bertini-parallel'])

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
    degrees = {2: [7,13,19,25,31,37,43,49],#,55,61],
                   3: [3,5,7,9,11,13],
                   4: [2,3,4],
                   5: [2,3]
                  }[dim]
    if solver.__name__ == 'bertini':
       degrees = [i for i in degrees if i < 19]
    for deg in degrees:
        np.random.seed(121*deg)
        tot_time = 0
        for _ in range(args.trials):
            polys = [getPoly(deg, dim=dim, power=power) for _ in range(dim)]
            start = time.time()
            solver(polys)
            tot_time += time.time()-start
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
    results['Division power'] = times
    print('Finished trials for division power')

    degrees, times = timer(_mult, args.dim, power=True)
    results['Multiplication power'] = times
    print('Finished trials for multiplication power')

    if TVB_avail:
        degrees, times = timer(_TVB, args.dim, power=True)
        results['TVB power'] = times
        print('Finished trials for TVB power')

    if args.bertini:
        degrees, times = timer(bertini, args.dim, power=True)
        results['bert_degrees'] = degrees
        results['bertini'] = times
        print('Finished trials for multiplication power')

    degrees, times = timer(_div, args.dim, power=False)
    results['div cheb'] = times
    print('Finished trials for division chebyshev')

    degrees, times = timer(_mult, args.dim, power=False)
    results['mult cheb'] = times
    print('Finished trials for multiplication chebyshev')

    if TVB_avail:
        degrees, times = timer(_TVB, args.dim, power=False)
        results['TVB cheb'] = times
        print('Finished trials for TVB cheb')

    if args.dim == 1:
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
    ymax = 1.05*max([max(v) for k,v in results.items() if ('degrees' not in k)])
    ymax = max(ymax, 0.1)
    plt.figure(figsize=(11,5))
    # plt.figure(figsize=(11/2,5))
    plt.subplot(121)
    plot = plt.semilogy if args.bertini else plt.plot
    for key,times in results.items():
        if 'power' not in key: continue
        plot(degrees,  times, label=key.split()[0])

    if args.bertini:
        plot(results['bert_degrees'], results['bertini'], label='Bertini')
    plt.xlim(0,xmax)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Polynomial Degree", fontsize=14)
    plt.ylabel("Average Time over {} Trials (seconds)".format(args.trials), fontsize=14)
    plt.ylim(0,ymax)
    plt.yticks(fontsize=12)
    plt.legend(loc='best')
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
        plt.show(block=True)

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
    if args.method in ['mult', 'div']:
        prsolve(polys, {'mult':0, 'div':-1}[args.method])
    elif args.method == 'bertini':
        bertini(polys)
    prof.disable()

    s = io.StringIO()
    ps = pstats.Stats(prof, stream=s).strip_dirs().sort_stats(args.sort)

    if args.verbosity < 5:
        ps.print_stats(args.verbosity*20 - 15)
    else:
        ps.print_stats()

    print(s.getvalue())

def iterate(filename, ext):
    print(filename, ext, os.path.exists(filename+ext))
    if not os.path.exists(filename+ext):
        return filename+ext
    else:
        i = 1
        while os.path.exists(filename+"({})".format(i)+ext):
            i+=1
        return filename+"({})".format(i)+ext

def save_results(obj, filename):
    print(filename)
    with open(filename, "wb") as f:
        print("Before write")
        pickle.dump(obj, f)
    print("After write")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Timing Options")

    #both
    parser.add_argument('--dim', type=int, default=2, choices=[1,2,3,4,5], help='Dimension of polynomials')
    parser.add_argument('-W', '--nowarn', action='store_true', help='Turn off warnings')

    # graph options
    parser.add_argument('-t', '--trials', type=int, default=10, help='Numbers of trials for graph results')
    parser.add_argument('-d', '--display', action='store_true', help='Display timing graphs')
    parser.add_argument('-b', '--bertini', action='store_true', help='Include bertini')
    parser.add_argument('-s', '--save', type=str, default='', help='Save timing graphs to file')

    # single timing options
    parser.add_argument('-v', '--verbosity', type=int, default=1, choices=[0,1,2,3,4,5], help='Level of detail on single timing. Level 0 will not run single timing.')
    parser.add_argument('--deg', type=int, default=0, help='Degree of polynomial on single timing')
    parser.add_argument('-p', '--power', action='store_true', help='Use power basis polynomial on single timing')
    parser.add_argument('-c', '--cheb', action='store_true', help='Use chebyshev basis polynomial on single timing')
    parser.add_argument('-m', '--method', default='mult', choices=['mult','div', 'bertini'], help='Method to use on single timing')
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
        save_results(results, iterate("timing_results",".pkl"))
        create_graph(results, args)

    # run one large computation
    if args.verbosity > 0:
        name = {'mult':'Multiplication','div':'Division','bertini':'Bertini'}
        print('\n'+"-"*40)
        print("Polynomial Basis : {}".format("Power" if args.power else "Chebyshev"))
        print("Dimension        : {}".format(args.dim))
        print("Degree           : {}".format(args.deg))
        print("Solver Method    : {}".format(name[args.method]))
        print("-"*40+'\n')
        run_single_problem(args)
