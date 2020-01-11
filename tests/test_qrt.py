import os
from timeit import default_timer as timer
import numpy as np
from yroots.polynomial import MultiCheb, MultiPower
from yroots.polyroots import solve
from yroots.utils import ConditioningError

infilefmt="random_tests/dim{dim}_deg{deg}.npy"
outfilefmt="test_qrt/{ver}/dim{dim}_{basis}.npy"
def rand_coeffs(dim,deg,N):
    shape = (*(deg+1,)*dim,dim,N)
    coeffs = np.random.randn(*shape)
    for idx in np.ndindex((deg+1,)*dim):
        if np.sum(i) > deg:
            coeffs[idx] = 0
    return coeffs.T

def gen_tests(dim,degrees,N,filefmt=infilefmt):
    for deg in degrees:
        coeffs = rand_coeffs(dim,deg,N)
        np.save(filefmt.format(dim=dim,deg=deg),coeffs)
        print(f"dim={dim}, deg={deg}, N={N} saved")

def load_tests(dim,deg,basis,N=None,filefmt=infilefmt):
    arr = np.load(filefmt.format(dim=dim,deg=deg,basis=basis))
    if basis == "power": MultiX = MultiPower
    else: MultiX = MultiCheb
    tests = []
    if N is None: N = arr.shape[0]
    for test in arr[:N]:
        polys = []
        for coeff in test:
            polys.append(MultiX(coeff))
        tests.append(polys)
    return tests

def run_test(polys):
    try:
        t = timer()
        roots = solve(polys)
        t = timer()-t
        res = np.abs([poly(roots) for poly in polys])
        logres = np.log10(res,out=-16*np.ones_like(res),where=(res!=0))
        return logres.max(),logres.mean(),t,0
    except ConditioningError:
        return 0,0,0,1

def run_tests(dim,degrees,basis,N=None,filefmt=infilefmt):
    arr = np.zeros((len(degrees),5))
    for i,deg in enumerate(degrees):
        tests = load_tests(dim,deg,basis,N=N,filefmt=infilefmt)
        for polys in tests:
            arr[i,:4] += run_test(polys)
    arr[:,4] = len(tests)
    arr[:,:3] /= (arr[:,4]-arr[:,3])[:,np.newaxis]
    arr[:,:2] = (10**arr[:,:2])
    return arr

def run_save(dim,degrees,basis,ver,N=None,infilefmt=infilefmt,outfilefmt=outfilefmt):
    arr = run_tests(dim,degrees,basis,N=N,filefmt=infilefmt)
    np.save(outfilefmt.format(dim=dim,basis=basis,ver=ver),arr)

if __name__=="__main__":
    # dimension 2
    degrees = np.arange(2,21)
    gen_tests(2,degrees)
