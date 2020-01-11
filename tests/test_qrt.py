from sys import argv
from timeit import default_timer as timer
import numpy as np
import pandas as pd
from yroots.polyroots import solve
from yroots.utils import ConditioningError
from random_tests import load_tests

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

infilefmt="random_tests/dim{dim}_deg{deg}.npy"
def run_tests(dim,degrees,basis,N=None,filefmt=infilefmt):
    arr = np.zeros((len(degrees),6))
    for i,deg in enumerate(degrees):
        arr[i,0] = deg
        tests = load_tests(dim,deg,basis,N=N,filefmt=infilefmt)
        for polys in tests:
            arr[i,1:5] += run_test(polys)
    arr[:,5] = len(tests)
    arr[:,1:4] /= (arr[:,5]-arr[:,4])[:,np.newaxis]
    arr[:,1:3] = (10**arr[:,1:3])
    return arr

outfilefmt="test_qrt/{ver}/dim{dim}_{basis}.csv"
columns = ['deg','maxres','avgres','time','fails','N']
intcols = ['deg','fails','N']
def run_save(dim,degrees,basis,ver,N=None,infilefmt=infilefmt,outfilefmt=outfilefmt):
    arr = run_tests(dim,degrees,basis,N=N,filefmt=infilefmt)
    df = pd.DataFrame(arr,columns=columns)
    df[intcols] = df[intcols].applymap(np.int64)
    df = df.set_index('deg')
    print(df)
    df.to_csv(outfilefmt.format(dim=dim,basis=basis,ver=ver))

degrees = {}
degrees[2] = np.arange(2,26)
degrees[3] = np.arange(2,11)
degrees[4] = np.arange(2,6)
degrees[5] = [2,3,4]
degrees[6] = [2,3]
degrees[7] = [2]
degrees[8] = [2]
degrees[9] = [2]
degrees[10] = [2]

if __name__ == "__main__":
    basis = argv[1]
    ver = argv[2]
    for arg in argv[3:]:
        dim = int(arg)
        run_save(dim,degrees[dim],basis,ver)
        print(f"{basis} dim {dim} complete")
