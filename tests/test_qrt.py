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

if __name__ == "__main__":
    # dimension 2
    degrees = np.arange(2,26)
    run_save(2,degrees,"power")
    run_save(2,degrees,"cheb")

    # dimension 3
    degrees = np.arange(2,11)
    run_save(3,degrees,"power")
    run_save(3,degrees,"cheb")

    # dimension 4
    degrees = np.arange(2,6)
    run_save(4,degrees,"power")
    run_save(4,degrees,"cheb")

    # dimension 5
    run_save(5,[2,3,4],"power")
    run_save(5,[2,3,4],"cheb")

    # # dimension 6
    # run_save(6,[2,3],"power")
    # run_save(6,[2,3],"cheb")
    #
    # # dimension 7
    # run_save(7,[2],"power")
    # run_save(7,[2],"cheb")
    #
    # # dimension 8
    # run_save(8,[2],"power")
    # run_save(8,[2],"cheb")
    #
    # # dimension 9
    # run_save(9,[2],"power")
    # run_save(9,[2],"cheb")
    #
    # # dimension 10
    # run_save(10,[2],"power")
    # run_save(10,[2],"cheb")
