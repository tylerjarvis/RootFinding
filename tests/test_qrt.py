from timeit import default_timer as timer
import numpy as np
import pandas as pd
from yroots.polyroots import solve
from yroots.utils import ConditioningError
from random_tests import load_tests
from yroots.polynomial import MultiPower, MultiCheb

def run_test(polys):
    try:
        t = timer()
        roots = solve(polys)
        t = timer()-t
        res = np.abs([poly(roots) for poly in polys])
        logres = np.log10(res,out=-16*np.ones_like(res),where=(res!=0))
        return np.array([logres.max(),logres.mean(),t,0],dtype='float64')
    except ConditioningError:
        return np.array([0,0,0,1],dtype='float64')

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

def run_tests_parallel(tests,MultiX):
    arr = np.zeros(6,dtype='float64')
    for test in tests:
        polys = [MultiX(coeff) for coeff in test]
        arr[1:5] += run_test(polys)
    arr[5] = tests.shape[0]
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

def run_save_parallel(COMM,RANK,SIZE,dim,degrees,basis,ver,N=None,infilefmt=infilefmt,outfilefmt=outfilefmt):
    # set polynomial basis
    if basis == "power": MultiX = MultiPower
    else: MultiX = MultiCheb

    if RANK == 0:
        arr = np.empty((len(degrees),6),dtype='float64')

    for i,deg in enumerate(degrees):
        if RANK == 0:
            # load test array
            tests = np.ascontiguousarray(np.load(infilefmt.format(dim=dim,deg=deg)))

            # compute splitting into different processes
            if N is None:
                N = tests.shape[0]
            n = N//SIZE

            # send number of tests to each process
            for j in range(1,SIZE-1):
                COMM.Isend(np.array(n,dtype='int'), dest=j)
            COMM.Send(np.array(N-(SIZE-1)*n,dtype='int'), dest=SIZE-1)

            # send arrays to different processes
            for j in range(1,SIZE-1):
                COMM.Isend(tests[j*n:(j+1)*n],dest=j)
            COMM.Isend(tests[(SIZE-1)*n:N],dest=SIZE-1)

            # run local tests
            arr[i] = run_tests_parallel(tests[:n],MultiX)

        else:
            n = np.empty(1,dtype='int')
            COMM.Recv(n, source=0)
            n = int(n)
            tests = np.empty((n,dim,*(deg+1,)*dim),dtype='float')
            COMM.Recv(tests, source=0)
            buffer = run_tests_parallel(tests,MultiX)

        if RANK == 0:
            # recieve results from other processes
            buffer = np.empty((SIZE-1,6),dtype='float64')
            for j in range(1,SIZE-1):
                COMM.Irecv(buffer[j-1], source=j)
            COMM.Irecv(buffer[-1], source=SIZE-1)

        else:
            COMM.Send(buffer, dest=0)

        COMM.barrier()

        if RANK == 0:
            # print("deg = ",deg,"\nbuffer =\n", buffer.sum(axis=0))
            # sum up results
            arr[i] += buffer.sum(axis=0)
            arr[i,0] = deg

    if RANK == 0:
        arr[:,1:4] /= (arr[:,5]-arr[:,4])[:,np.newaxis]
        arr[:,1:3] = (10**arr[:,1:3])
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
    from sys import argv
    from mpi4py import MPI

    N = int(argv[1])
    if N == 0: N = None

    basis = argv[2]
    ver = argv[3]

    # run in parallel
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
    if SIZE > 1:
        for arg in argv[4:]:
            dim = int(arg)
            print(f"{basis} dim {dim} running on {SIZE} processes")
            COMM.barrier()
            run_save_parallel(COMM,RANK,SIZE,dim,degrees[dim],basis,ver,N)

    # run on a single process
    else:
        for arg in argv[4:]:
            dim = int(arg)
            print(f"{basis} dim {dim} running on 1 process")
            run_save(dim,degrees[dim],basis,ver,N)
