from timeit import default_timer as timer
import numpy as np
import pandas as pd
from yroots.polyroots import solve
from yroots.utils import ConditioningError
from random_tests import load_tests
from yroots.polynomial import MultiPower, MultiCheb

def run_test(polys,method):
    try:
        t = timer()
        roots,cond,cond_back,cond_eig = solve(polys,method)
        t = timer()-t
        res = np.abs([poly(roots) for poly in polys])
        logres = np.log10(res,out=-16*np.ones_like(res),where=(res!=0))
        logcond = np.log10(cond_eig)
        return np.array([logres.max(),
                         logres.mean(),
                         np.log10(cond),
                         np.log10(cond_back),
                         logcond.max(),
                         logcond.mean(),
                         t,
                         0],dtype='float64')
    except ConditioningError:
        return np.array([0,0,0,0,0,0,0,1],dtype='float64')

infilefmt="random_tests/coeffs/dim{dim}_deg{deg}_{kind}.npy"
def run_tests(dim,degrees,basis,method,kind,N=None,filefmt=infilefmt):
    arr = np.zeros((len(degrees),10))
    for i,deg in enumerate(degrees):
        arr[i,0] = deg
        tests = load_tests(dim,deg,basis,kind,N=N,filefmt=infilefmt)
        for polys in tests:
            arr[i,1:9] += run_test(polys,method)
        print(f"deg {deg} complete")
    arr[:,9] = len(tests)
    arr[:,1:8] /= (arr[:,9]-arr[:,8])[:,np.newaxis]
    arr[:,1:7] = (10**arr[:,1:7])
    return arr

def run_tests_parallel(tests,MultiX,method):
    arr = np.zeros(10,dtype='float64')
    for test in tests:
        polys = [MultiX(coeff) for coeff in test]
        arr[1:9] += run_test(polys,method)
    arr[9] = tests.shape[0]
    return arr

outfilefmt="random_tests/{title}/{method}/dim{dim}_{basis}_{kind}.csv"
columns = ['deg','maxres','avgres','macaulay_cond_1','macaulay_cond_2','maxcondeig','avgcondeig','time','fails','N']
intcols = ['deg','fails','N']
def run_save(dim,degrees,basis,method,kind,title,N=None,infilefmt=infilefmt,outfilefmt=outfilefmt):
    arr = run_tests(dim,degrees,basis,method,kind,N=N,filefmt=infilefmt)
    df = pd.DataFrame(arr,columns=columns)
    df[intcols] = df[intcols].applymap(np.int64)
    df = df.set_index('deg')
    print(df)
    df.to_csv(outfilefmt.format(title=title,dim=dim,basis=basis,method=method,kind=kind))

def run_save_parallel(COMM,RANK,SIZE,dim,degrees,basis,method,kind,title,N=None,infilefmt=infilefmt,outfilefmt=outfilefmt):
    # set polynomial basis
    if basis == "power": MultiX = MultiPower
    else: MultiX = MultiCheb

    # dictonary to store coefficent tensors by degree
    tests = {}

    # load coefficient tensors by degree
    itemsize = MPI.DOUBLE.Get_size()
    for deg in degrees:
        # compute number of tests per process
        if N is None:
            N = Ntests[deg]
        n = N//SIZE

        # set up shared memory
        # shape of array containing coefficent tensors
        shape = (N,dim,*(deg+1,)*dim)

        # allocate bytes on rank 0 process
        if RANK == 0: nbytes = itemsize * np.prod(shape)
        else: nbytes = 0
        win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=COMM)

        # get the window to the rank 0 array
        buffer, itemzie = win.Shared_query(0)
        assert itemsize == MPI.DOUBLE.Get_size()
        coeffs = np.ndarray(buffer=buffer, dtype='d', shape=shape)

        # rank 0 process loads the array of coefficent tensors
        if RANK == 0:
            coeffs[...] = np.load(infilefmt.format(dim=dim,deg=deg,kind=kind))[:N]

        # each process stores a reference to a subset of the array
        if RANK < SIZE -1:
            tests[deg] = coeffs[RANK*n:(RANK+1)*n]
        else: tests[deg] = coeffs[(SIZE-1)*n:]

    # wait until every process has loaded all coefficient tensors
    COMM.barrier()

    # run tests
    results = np.zeros((len(degrees),10),dtype='float64')
    for i,deg in enumerate(degrees):
        results[i] = run_tests_parallel(tests[deg],MultiX,method)
        print(f"deg {deg} complete")

    # collect results
    if RANK == 0:
        # recieve results from other processes
        buffer = np.empty((SIZE-1,len(degrees),10),dtype='float64')
        for j in range(1,SIZE):
            COMM.Irecv(buffer[j-1], source=j)

    else:
        # send results to rank 0 process
        COMM.Isend(results, dest=0)

    # wait until all processes have sent results
    COMM.barrier()

    # save results to file
    if RANK == 0:
        results += buffer.sum(axis=0)
        results[:,0] = degrees
        results[:,1:8] /= (results[:,9]-results[:,8])[:,np.newaxis]
        results[:,1:7] = (10**results[:,1:7])
        df = pd.DataFrame(results,columns=columns)
        df[intcols] = df[intcols].applymap(np.int64)
        df = df.set_index('deg')
        print(df)
        df.to_csv(outfilefmt.format(title=title,dim=dim,basis=basis,kind=kind,method=method))

degrees = {}
degrees[2] = np.arange(2,36)
degrees[3] = np.arange(2,13)
degrees[4] = np.arange(2,7)
degrees[5] = [2,3,4]
degrees[6] = [2,3]
degrees[7] = [2]
degrees[8] = [2]
degrees[9] = [2]
degrees[10] = [2]

Ntests = {}
Ntests[2] = 200
Ntests[3] = 100
Ntests[4] = 50
Ntests[5] = 25
Ntests[6] = 25
Ntests[7] = 25
Ntests[8] = 25
Ntests[9] = 15
Ntests[10] = 15

if __name__ == "__main__":
    from sys import argv
    from mpi4py import MPI

    title = argv[1]
    N = int(argv[2])
    if N == 0: N = None

    basis = argv[3]
    method = argv[4]
    kind = argv[5]

    # run in parallel
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
    if SIZE > 1:
        for arg in argv[6:]:
            dim = int(arg)
            if RANK == 0: print(f"{basis} dim {dim} running on {SIZE} processes")
            COMM.barrier()
            run_save_parallel(COMM,RANK,SIZE,dim,degrees[dim],basis,method,kind,title,N)

    # run on a single process
    else:
        for arg in argv[6:]:
            dim = int(arg)
            print(f"{basis} dim {dim} running on 1 process")
            run_save(dim,degrees[dim],basis,method,kind,title,N)
