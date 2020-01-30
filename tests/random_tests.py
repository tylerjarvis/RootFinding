import numpy as np
from yroots.polynomial import MultiCheb, MultiPower

filefmt="random_tests/coeffs/dim{dim}_deg{deg}_{kind}.npy"
def rand_coeffs(dim,deg,N,kind,maxint=10):
    shape = (*(deg+1,)*dim,dim,N)
    if kind == 'randint':
        coeffs = np.random.randint(-maxint,maxint+1,size=shape)
    elif kind == 'randn':
        coeffs = np.random.randn(*shape)
    # if pcnt_sparse is not None:
    #     #make sparse
    #     idx = np.random.choice(np.arange(coeffs.size),replace=False,size=int(coeffs.size * pcnt_sparse))
    #     idx = np.unravel_index(idx,shape)
        # coeffs[idx] = 0
    for idx in np.ndindex((deg+1,)*dim):
        if np.sum(idx) > deg:
            coeffs[idx] = 0
    return coeffs.T

def gen_tests(dim,degrees,N,kind):
    arrs = []
    for deg in degrees:
        arrs.append(rand_coeffs(dim,deg,N,kind))
    return arrs

def save_tests(dim,degrees,N,kind,filefmt=filefmt):
    arrs = gen_tests(dim,degrees,N)
    for i,deg in enumerate(degrees):
        np.save(filefmt.format(dim=dim,deg=deg,kind=kind),arrs[i])
        print(f"dim {dim}/deg {deg}: saved N={N} systems")

def load_tests(dim,deg,basis,kind,N=None,filefmt=filefmt):
    arr = np.load(filefmt.format(dim=dim,deg=deg,kind=kind))
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
