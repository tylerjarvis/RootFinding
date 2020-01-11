import numpy as np
from yroots.polynomial import MultiCheb, MultiPower

filefmt="random_tests/dim{dim}_deg{deg}.npy"
def rand_coeffs(dim,deg,N):
    shape = (*(deg+1,)*dim,dim,N)
    coeffs = np.random.randn(*shape)
    for idx in np.ndindex((deg+1,)*dim):
        if np.sum(idx) > deg:
            coeffs[idx] = 0
    return coeffs.T

def gen_tests(dim,degrees,N):
    arrs = []
    for deg in degrees:
        arrs.append(rand_coeffs(dim,deg,N))
    return arrs

def save_tests(dim,degrees,N,filefmt=filefmt):
    arrs = gen_tests(dim,degrees,N)
    for i,deg in enumerate(degrees):
        np.save(filefmt.format(dim=dim,deg=deg),arrs[i])
        print(f"dim {dim}/deg {deg}: saved N={N} systems")

def load_tests(dim,deg,basis,N=None,filefmt=filefmt):
    arr = np.load(filefmt.format(dim=dim,deg=deg))
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
