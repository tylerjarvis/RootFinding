from random_tests import *
from yroots.polynomial import MultiPower, MultiCheb
from yroots.polyroots import solve
from timeit import default_timer as timer

def test_maxdeg(dim,degrees,basis):
    if basis == "power": MultiX = MultiPower
    else: MultiX = MultiCheb
    for deg in degrees:
        coeffs = rand_coeffs(dim,deg,1)[0]
        polys = [MultiX(coeff) for coeff in coeffs]
        t = timer()
        solve(polys)
        print(f"dim {dim}/deg {deg}: time = {timer()-t}")
