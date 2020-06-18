import yroots as yr
import numpy as np
from .growth_factors import devestating_growth_factors

polys = devestating_growth_factors([7],.1,'power',newton=True,N=50,just_dev_root=True,seed=468,perturb_eps=0,save=False)
roots = yr.polysolve(polys)
print(roots)
