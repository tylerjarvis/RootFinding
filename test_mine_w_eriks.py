from yroots.subdivision import interval_approximate_nd
from yroots import eriks_code
import numpy as np

f = lambda x,y: 144*(x**4+y**4)-225*(x**2+y**2) + 350*x**2*y**2+81
g = lambda x,y: y-x**6
a,b = np.array([-1,-1]),np.array([1,1])
f_deg,g_deg = 4,6
Mf = interval_approximate_nd(f,a,b,f_deg)
Mg = interval_approximate_nd(g,a,b,g_deg)

err_f = Mf[0,0] + Mf[0,1] + Mf[1,0]
err_g = Mg[0,0] + Mg[0,1] + Mg[1,0]


roots = eriks_code.solveChebyshevSubdivision([Mf,Mg],np.array([[-1,1],[-1,1]]),np.array([err_f,err_g]))
print(roots)