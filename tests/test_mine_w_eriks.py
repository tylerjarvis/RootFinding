from yroots.subdivision import interval_approximate_nd
from yroots import eriks_code

f = lambda x,y: 144*(x**4+y**4)-225*(x**2+y**2) + 350*x**2*y**2+81
g = lambda x,y: y-x**6

interval_approximate_nd()