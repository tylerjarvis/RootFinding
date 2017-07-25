# #### To caclulate a Gröbner Basis

x,y,z = QQ["x,y,z"].gens()
f1, f2 = y-2*x, 1+2*x*y
I = ideal(f1,f2)
B = I.groebner_basis()
print(B)

I2 = ideal(x - 2*y, 1+2*x*y)
B2 = I2.groebner_basis()
print(B2)

I3 = ideal(1+x*z, 2+y**2*x**2)
B3 = I3.groebner_basis()
print(B3)

[0,-1,0]
[0,0,3]
[0,0,0]

[-1,0,0]
[0,0,0]
[0,5,0]

# #### To use verbose Buchberger's

from sage.rings.polynomial.toy_buchberger import *

I3 = ideal(1+x, 1+y)
B3 = I3.groebner_basis()
print(B3)

set_verbose(1)
buchberger(I3)

f1 = 1 + x
f2 = 1+ y
f3 = 1+x**2*y+x
I = ideal(f1,f2,f3)
buchberger(I)

