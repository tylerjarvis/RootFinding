from numalgsolve import polynomial
from numalgsolve import polyroots
import numpy as np

p1 = polynomial.MultiPower(np.array([[1, -4, 0],[0, 3, 0],[1, 0, 0]]).T) #y^2 + 3xy - 4x +1
p2 = polynomial.MultiPower(np.array([[3, 0, -2],[6, -6, 0],[0, 0, 0]]).T) #-6xy -2x^2 + 6y +3
c1 = polynomial.MultiCheb(np.array([[2, 0, -1],[6, -6, 0], [0, 0, 0]]).T) #p2 in Cheb form
c2 = polynomial.MultiCheb(np.array([[1.5, -4, 0],[0, 3, 0], [.5, 0, 0]]).T) #p2 in Cheb form

print("~ ~ ~ Power Form, Mx Matrix ~ ~ ~")
print("Roots\n", polyroots.solve([p1, p2], MSmatrix = 1, verbose=True))

print("~ ~ ~ Cheb Form, Mx Matrix ~ ~ ~")
print("Roots\n",polyroots.solve([c1, c2], MSmatrix = 1, verbose=True))

print("~ ~ ~ Power Form,  My Matrix ~ ~ ~")
print("Roots\n", polyroots.solve([p1, p2], MSmatrix = 2, verbose=True))

print("~ ~ ~ Cheb Form, My Matrix ~ ~ ~")
print("Roots\n",polyroots.solve([c1, c2], MSmatrix = 2, verbose=True))

print("~ ~ ~ Power Form, Pseudorandom Multiplication Matrix ~ ~ ~")
print("Roots\n", polyroots.solve([p1, p2], MSmatrix = 0, verbose=True))

print("~ ~ ~ Cheb Form, Pseudorandom Multiplication Matrix ~ ~ ~")
print("Roots\n",polyroots.solve([c1, c2], MSmatrix = 0, verbose=True))

print("~ ~ ~ Power Form, Division Matrix 1/x ~ ~ ~")
print("Roots\n",polyroots.solve([p1, p2], MSmatrix = -1, verbose=True))

print("~ ~ ~ Cheb Form, Division Matrix 1/x ~ ~ ~")
print("Roots\n",polyroots.solve([c1, c2], MSmatrix = -1, verbose=True))

print("\n\n\nSimple One Dimensional Example")
p = polynomial.MultiPower(np.array([2, -4,1.])) #x^2 - 4x + 2
c = polynomial.MultiCheb(np.array([2.5, -4, .5])) #p in Cheb form

print("~ ~ ~ Companion Matrix ~ ~ ~")
print("Roots\n", polyroots.solve([p], MSmatrix = 1, verbose=True))

print("~ ~ ~ Rotated Companion Matrix ~ ~ ~")
print("Roots\n", polyroots.solve([p], MSmatrix = 0, verbose=True))

print("~ ~ ~ Power Division Matrix ~ ~ ~")
print("Roots\n",polyroots.solve([p], MSmatrix = -1, verbose=True))

print("~ ~ ~ Colleague Matrix ~ ~ ~")
print("Roots\n",polyroots.solve([c], MSmatrix = 1, verbose=True))

print("~ ~ ~ Rotated Colleague Matrix ~ ~ ~")
print("Roots\n", polyroots.solve([c], MSmatrix = 0, verbose=True))

print("~ ~ ~ Cheb Division Matrix ~ ~ ~")
print("Roots\n",polyroots.solve([c], MSmatrix = -1, verbose=True))
