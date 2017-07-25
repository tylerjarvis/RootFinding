#RootFinding

Root finding methods for multivariate polynomial equations.

**Organization of this Repository**

**Solving a System of Polynomial Equations**

Groebner bases can be used to find solutions to a system of polynomial equations.
Unfortunately, computing Groebner bases is known to be unstable due to floating point error propagation.
Polynomials in the power basis can be represented by a summation of Chebyshev polynomials, which are good for stable computation.
Converting the polynomials to Chebyshev polynomials reduces the error significantly.
Here's an example using the code from this repository:

*Example*

Developed at Brigham Young University 2016-2017
