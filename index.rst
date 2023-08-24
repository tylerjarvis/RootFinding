YRoots
================================================================================

*A fast-working package for finding the roots of multivariate systems of equations.*

How YRoots Works
--------------------------------------------------------------------------------

YRoots harnesses the properties of Chebyshev polynomial approximation to quickly and precisely find and
return the roots of various systems of functions.

Given a list of smooth, continuous functions and a compact search interval, YRoots generates an accurate
approximation for each function on the interval and recursively uses numerical methods to zero in on any
roots contained in the interval.

Getting Started with YRoots
--------------------------------------------------------------------------------

Getting started with YRoots is quick and simple. To learn how to use the solver, navigate to the
yroots.solve() page for the documentation and examples.

Some users may wish to use two special YRoots class objects, MultiCheb and MultiPower, built for faster
function evaluations of Chebyshev-based or power-based polynomials. To learn how to use these, see the
corresponding documentation. 