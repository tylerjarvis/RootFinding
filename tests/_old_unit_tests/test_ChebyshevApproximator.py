import pytest
from yroots.ChebyshevApproximator import chebApproximate, interval_approximate_nd
import numpy as np

def test_ChebyshevApproximator():
    #Test Chebyshev Approximations!
    testCases = [] #[Function, expected degree]
    testCases.append([lambda x: np.cos(x), np.array([14])])
    testCases.append([lambda x: np.cos(45*np.arccos(x)), np.array([45])])
    testCases.append([lambda x,y: np.cos(x)+np.exp(y), np.array([14, 14])])
    testCases.append([lambda x,y:np.cos(1e3*x)*np.cos(1e3*y), np.array([1096, 1096])])
    testCases.append([lambda x,y:np.cos(np.exp(x / (1.2+y))), np.array([327, 676])])
    testCases.append([lambda x,y:np.cos(np.exp(x+y+4)+1e5), np.array([285, 285])])
    testCases.append([lambda x0,x1,x2,x3: np.cos(np.exp(x0)+np.exp(x1)+np.exp(x2))+np.sin(np.exp(x3)*np.exp(x2)), np.array([26, 26, 32, 32])])
    testCases.append([lambda x,y,z,x4: (np.exp(y-z)**2-x**3)*((y-0.7)**2-(x-0.3)**3)*((np.exp(y-z)+0.2)**2-(x+0.8)**3)*((y+0.2)**2-(x-0.8)**3), np.array([12, 24, 22, 0])])
    testCases.append([lambda x,y,z,x4: ((np.exp(y-z)+.4)**3-(x-.4)**2)*((np.exp(y-z)+.3)**3-(x-.3)**2)*((np.exp(y-z)-.5)**3-(x+.6)**2)*((np.exp(y-z)+0.3)**3-(2*x-0.8)**3), np.array([9, 33, 32, 0])])
    testCases.append([lambda x,y,z,x4: x + y + z, np.array([1, 1, 1, 0])])
    testCases.append([lambda x,y,z,x4: x + y + z + x4, np.array([1, 1, 1, 1])])    

    #Run through the test cases
    for f, expectedDegs in testCases:
        dim = len(expectedDegs)
        a = -np.ones(dim)
        b = np.ones(dim)
        coeff, error = chebApproximate(f, a, b)
        expectedCoeff, supNorm = interval_approximate_nd(f, expectedDegs, a, b, retSupNorm=True)
        D = np.zeros(np.maximum(coeff.shape, expectedCoeff.shape))
        D[tuple([slice(0, d) for d in coeff.shape])] = coeff
        D[tuple([slice(0, d) for d in expectedCoeff.shape])] -= expectedCoeff
        coeffDiff = np.average(np.abs(D))
        relErr = 2**52 * coeffDiff / supNorm #In terms of macheps
        #Assert sure we are still getting a good approximation
        assert(relErr < 1)
        degDiffs = np.array(coeff.shape) - np.array(expectedCoeff.shape)
        #Assert the degree is close
        assert(np.all(degDiffs >= -1))
        assert(np.all(degDiffs <= 5))