import numpy as np
import yroots as yr
from os import sys

TwoD_Basis = ["x", 
             "y",
             "x**2",
             "y**2",
             "x**3",
             "y**3",
             "np.cos(x)",
             "np.cos(y)",
             "np.sin(y)",
             "np.cos(y)",
             "np.cos(x/2)",
             "np.cos(y/2)",
             "np.sin(x/2)",
             "np.sin(y/2)",
             "np.log(np.abs(x)) - 5",
             "np.log(np.abs(y)) - 5"]

ThreeD_Basis = ["x", 
                "y",
                "z",
                "x**2",
                "y**2",
                "z**2",
                "x**3",
                "y**3",
                "z**3",
                "np.cos(x)",
                "np.cos(y)",
                "np.cos(z)",
                "np.sin(x)",
                "np.sin(y)",
                "np.sin(z)",
                "np.cos(x/2)",
                "np.cos(y/2)",
                "np.cos(z/2)",
                "np.sin(x/2)",
                "np.sin(y/2)",
                "np.sin(z/2)",
                "np.log(np.abs(x)) - 5",
                "np.log(np.abs(y)) - 5",
                "np.log(np.abs(z)) - 5"]



class RandSmoothFunc:
    """ Random Smooth Function class. It takes in a randomly generated function
    and keeps track of it as both a callable lambda function and the string.
    """

    def __init__(self, basis=None, n=5, dim=2):
        """
        Parameters
        ----------
            basis : list of strings
                The basis from which to build a random function.
            n : int
                Maximum number of subfunctions from which to choose.
            dim : int
                The dimension of the domain.
        """
        if basis is None:
            self.basis = TwoD_Basis
        else:
            self.basis = basis
        
        self.dim = dim
        random_int = np.random.randint(1, n)
        func_str = self.lin_combos(np.random.choice(self.basis, random_int, replace=False), np.random.choice(self.basis, random_int, replace=True))
        self.string = func_str
        self.func = eval(func_str)

    def lin_combos(self, funcs, coeffs=None):
        """Create a random linear combiniation of the given functions and coeffs.
        
        Parameters
        ----------
            funcs : list of strings (easily turned into callables)
                The functions to use in the linear combiniations. 2D.
            coeffs : list of floats/callables
                The list from which to take the coefficients.
        
        Returns
        -------
            lin_combo : str
                New function defined by the linear combination.
        """
        lin_combo = ""
        n = len(funcs)
        if coeffs is None:
            coeffs = np.random.random(n)
        for i, func in enumerate(funcs):
            lin_combo = lin_combo + str(np.random.random()) + "*" + str(np.random.choice(coeffs)) + "*(" + func + ") + "
    
        if self.dim == 2:
            return "lambda x, y: " + lin_combo[:-2]
        else:
            return "lambda x, y, z : " + lin_combo[:-2]

    def __str__(self):
        """Return the function as a string (written in python)"""
        if self.dim == 2:
            return self.string[13:]
        elif self.dim == 3:
            return self.string[17:]
        else:
            return self.string


    def __call__(self, *args):
        """Call the function represented."""
        return self.func(*args)

# Choose an interesting random seed like 0, 3, 9, 49, 81, 102, 123, 224, 265, 316, 329, 456, 1011, 1234, 8675309
if __name__ == "__main__":

    if len(sys.argv) == 1:
        dim = 2
    else:
        dim = int(sys.argv[1])

    if dim == 2:
        # Run 2D Tests - These seeds have known roots
        for seed in [0, 3, 9, 49, 102, 123, 224, 265, 316, 329, 456, 1011, 1234, 8675309]:
            print("Running 2D Tests with seed {}".format(seed))
            np.random.seed(seed)
            F = RandSmoothFunc()
            G = RandSmoothFunc()
            print("Functions")
            print(F)
            print(G)
            print("\n\n")
            roots = yr.solve([F, G], -5*np.ones(2), 5*np.ones(2), plot=False, plot_name="2D Random Funcs".format(seed), max_level=9, plot_err=True)

    if dim == 3:
        # Run 3D Tests - These seeds have not been tested for roots. Just for subdivision purposes.
        for seed in range(0,10):
            print("Running 3D Tests with seed {}".format(seed))
            np.random.seed(seed)
            F = RandSmoothFunc(ThreeD_Basis, dim=3)
            G = RandSmoothFunc(ThreeD_Basis, dim=3)
            H = RandSmoothFunc(ThreeD_Basis, dim=3)
            print("Functions")
            print(F)
            print(G)
            print(H)
            print("\n\n")
            roots = yr.solve([F, G, H], -5*np.ones(3), 5*np.ones(3), plot=False, plot_name="3D Random Funcs".format(seed), max_level=8, plot_err=True)

def tau(l, n, d):
    """Calculates average degree reduction (tau) given the relationship
        tau^level * n = d
        
    Parameters
    ----------
        l : int
            Levels of subdivision. How many times the interval was subdivided.
        n : int
            Approximate initial degree.
        d : int
            Approximate target degree.
    
    Returns
    -------
        float
            The average degree reduction.
    """
    return np.exp(np.log(d/n)/l)