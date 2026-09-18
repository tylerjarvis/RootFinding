# YRoots

YRoots is a Python package designed for numerical rootfinding of multivariate systems of equations.

For a tutorial on YRoots syntax, set-up, and examples on how to use it with different function systems, see [Combined Notebook](https://github.com/tylerjarvis/RootFinding/blob/main/CombinedNotebook.ipynb). 

To try YRoots without installing anything, open that notebook on Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wlgns0330/RootFinding/blob/main/CombinedNotebook.ipynb)

Its first code cell installs YRoots into the Colab runtime. Colab runs the standard build of CPython rather than
the free-threaded 3.14t build below, so the cell waives pip's version check: every example returns the same roots
to the same precision, but the parallel examples get no speedup out of the extra threads.

Documentation is posted at https://tylerjarvis.github.io/RootFinding/

This project was supported in part by the National Science Foundation, grant number DMS-1564502.

<!-- [![Build Status](https://travis-ci.com/tylerjarvis/RootFinding.svg?branch=master)](https://travis-ci.com/tylerjarvis/RootFinding) -->
<!-- [![codecov](https://codecov.io/gh/mtmoncur/tylerjarvis/branch/master/graphs/badge.svg)](https://codecov.io/gh/tylerjarvis/RootFinding) -->
<!-- [![PyPI version](https://badge.fury.io/py/RootFinding.svg)](https://badge.fury.io/py/RootFinding) -->
<!-- [![Code Health](https://landscape.io/github/tylerjarvis/RootFinding/pypackage/landscape.svg)](https://landscape.io/github/tylerjarvis/RootFinding/pypackage) -->

### Requirements
* Python 3.14t (free-threaded build — see note below)
* NumPy ≥ 2.4.4
* Numba ≥ 0.65.1
* SciPy ≥ 1.17.1
* SymPy ≥ 1.12

> **Why 3.14t?** YRoots requires the free-threaded build of Python 3.14, which runs without the Global Interpreter Lock (GIL) for better parallelism. The `t` suffix identifies this build — it is a different download from the standard Python 3.14.

## Installation

With uv (recommended):
```
uv python install 3.14t
uv pip install git+https://github.com/tylerjarvis/RootFinding.git
```
Or clone and install for development:

```
git clone https://github.com/tylerjarvis/RootFinding.git
cd RootFinding
uv sync
```

With pip (requires Python 3.14t already installed):

```
pip install git+https://github.com/tylerjarvis/RootFinding.git
```

The package can then be imported using `import yroots`.
(We are currently working on adding the yroots package to The Python Package Index)

## Usage

```python
#imports
import numpy as np
import yroots as yr

#define the functions -- must be smooth on the domain and vectorized
f = lambda x,y : np.sin(x*y) + x*np.log(y+3) - x**2 + 1/(y-4)
g = lambda x,y : np.cos(3*x*y) + np.exp(3*y/(x-2)) - x - 6

#define a search domain
a = np.array([-1,-2]) #lower bounds on x and y
b = np.array([0,1]) #upper bounds on x and y

#solve
yr.solve([f,g],a,b)
```

If the system includes polynomials, there are specialized `Polynomial` objects which may allow for faster solving. See [Combined Notebook](https://github.com/tylerjarvis/RootFinding/blob/main/CombinedNotebook.ipynb) for more details.


## Examples of Applications
Below is a list of Jupyter notebooks in which YRoots has been used to solve real-world problems:
- [Solving Equilibrium Points of First Order ODE Systems](https://github.com/tylerjarvis/RootFinding/blob/main/Applications/Equilibrium%20Points.ipynb)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
