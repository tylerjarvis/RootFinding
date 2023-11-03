# YRoots

YRoots is a Python package for numerical root finding. See both YRootsDemo.ipynb and YRootsTutorial.ipynb in the main branch for JupyterNotebook demonstrations of the code's capabilities.  

Documentation is posted at https://tylerjarvis.github.io/RootFinding/

This project was supported in part by the National Science Foundation, grant number DMS-1564502.

<!-- [![Build Status](https://travis-ci.com/tylerjarvis/RootFinding.svg?branch=master)](https://travis-ci.com/tylerjarvis/RootFinding) -->
<!-- [![codecov](https://codecov.io/gh/mtmoncur/tylerjarvis/branch/master/graphs/badge.svg)](https://codecov.io/gh/tylerjarvis/RootFinding) -->
<!-- [![PyPI version](https://badge.fury.io/py/RootFinding.svg)](https://badge.fury.io/py/RootFinding) -->
<!-- [![Code Health](https://landscape.io/github/tylerjarvis/RootFinding/pypackage/landscape.svg)](https://landscape.io/github/tylerjarvis/RootFinding/pypackage) -->

<!-- [![Build Status](https://travis-ci.com/tylerjarvis/RootFinding.svg?branch=master)](https://travis-ci.com/tylerjarvis/RootFinding) -->
<!-- [![codecov](https://codecov.io/gh/mtmoncur/tylerjarvis/branch/master/graphs/badge.svg)](https://codecov.io/gh/tylerjarvis/RootFinding) -->
<!-- [![PyPI version](https://badge.fury.io/py/RootFinding.svg)](https://badge.fury.io/py/RootFinding) -->
<!-- [![Code Health](https://landscape.io/github/tylerjarvis/RootFinding/pypackage/landscape.svg)](https://landscape.io/github/tylerjarvis/RootFinding/pypackage) -->

### Requirements
* Python 3.10
* Pip 21.1
* Numpy 1.22.0
* Numba 0.37.0
* Scipy 1.10.0
* Sympy 1.5.1

## Installation

`$ git clone https://github.com/tylerjarvis/RootFinding.git`

(We are currently working on getting a `pip` or `conda` for download)

Rootfinding can now be installed locally by using `pip install -e .` while inside the RootFinding folder.
The package can then by imported using `import yroots`.

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

If the system includes polynomials, there are specialized `Polynomial` objects which may be allow for faster solving. See YRootsDemo.ipynb and YRootsTutorial.ipynb for details.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Build status

|             | [master](https://github.com/tylerjarvis/RootFinding/tree/master) | [develop](https://github.com/tylerjarvis/RootFinding/tree/develop) |
|-------------|--------|-----|
| Status      |  [![Build Status](https://travis-ci.com/tylerjarvis/RootFinding.svg?branch=master)](https://travis-ci.com/tylerjarvis/RootFinding)      |  [![Build Status](https://travis-ci.com/tylerjarvis/RootFinding.svg?branch=develop)](https://travis-ci.com/tylerjarvis/RootFinding)    |
| Codecov     |  [![Coverage Status](https://codecov.io/gh/mtmoncur/tylerjarvis/branch/master/graphs/badge.svg)](https://codecov.io/gh/tylerjarvis/RootFinding)  |  [![Coverage Status](https://codecov.io/gh/mtmoncur/tylerjarvis/branch/develop/graphs/badge.svg)](https://codecov.io/gh/tylerjarvis/RootFinding)   |

## License
[MIT](https://choosealicense.com/licenses/mit/)
