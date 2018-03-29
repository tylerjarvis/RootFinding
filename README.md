# NumAlgSolve ![CI status](https://img.shields.io/badge/build-passing-brightgreen.svg)

---

NumAlgSolve is a Python module for numerical and algebraic rootfinding. For our mathematical methods and their comparisons with other rootfinders, refer to [this paper](paper).

### Requirements
* Python 3.3 and up

## Installation

`$ git clone https://github.com/tylerjarvis/RootFinding.git`

(We are currently working on getting a `pip` or `conda` for download)

## Usage

```python
#conda imports
import numpy as np

#local imports
from numalgsolve.polynomial import MultiCheb, MultiPower
from numalgsolve.root_finder import roots

A = MultiCheb(np.array([[1,2,3,1],[2,3,1,0],[2,3,0,0],[1,0,0,0]]))
B = MultiCheb(np.array([[1,0,0,1],[1,0,1,0],[0,0,0,0],[1,0,0,0]]))
roots([A,B], method='TVB')
#insert user code here
```

For a demonstration notebook with examples, see CHEBYSHEV/DEMO.ipynb.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)