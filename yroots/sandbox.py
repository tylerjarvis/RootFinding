import numpy as np
import time
import matplotlib.pyplot as plt
from test_trimMs import ALL_MS, ALL_ERRORS
#ALL_MS is a list of lists of numpy arrays
#ALL_ERRORS is a single numpy array whose length is equal to the dimension of each array in Ms

def trimMs(Ms, errors, relApproxTol=1e-3, absApproxTol=0):
    """Reduces the degree of each chebyshev approximation M when doing so has negligible error.

    The coefficient matrices are trimmed in place. This function iteratively looks at the highest
    degree coefficient row of each M along each dimension and trims it as long as the error introduced
    is less than the allowed error increase for that dimension.

    Parameters
    ----------
    Ms : list of numpy arrays
        The chebyshev approximations of the functions
    errors : numpy array
        The max error of the chebyshev approximation from the function on the interval
    relApproxTol : double
        The relative error increase allowed
    absApproxTol : double
        The absolute error increase allowed
    """
    dim = Ms[0].ndim
    for polyNum in range(len(Ms)): #Loop through the polynomials
        allowedErrorIncrease = absApproxTol + errors[polyNum] * relApproxTol
        #Use slicing to look at a slice of the highest degree in the dimension we want to trim
        slices = [slice(None) for i in range(dim)] # equivalent to selecting everything
        for currDim in range(dim):
            slices[currDim] = -1 # Now look at just the last row of the current dimension's approximation
            lastSum = np.sum(np.abs(Ms[polyNum][tuple(slices)]))

            # Iteratively eliminate the highest degree row of the current dimension if
            # the sum of its approximation coefficients is of low error, but keep deg at least 2
            while lastSum < allowedErrorIncrease and Ms[polyNum].shape[currDim] > 3:
                # Trim the polynomial
                slices[currDim] = slice(None,-1)
                Ms[polyNum] = Ms[polyNum][tuple(slices)]
                # Update the remaining error increase allowed an the error of the approximation.
                allowedErrorIncrease -= lastSum
                errors[polyNum] += lastSum
                # Reset for the next iteration with the next highest degree of the current dimension.
                slices[currDim] = -1
                lastSum = np.sum(np.abs(Ms[polyNum][tuple(slices)]))
            # Reset to select all of the current dimension when looking at the next dimension.
            slices[currDim] = slice(None)

def trimMsOptimized1(Ms, errors, relApproxTol=1e-3, absApproxTol=0): #Best for loose error tolerances
    """Reduces the degree of each chebyshev approximation M when doing so has negligible error.

    The coefficient matrices (Ms) are trimmed in place, for each matrix, we trim 1 degree off in each dimension until no trimming is possible

    Parameters
    ----------
    Ms : list of numpy arrays
        The chebyshev approximations of the functions, also known as the coefficient matrices
    errors : numpy array
        The max error of the chebyshev approximation from the function on the interval. If Ms is trimmed, these are modified. 
    relApproxTol : double
        The relative error increase allowed
    absApproxTol : double
        The absolute error increase allowed
    """
    dim = Ms[0].ndim
    for polyNum in range(len(Ms)):
        allowedErrorIncrease = absApproxTol + errors[polyNum] * relApproxTol
        trimmable = True
        while trimmable:
            trimmable = False
            totalErrorIncrease = 0
            for currDim in range(dim):
                lastSum = np.sum(np.abs(np.take(Ms[polyNum], indices=-1, axis=currDim)))
                if Ms[polyNum].shape[currDim] > 2 and lastSum < allowedErrorIncrease:
                    trimmable = True
                    Ms[polyNum] = np.delete(Ms[polyNum], -1, axis=currDim)
                    allowedErrorIncrease -= lastSum
                    totalErrorIncrease += lastSum
            errors[polyNum] += totalErrorIncrease

def trimMsOptimized2(Ms, errors, relApproxTol=1e-3, absApproxTol=0): #Most likely to trim the most off, but has the longest run time
    """Reduces the degree of each Chebyshev approximation M when doing so has negligible error.

    The coefficient matrices (Ms) are trimmed in place. For each matrix, the function trims 
    the dimension with the smallest sum iteratively until no more trimming is possible.

    Parameters
    ----------
    Ms : list of numpy arrays
        The Chebyshev approximations of the functions, also known as the coefficient matrices.
    errors : numpy array
        The max error of the Chebyshev approximation from the function on the interval. If Ms is trimmed, these are modified.
    relApproxTol : double
        The relative error increase allowed.
    absApproxTol : double
        The absolute error increase allowed.
    """
    dim = Ms[0].ndim
    for polyNum in range(len(Ms)):
        allowedErrorIncrease = absApproxTol + errors[polyNum] * relApproxTol
        while True:
            minSum = float('inf')
            minDim = None
            
            for currDim in range(dim):
                lastSum = np.sum(np.abs(np.take(Ms[polyNum], indices=-1, axis=currDim))) #The sum of the last elements of the current dimension 
                
                if lastSum < minSum and Ms[polyNum].shape[currDim] > 2:
                    minSum = lastSum
                    minDim = currDim
            
            if minDim is None or minSum >= allowedErrorIncrease: #If no trimming is possible, we're done
                break
            
            Ms[polyNum] = np.delete(Ms[polyNum], -1, axis=minDim)
            allowedErrorIncrease -= minSum
            errors[polyNum] += minSum

def trimMsOptimized3(Ms, errors, relApproxTol=1e-3, absApproxTol=0): #Likely a direct improvement of the original. A good all round function, trims more than the original, but less than the other two, but shouldn't have nearly as long a run time
    """Reduces the degree of each chebyshev approximation M when doing so has negligible error.

    The coefficient matrices (Ms) are trimmed in place, for each matrix, we take as much off as much as we can from the dimension with the most elements until we can't take any more off

    Parameters
    ----------
    Ms : list of numpy arrays
        The chebyshev approximations of the functions, also known as the coefficient matrices
    errors : numpy array
        The max error of the chebyshev approximation from the function on the interval. If Ms is trimmed, these are modified. 
    relApproxTol : double
        The relative error increase allowed
    absApproxTol : double
        The absolute error increase allowed
    """
    dim = Ms[0].ndim
    for polyNum in range(len(Ms)): #Loop through the polynomials
        allowedErrorIncrease = absApproxTol + errors[polyNum] * relApproxTol
        
        sorted_dims = np.argsort(Ms[polyNum].shape)[::-1]
        totalErrorIncrease = 0

        for currDim in sorted_dims:
            while True:
                lastSum = np.sum(np.abs(np.take(Ms[polyNum], indices=-1, axis=currDim)))
                
                if lastSum < allowedErrorIncrease and Ms[polyNum].shape[currDim] > 2:
                    Ms[polyNum] = np.delete(Ms[polyNum], -1, axis=currDim)
                    allowedErrorIncrease -= lastSum
                    totalErrorIncrease += lastSum
                else:
                    break
            
            errors[polyNum] += totalErrorIncrease

def test_trimMs_and_plot():
    trim_functions = [trimMs, trimMsOptimized1, trimMsOptimized2, trimMsOptimized3]
    runtimes = {f.__name__: [] for f in trim_functions} #A dictionary of a list of tuples in the format (size_of_dimension, runtime)
    numElementsTrimmed = {f.__name__: [] for f in trim_functions} #A dictionary of a list of tuples in the format (size_of_dimension, num_elements_trimmed)

    for Ms, errors in zip(ALL_MS, ALL_ERRORS):
        original_num_elements = np.sum([np.size(M) for M in Ms])
        size_of_dimension = len(errors)  # M matrices have the same dimension size and errors are the same size as Ms

        for trim_func in trim_functions:

            Ms_copy = [M.copy() for M in Ms]  # Copy the matrices to avoid in-place modification
            errors_copy = errors.copy()

            start_time = time.time()
            trim_func(Ms_copy, errors_copy)
            end_time = time.time()

            runtimes[trim_func.__name__].append((size_of_dimension, end_time - start_time))
            numElementsTrimmed[trim_func.__name__].append((size_of_dimension, original_num_elements - np.sum([np.size(M) for M in Ms_copy])))
    
    # Plotting runtime
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    bar_width = 0.2
    indices = np.arange(len(runtimes[list(runtimes.keys())[0]]))
    for i, (func_name, data) in enumerate(runtimes.items()):
        dimensions, times = zip(*data)
        plt.bar(indices + i * bar_width, times, bar_width, label=func_name)
    plt.xlabel('Size of Dimension')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime vs Size of Dimension')
    plt.xticks(indices + bar_width * (len(trim_functions) - 1) / 2, [dim for dim, _ in runtimes[list(runtimes.keys())[0]]])
    plt.legend()

    # Plotting size of Ms
    plt.subplot(1, 2, 2)
    for i, (func_name, data) in enumerate(numElementsTrimmed.items()):
        dimensions, elements = zip(*data)
        plt.bar(indices + i * bar_width, elements, bar_width, label=func_name)
    plt.xlabel('Size of Dimension')
    plt.ylabel('Number of Elements Trimmed')
    plt.title('Size of Ms vs Size of Dimension')
    plt.xticks(indices + bar_width * (len(trim_functions) - 1) / 2, [dim for dim, _ in numElementsTrimmed[list(numElementsTrimmed.keys())[0]]])
    plt.legend()

    plt.tight_layout()
    plt.savefig('TrimMsRuntime.png')
    plt.show()

test_trimMs_and_plot()