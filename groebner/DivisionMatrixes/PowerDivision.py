import numpy as np
import itertools
from groebner.utils import clean_zeros_from_matrix, get_var_list
from groebner.TelenVanBarel import create_matrix, add_polys
from groebner.root_finder import _random_poly
from scipy.linalg import solve_triangular
from scipy.stats import mode
from scipy.linalg import qr

def division_power(polys):
    '''Calculates the common zeros of polynomials using a division matrix.
    
    Parameters
    --------
    polys: MultiPower Polynomials
        The polynomials for which the common roots are found.

    Returns
    -----------
    zeros : list
        The common zeros of the polynomials. Each list element is a numpy array of complex entries
        that contains the coordinates in each dimension of the zero.
    '''
    #This first section creates the Macaulay Matrix with the monomials that only have ys first.
    dim = polys[0].dim
    matrix_degree = np.sum(poly.degree for poly in polys) - len(polys) + 1
    poly_coeff_list = []
    for i in polys:
        poly_coeff_list = add_polys(matrix_degree, i, poly_coeff_list)
    matrix, matrix_terms, matrix_shape_stuff = create_matrix(poly_coeff_list, matrix_degree, dim)
    #perm is a permutation to reorder the matrix columns to put the ys first.
    perm, cut = matrix_term_perm(matrix_terms)
    matrix = matrix[:,perm]
    matrix_terms = matrix_terms[perm]

    #Reduces the Macaulay matrix like normal.
    A,B = matrix[:,:cut], matrix[:,cut:]
    Q,A,P = qr(A, pivoting=True)
    matrix_terms[:cut] = matrix_terms[:cut][P]
    B = Q.T@B
    C,D = B[:cut], B[cut:]
    Q0,D,P0 = qr(D, pivoting=True)
    C = C[:,P0]
    matrix_terms[cut:] = matrix_terms[cut:][P0]
    matrix[:,:cut] = A
    matrix[:cut,cut:] = C
    matrix[cut:,cut:] = D
    rows,columns = matrix.shape

    VB = matrix_terms[matrix.shape[0]:]
    matrixFinal = np.hstack((np.eye(rows),solve_triangular(matrix[:,:rows],matrix[:,rows:])))
    basisDict = makeBasisDict(matrixFinal, matrix_terms, VB)

    #Dictionary of terms in the vector basis their spots in the matrix.
    VBdict = {}
    spot = 0
    for row in VB:
        VBdict[tuple(row)] = spot
        spot+=1

    
    # Build division matrix
    dMatrix = np.zeros((len(VB), len(VB)))
    for i in range(VB.shape[0]):
        var = [1,0]
        term = tuple(VB[i] - var)
        if term in VBdict:
            dMatrix[VBdict[term]][i] += 1
        else:
            dMatrix[:,i] -= basisDict[term]
    
    #Calculate the eigenvalues and eigenvectors.
    vals, vecs = np.linalg.eig(dMatrix.T)
    
    #Finds two spots in the vector basis that differ by y so we can use the eigenvalues to calculate the y values.
    ys = list()
    for row in VB:
        if tuple(row-[0,1]) in VBdict:
            ys.append(VBdict[tuple(row)])
            ys.append(VBdict[tuple(row-[0,1])])
            break
    
    #Finds the zeros, the x values from the eigenvalues and the y values from the eigenvectors.
    zeros = list()
    for i in range(len(vals)):
        zeros.append(np.array([1/vals[i], vecs[ys[0]][i]/vecs[ys[1]][i]]))
    
    return zeros

def matrix_term_perm(matrix_terms):
    '''Finds the needed column permutation to have all the y terms first in the Macaulay Matrix.
    
    Parameters
    --------
    matrix_terms: numpy array
        The current order of the terms in the matrix.

    Returns
    -----------
    perm : numpy array
        The desired column permutation.
    cut : The number of y terms in the matrix. This is where the matrix is cut in the matrix reduction,
          pivoting past this point is not allowed.
    '''
    boundary = np.where(matrix_terms[:,0] == 0)[0]
    bSet = set(boundary)
    other = np.arange(len(matrix_terms))
    mask = [num not in bSet for num in other]
    other = other[mask]
    return np.hstack((boundary,other)), len(boundary)

def makeBasisDict(matrix, matrix_terms, VB):
    '''Calculates and returns the basisDict.
    
    This is a dictionary of the terms on the diagonal of the reduced TVB matrix to the terms in the Vector Basis.
    It is used to create the multiplication matrix in root_finder.
    
    Parameters
    --------
    matrix: numpy array
        The reduced TVB matrix.
    matrix_terms : numpy array
        The terms in the matrix. The i'th row is the term represented by the i'th column of the matrix.
    VB : numpy array
        Each row is a term in the vector basis.

    Returns
    -----------
    basisDict : dict
        Maps terms on the diagonal of the reduced TVB matrix (tuples) to numpy arrays of the shape remainder_shape
        that represent the terms reduction into the Vector Basis.
    '''
    basisDict = {}
    
    VBSet = set()
    for i in VB:
        VBSet.add(tuple(i))
    
    #We don't actually need most of the rows, so we only get the ones we need.
    neededSpots = set()
    for term, mon in itertools.product(VB,get_var_list(VB.shape[1])):
        if tuple(term-mon) not in VBSet:
            neededSpots.add(tuple(term-mon))

    for i in range(matrix.shape[0]):
        term = tuple(matrix_terms[i])
        if term not in neededSpots:
            continue
        basisDict[term] = matrix[i][matrix.shape[0]:]

    return basisDict