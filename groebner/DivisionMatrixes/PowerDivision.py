import numpy as np
import itertools
from groebner.utils import get_var_list, slice_top, row_swap_matrix, mon_combos
from groebner.TelenVanBarel import add_polys, rrqr_reduceTelenVanBarel2, rrqr_reduceTelenVanBarel
from groebner.root_finder import newton_polish
from scipy.linalg import solve_triangular
from matplotlib import pyplot as plt

def division_power(polys, divisor_var = 0):
    '''Calculates the common zeros of polynomials using a division matrix.
    
    Parameters
    --------
    polys: MultiPower Polynomials
        The polynomials for which the common roots are found.
    divisor_var : int
        What variable is being divided by. 0 is x, 1 is y, etc. Defaults to x.

    Returns
    -----------
    zeros : list
        The common zeros of the polynomials. Each list element is a numpy array of complex entries
        that contains the coordinates in each dimension of the zero.
    '''
    #This first section creates the Macaulay Matrix with the monomials that don't have
    #the divisor variable in the first columns.
    dim = polys[0].dim
    matrix_degree = np.sum(poly.degree for poly in polys) - len(polys) + 1
        
    poly_coeff_list = []
    for i in polys:
        poly_coeff_list = add_polys(matrix_degree, i, poly_coeff_list)
        
    matrix, matrix_terms, cuts = create_matrix(poly_coeff_list, matrix_degree, dim, divisor_var)
        
    #Reduces the Macaulay matrix like normal.
    matrix, matrix_terms = rrqr_reduceTelenVanBarel2(matrix, matrix_terms, cuts)
    rows,columns = matrix.shape
    matrixFinal = np.hstack((np.eye(rows),solve_triangular(matrix[:,:rows],matrix[:,rows:])))
        
    VB = matrix_terms[matrix.shape[0]:]
        
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
        var = np.zeros(dim)
        var[divisor_var] = 1
        term = tuple(VB[i] - var)
        if term in VBdict:
            dMatrix[VBdict[term]][i] += 1
        else:
            dMatrix[:,i] -= basisDict[term]
    
    #Calculate the eigenvalues and eigenvectors.
    vals, vecs = np.linalg.eig(dMatrix.T)
                
    #Finds the zeros, the divisor variable values from the eigenvalues and the other variable values from the eigenvectors.
    zeros = list()
    for i in range(len(vals)):
        if abs(vecs[-1][i]) < 1.e-12: #This should be a root at infinity
            continue
        root = np.zeros(dim, dtype=complex)
        root[divisor_var] = 1/vals[i]
        for spot in range(0,divisor_var):
            root[spot] = vecs[-(2+spot)][i]/vecs[-1][i]
        for spot in range(divisor_var+1,dim):
            root[spot] = vecs[-(1+spot)][i]/vecs[-1][i]
        #root = newton_polish(polys,root, tol = 1.e-8)
        zeros.append(root)
    
    return zeros

def get_matrix_terms(poly_coeffs, dim, divisor_var, deg):
    '''Finds the terms in the Macaulay matrix.
    
    Parameters
    --------
    poly_coeffs: list
        A list of numpy arrays that contain the coefficients of the polynomials to go into the Macaualy Matrix.
    dim : int
        The dimension of the polynomials in the matrix.
    divisor_var : int
        What variable is being divided by. 0 is x, 1 is y, etc. Defaults to x.

    Returns
    -----------
    matrix_terms : numpy array
        The matrix_terms. The ith row is the term represented by the ith column of the matrix.
    cuts : tuple
        When the matrix is reduced it is split into 3 parts with restricted pivoting. These numbers indicate
        where those cuts happen.    
    '''
    matrix_term_set_y= set()
    matrix_term_set_other= set()
    for coeffs in poly_coeffs:
        for term in zip(*np.where(coeffs != 0)):
            if term[divisor_var] == 0:
                matrix_term_set_y.add(term)
            else:
                matrix_term_set_other.add(term)
    
    needed_terms = list()
    base = np.zeros(dim, dtype = 'int')
    base[divisor_var] = 1
    matrix_term_set_other.remove(tuple(base))
    matrix_term_end = base.copy()
    for i in range(dim):
        if i != divisor_var:
            base[i] = 1
            term = tuple(base)
            matrix_term_set_other.remove(term)
            matrix_term_end = np.vstack((term,matrix_term_end))
            base[i] = 0
    for term in needed_terms:
        matrix_term_set_other.remove(term)
    matrix_terms = np.vstack((np.vstack(matrix_term_set_y),np.vstack(matrix_term_set_other),matrix_term_end))
    
    return matrix_terms, tuple([len(matrix_term_set_y), len(matrix_term_set_y)+len(matrix_term_set_other)])
    
def create_matrix(poly_coeffs, degree, dim, divisor_var):
    ''' Builds a Macaulay matrix for reduction.

    Parameters
    ----------
    poly_coeffs : list.
        Contains numpy arrays that hold the coefficients of the polynomials to be put in the matrix.
    degree : int
        The degree of the TVB Matrix
    dim : int
        The dimension of the polynomials going into the matrix.
    divisor_var : int
        What variable is being divided by. 0 is x, 1 is y, etc.
    
    Returns
    -------
    matrix : 2D numpy array
        The Telen Van Barel matrix.
    matrix_terms : numpy array
        The ith row is the term represented by the ith column of the matrix.
    cuts : tuple
        When the matrix is reduced it is split into 3 parts with restricted pivoting. These numbers indicate
        where those cuts happen.
    '''
    bigShape = [degree+1]*dim
    matrix_terms, cuts = get_matrix_terms(poly_coeffs, dim, divisor_var, degree)
    
    #Get the slices needed to pull the matrix_terms from the coeff matrix.
    matrix_term_indexes = list()
    for row in matrix_terms.T:
        matrix_term_indexes.append(row)
    
    #Adds the poly_coeffs to flat_polys, using added_zeros to make sure every term is in there.
    added_zeros = np.zeros(bigShape)
    flat_polys = list()
    for coeff in poly_coeffs:
        slices = slice_top(coeff)
        added_zeros[slices] = coeff
        flat_polys.append(added_zeros[matrix_term_indexes])
        added_zeros[slices] = np.zeros_like(coeff)
        coeff = 0
    poly_coeffs = 0

    #Make the matrix. Reshape is faster than stacking.
    matrix = np.reshape(flat_polys, (len(flat_polys),len(matrix_terms)))

    #Sorts the rows of the matrix so it is close to upper triangular.
    matrix = row_swap_matrix(matrix)
    return matrix, matrix_terms, cuts

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