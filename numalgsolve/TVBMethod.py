"""Methods for solving a system of multivariate polynomials using the
Telen and Van Barel's simultaneous diagonalization method"""

import numpy as np
from scipy.linalg import eig
from numalgsolve.utils import match_poly_dimensions, sort_polys_by_degree, row_swap_matrix, MacaulayError
from numalgsolve.polynomial import MultiCheb, MultiPower, is_power
from numalgsolve.MacaulayReduce import find_degree, add_polys
from scipy.linalg import qr, solve_triangular, qr_multiply
from numalgsolve.Multiplication import create_matrix, makeBasisDict

def solve(polys, verbose=False):
    '''
    Finds the roots of a list of multivariate polynomials by simultaneously
    diagonalizing a multiplication matrices created with the TVB method.

    Parameters
    ----------
    polys : list of polynomial objects
        Polynomials to find the common roots of.
    verbose : bool
        Print information about how the roots are computed.
    returns
    -------
    roots : numpy array
        The common roots of the polynomials. Each row is a root.
    '''
    polys = match_poly_dimensions(polys)
    poly_type = is_power(polys, return_string = True)
    dim = polys[0].dim

    #Reduce the Macaulay Matrix TVB-style to generate a basis for C[]/I and
    # a dictionary that expresses other monomials in terms of that basis
    basisDict, VB = TVB_MacaulayReduction(polys, accuracy = 1.e-10, verbose=verbose)
    if verbose:
        print('Basis for C[]/I\n', VB)
        print('Dictionary which represents non-basis monomials in terms of the basis\n', basisDict)

    #See what happened to dimension of C[]/I. Did we loose roots?
    len_VB = len(VB)
    degrees = [poly.degree for poly in polys]
    max_number_of_roots = np.prod(degrees)
    if len_VB < max_number_of_roots:
        raise MacaulayError('Roots were lost during the Macaulay Reduction')
    if len_VB > max_number_of_roots:
        raise MacaulayError('Dimension of C[]/I is too large')

    #make Mx1, ..., Mxn
    mult_matrices = np.zeros((dim, len_VB, len_VB))
    for i in range(1, dim+1):
        mult_matrices[i-1] = Mxi_Matrix(i, basisDict, VB, dim, poly_type)

    if verbose:
        print('The Multiplication Matrices:\n', mult_matrices)

    #simultaneously diagonalize and return roots
    roots = sim_diag(mult_matrices, verbose=verbose).T
    if verbose:
        print("Roots:\n", roots)
    return roots

def Mxi_Matrix(i, basisDict, VB, dim, poly_type, verbose=False):
    '''
    Uses the reduced Macaulay matrix to construct the Moller-Stetter matrix M_xi, which
    represents the linear map of multiplying by xi in the space C[x1, ..., xn]/I.

    Parameters
    ----------
    i : int
        The index of the variable xi to make the Moller-Stetter matrix of, where variables
        are indexed as x1, x2, ..., xn.
    basisDict: dictionary
        A dictionary which maps monomials not in the basis to linear combinations
        of monomials in the basis. Generated using the TVB method.
    VB: numpy array
        Represents a vector basis for the space C[x1, ..., xn]/I created with the TVB method.
        Each row represents a monomial in the basis as the degrees of each variable.
        For example, x^2y^5 would be represented as [2,5].
    dim: int
        The dimension of the system (n)
    verbose : bool
        Prints information about how the roots are computed.

    Returns
    -------
    Mxi : 2D numpy array
        The Moller-Stetter matrix which represents multiplying by xi
    '''
    VB = VB.tolist() #convert to list bc numpy's __contains__() function is broken

    #Construct the polynomial to create the MS Matrix of (xi)
    xi_ind = np.zeros(dim, dtype=int)
    xi_ind[i-1] = 1
    coef = np.zeros((2,)*dim)
    coef[tuple(xi_ind)] = 1
    if poly_type == "MultiPower":
        xi = MultiPower(np.array(coef))
    elif poly_type == "MultiCheb":
        xi = MultiCheb(np.array(coef))

    if verbose:
        print("\nCoefficients of polynomial whose Moller-Stetter matrix we construt\n", xi.coeff)

    # Build multiplication matrix M_xi
    Mxi = np.zeros((len(VB), len(VB)))
    for j in range(len(VB)): #multiply each monomial in the basis by xi
        product_coef = xi.mon_mult(VB[j], returnType = 'Matrix')
        for monomial in zip(*np.where(product_coef != 0)):
            if list(monomial) in VB: #convert to list to test if list of lists
                Mxi[VB.index(list(monomial))][j] += product_coef[monomial]
            else:
                Mxi[:,j] -= product_coef[monomial]*basisDict[monomial]

    # Construct var_dict
    var_dict = {}
    for i in range(len(VB)):
        mon = VB[i]
        if np.sum(mon) == 1 or np.sum(mon) == 0:
            var_dict[tuple(mon)] = i

    return Mxi

def TVB_MacaulayReduction(initial_poly_list, accuracy = 1.e-10, verbose=False):
    '''
    Reduces the Macaulay matrix to find a vector basis for the system of polynomials.

    Parameters
    --------
    polys: list
        The polynomials in the system we are solving.
    accuracy: float
        How small we want a number to be before assuming it is zero.

    Returns
    -----------
    basisDict : dict
        A dictionary of terms not in the vector basis a matrixes of things in the vector basis that the term
        can be reduced to.
    VB : nparray
        The terms in the vector basis, each list being a term.
    '''
    power = is_power(initial_poly_list)
    dim = initial_poly_list[0].dim
    poly_coeff_list = []
    degree = find_degree(initial_poly_list)
    initial_poly_list = sort_polys_by_degree(initial_poly_list, ascending = False)

    for poly in initial_poly_list:
        poly_coeff_list = add_polys(degree, poly, poly_coeff_list)

    #Creates the matrix for either of the above two methods. Comment out if using the third method.
    matrix, matrix_terms, cuts = create_matrix(poly_coeff_list, degree, dim)
    if verbose:
        np.set_printoptions(suppress=False, linewidth=200)
        print('\nStarting Macaulay Matrix\n', matrix)
        print('\nColumns in Macaulay Matrix\nFirst element in tuple is degree of x monomial, Second element is degree of y monomial\n', matrix_terms)
        print('\nLocation of Cuts in the Macaulay Matrix into [ Mb | M1* | M2* ]\n', cuts)

    #First QR reduction
    #If bottom left is zero only does the first QR reduction on top part of matrix (for speed). Otherwise does it on the whole thing
    if np.allclose(matrix[cuts[0]:,:cuts[0]], 0):
        #RRQR reduces A and D without pivoting, sticking the result in it's place and multiplying the rest of the matrix by Q.T
        C1,matrix[:cuts[0],:cuts[0]] = qr_multiply(matrix[:,:cuts[0]], matrix[:,cuts[0]:].T, mode = 'right')
        matrix[:cuts[0],cuts[0]:] = C1.T
        C1 = 0

        #check if there are zeros along the diagonal of R1
        if any(np.isclose(np.diag(matrix[:,:cuts[0]]),0, rtol=accuracy)):
            raise MacaulayError("R1 IS NOT FULL RANK")

        #set small values to zero before backsolving
        matrix[np.isclose(matrix, 0, rtol=accuracy)] = 0

        matrix[:cuts[0],cuts[0]:] = solve_triangular(matrix[:cuts[0],:cuts[0]],matrix[:cuts[0],cuts[0]:])
        matrix[:cuts[0],:cuts[0]] = np.eye(cuts[0])
        matrix[cuts[0]:,cuts[0]:] -= (matrix[cuts[0]:,:cuts[0]])@matrix[:cuts[0],cuts[0]:] #?
    else:
        #RRQR reduces A and D without pivoting, sticking the result in it's place.
        Q1,matrix[:,:cuts[0]] = qr(matrix[:,:cuts[0]])

        #check if there are zeros along the diagonal of R1
        if any(np.isclose(np.diag(matrix[:,:cuts[0]]),0, rtol=accuracy)):
            raise MacaulayError("R1 IS NOT FULL RANK")

        #Multiplying the rest of the matrix by Q.T
        matrix[:,cuts[0]:] = Q1.T@matrix[:,cuts[0]:]
        Q1 = 0 #Get rid of Q1 for memory purposes.

    #Second QR reduction on all the rest of the matrix
    matrix[cuts[0]:,cuts[0]:],P = qr(matrix[cuts[0]:,cuts[0]:], mode = 'r', pivoting = True)

    #Shift around the top right columns
    matrix[:cuts[0],cuts[0]:] = matrix[:cuts[0],cuts[0]:][:,P]
    #Shift around the columns labels
    matrix_terms[cuts[0]:] = matrix_terms[cuts[0]:][P]
    P = 0

    #set small values to zero
    matrix[np.isclose(matrix, 0, rtol=accuracy)] = 0

    #eliminate zero rows from the bottom of the matrix. Zero rows above
    #nonzero elements are not eliminated. This saves time since Macaulay matrices
    #we deal with are only zero at the very bottom
    #matrix = row_swap_matrix(matrix) #not for TVB's method needed bc QRP is on the whole matrix
    for row in matrix[::-1]:
        if np.allclose(row, 0):
            matrix = matrix[:-1]
        else:
            break

    height = matrix.shape[0]
    matrix[:,height:] = solve_triangular(matrix[:,:height],matrix[:,height:])
    matrix[:,:height] = np.eye(height)
    #return np.vstack((matrix[:,height:].T,np.eye(height))), matrix_terms

    if verbose:
        np.set_printoptions(suppress=True, linewidth=200)
        print("\nFinal Macaulay Matrix\n", matrix)
        print("\nColumns in Macaulay Matrix\n", matrix_terms)
    VB = matrix_terms[height:]

    basisDict = makeBasisDict(matrix, matrix_terms, VB, power)

    return basisDict, VB

def sim_diag(Matrices, verbose=False):
    '''
    Simultaneously diagonalizes several commuting matrices which have the same eigenvectors.

    Parameters
    ----------
    matrices : ndarray
        3D Tensor. Each matrix in the array must commute with every other matrix and they must share all eigenvectors)
    verbose: bool
        Prints information about the diagonalization.

    -------
    sim_diag : numpy array
        The diagonals of each diagonalized matrix. The diagonal matrix is a row
    '''
    sim_diag = np.zeros((Matrices.shape[0], Matrices.shape[1]), dtype='complex128')
    b = np.random.rand(Matrices.shape[0])
    lin_combo = sum([b[i] * Matrices[i] for i in range(Matrices.shape[0])]) #random linear combo of mult matrices to avoid issues with double eigenvalues
    vals, P = eig(lin_combo)
    vals = 0 #don't need the vals of the lin_combo matrix. Delete for memory
    if verbose:
        print("The linear combo of multiplication matrices:\n", lin_combo)
        print("The basis which simultanously diagonalizes the matrices:\n", P)

    for k in range(Matrices.shape[0]):
        A = Matrices[k] @ P
        i = np.argmax(P, axis = 0) #get largest value in each column of P
        for j in range(Matrices.shape[1]):
            sim_diag[k,j] = A[i[j]][j]/P[i[j]][j]

    #sligihtly less efficient
    # for k in range(Matrices.shape[0]):
    #     A = Matrices[k] @ P
    #     for j in range(Matrices.shape[1]):
    #         i = np.argmax(P[:,j]) #get largest value in each column of P
    #         print(A[i,j]/P[i,j])
    #         sim_diag[k,j] = A[i,j]/P[i,j]

    #brute force method
    # Pinv = np.linalg.inv(P)
    # for k in range(Matrices.shape[0]):
    #     if verbose:
    #         print("M_{} diagonalized:\n".format(k), Pinv @ Matrices[k] @ P)
    #     sim_diag[k] = (Pinv @ Matrices[k] @ P).diagonal()

    return sim_diag
