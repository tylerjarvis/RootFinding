import numpy as np
import itertools
from groebner.utils import clean_zeros_from_matrix, get_var_list
from groebner.TelenVanBarel import create_matrix, add_polys
from groebner.root_finder import _random_poly
from scipy.linalg import solve_triangular
from scipy.stats import mode
from scipy.linalg import qr

def division_power(test_polys):
    dim = test_polys[0].dim
    
    matrix_degree = np.sum(poly.degree for poly in test_polys) - len(test_polys) + 1

    poly_coeff_list = []
    for i in test_polys:
        poly_coeff_list = add_polys(matrix_degree, i, poly_coeff_list)
    matrix, matrix_terms, matrix_shape_stuff = create_matrix(poly_coeff_list, matrix_degree, dim)
    #matrix, matrix_terms, matrix_shape_stuff = createMatrix2(test_polys, matrix_degree, dim)
    perm, cut = matrix_term_perm(matrix_terms)
    matrix = matrix[:,perm]
    matrix_terms = matrix_terms[perm]

    #Reduce it.
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
    basisDict = makeBasisDict(matrixFinal, matrix_terms, VB, True, [matrix_degree+1]*dim)

    slices = list()
    for i in range(len(VB[0])):
        slices.append(VB.T[i])

    VBset = set()
    for mon in VB:
        VBset.add(tuple(mon))

    # Build division matrix
    dMatrix = np.zeros((len(VB), len(VB)))
    remainder = np.zeros([matrix_degree+1]*dim)

    for i in range(VB.shape[0]):
        var = [1,0]
        term = tuple(VB[i] - var)
        if term in VBset:
            remainder[term] += 1
        else:
            remainder[slices] -= basisDict[term][slices]
        dMatrix[:,i] = remainder[slices]
        remainder[slices] = 0
    vals, vecs = np.linalg.eig(dMatrix.T)    
    
    VBdict = {}
    num = 0
    for row in VB:
        VBdict[tuple(row)] = num
        num+=1

    xs = list()
    ys = list()
    for row in VB:
        if tuple(row-[1,0]) in VBset:
            xs.append(VBdict[tuple(row)])
            xs.append(VBdict[tuple(row-[1,0])])
            break
        elif tuple(row+[1,0]) in VBset:
            xs.append(VBdict[tuple(row+[1,0])])
            xs.append(VBdict[tuple(row)])
            break
    for row in VB:
        if tuple(row-[0,1]) in VBset:
            ys.append(VBdict[tuple(row)])
            ys.append(VBdict[tuple(row-[0,1])])
            break
        elif tuple(row+[0,1]) in VBset:
            ys.append(VBdict[tuple(row+[0,1])])
            ys.append(VBdict[tuple(row)])
            break

    newZeros = list()
    for v in vecs.T:
        newZeros.append(np.array([v[xs[0]]/v[xs[1]], v[ys[0]]/v[ys[1]]]))
    
    return newZeros

def checkEqual(lst):
    return lst.count(lst[0]) == len(lst)

def get_ranges(nums):
    ranges = []
    for i in nums:
        ranges.append(np.array([],dtype=int))
    start = 0
    count = 0
    n = len(nums)
    for num in nums:
        spot = count
        for r in ranges[count:]:
            r = np.hstack((r,np.arange(start,start+(n-count)*(num-len(r)),n-count)))
            ranges[spot] = r
            start+=1
            spot += 1
        start = ranges[-1][-1]+1
        count+=1
    return ranges

def matrix_term_perm(matrix_terms):
    boundary = np.where(matrix_terms[:,0] == 0)[0]
    #boundary = np.unique(np.where(matrix_terms == 0)[0])
    bSet = set(boundary)
    other = np.arange(len(matrix_terms))
    mask = [num not in bSet for num in other]
    other = other[mask]
    return np.hstack((boundary,other)), len(boundary)

def chebify_polys(polys, scale = 2):
    for poly in polys:
        for spot,val in np.ndenumerate(poly.coeff):
            if np.sum(spot) == 0:
                continue
            poly.coeff[spot] = val/(scale**np.sum(spot))
    return polys

def makeBasisDict(matrix, matrix_terms, VB, power, remainder_shape):
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
    power : bool
        If True, the initial polynomials were MultiPower. If False, they were MultiCheb.
    remainder_shape: list
        The shape of the numpy arrays that will be mapped to in the basisDict.

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
    
    if power: #We don't actually need most of the rows, so we only get the ones we need.
        neededSpots = set()
        for term, mon in itertools.product(VB,get_var_list(VB.shape[1])):
            if tuple(term-mon) not in VBSet:
                neededSpots.add(tuple(term-mon))

    spots = list()
    for dim in range(VB.shape[1]):
        spots.append(VB.T[dim])

    for i in range(matrix.shape[0]):
        term = tuple(matrix_terms[i])
        
        if power and term not in neededSpots:
            continue
        #print('USED')
        remainder = np.zeros(remainder_shape)
        row = matrix[i]
        remainder[spots] = row[matrix.shape[0]:]
        basisDict[term] = remainder

    return basisDict