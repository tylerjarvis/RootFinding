import numpy as np
import itertools
from math import isnan
from numpy.fft import fftn
from numpy.linalg import LinAlgError

from groebner.polynomial import MultiCheb, MultiPower
from groebner.root_finder import roots, newton_polish
from groebner.utils import clean_zeros_from_matrix
import time

def projective_solve(poly_list, rmSize = 1.e-2):
    '''Finds the roots of given polynomials using projective space.
    
    Parameters
    ----------
    poly_list : list
        A list of polynomials.
    rmSize : float
        The size of the pertubations in the rotation matrix. The rotation matrix is the identity matrix
        with pertubations of about this size in each spot to make it random.

    Returns
    -------
    zero_set : set
        A set of the distinct zeros of the system. In order to be able to put them in a set and not
        double count equivalent zeros found in sperate hyperplanes, the zeros are rounded to 5 decimal spots.
    '''
    dim = poly_list[0].dim
    inv_rotation = get_rotation_matrix(dim+1, size=rmSize) #The inverse of the matrix is how the projective space is rotated.
    proejctive_poly_list = project_poly_list(poly_list)
    all_zeros = list()
    for hyperplane in range(dim+1):
        values = list()
        cheb_poly_list = list()
        for poly in proejctive_poly_list:
            cheb = triangular_cheb_approx(poly, hyperplane, inv_rotation, dim, poly.degree)
            cheb_poly_list.append(cheb)
        zeros = roots(cheb_poly_list, method='TVB')
        for zero in zeros:
            pZero = np.insert(zero, hyperplane, 1)
            rZero = inv_rotation@pZero
            fullZero = rZero/rZero[-1]
            all_zeros.append(fullZero[:-1])
    return getZeroSet(all_zeros, poly_list)

def get_rotation_matrix(dim, size = 1.e-5):
    '''Finds the roots of given polynomials using projective space.
    
    Parameters
    ----------
    dim : int
        .
    rmSize : float
        The size of the pertubations in the rotation matrix. The rotation matrix is the identity matrix
        with pertubations of about this size in each spot to make it random.

    Returns
    -------
    zero_set : set
        A set of the distinct zeros of the system. In order to be able to put them in a set and not
        double count equivalent zeros found in sperate hyperplanes, the zeros are rounded to 5 decimal spots.
    '''
    A = np.eye(dim)
    A += np.random.rand(dim,dim)*size
    return A

def cheb_interpND(poly, hyperplane, inv_rotation, dim, deg):
    '''Gives an n-dimensional interpolation of polynomial on a given hyperplace after a given rotation.
    
    It finds the chebyshev nodes in the given hyperplane, projects them back into projective space, uses inv_rotation
    to rotate them back, and then evaluates them on poly to get the values of the cheb nodes on the rotated projected
    polynomial, but in a stable way. The chebyshev coefficients are then found using a fast fourier transform.
    
    Parameters
    ----------
    poly : Polynomial
        This is one of the given polynomials in projective space. So it must be projected into proejective space first.
    hyperplace : int
        Which hyperplance we want to approximate in. Between 0 and n inclusive when the original polynomials are n-1
        dimensional. So the projective space polynomials are n dimensional. n is the original space.
    inv_rotation : numpy array
        The inverse of the rotation of the projective space.
    dim : int
        The dimension of the chebysehv polynomial we want.
    deg : int
        The degree of the chebyshev polynomial we want.

    Returns
    -------
    coeffs : numpy array
        The coefficients of the chebyshev polynomial.
    '''
    nodes = getChebNodes(dim,deg)
    newLevel = np.ones_like(nodes[0])
    shape = [1]+list(newLevel.shape)
    newLevel = newLevel.reshape(shape)
    final = np.concatenate((nodes, newLevel), axis = 0)
    final[-1], final[hyperplane] = final[hyperplane], final[-1].copy()
    rotated = np.apply_along_axis(mult,0,final,inv_rotation)
    #values = np.apply_along_axis(evaluate,0,rotated,poly)
    values = poly.evaluate_at(rotated)
    coeffs = np.real(fftn(values/deg**dim))

    for i in range(dim):
        idx0 = [slice(None)] * (dim)
        idx0[i] = 0
        idx00 = [slice(None)] * (dim)
        idx00[i] = deg
        coeffs[idx0] = coeffs[idx0]/2
        coeffs[idx00] = coeffs[idx00]/2
    
    slices = list()
    for i in range(dim):
        slices.append(slice(0,deg+1))

    return coeffs[slices]

def triangular_cheb_approx(poly, hyperplane, inv_rotation, dim, deg, accuracy = 1.e-10):
    '''Gives an n-dimensional triangular interpolation of polynomial on a given hyperplace after a given rotation.
    
    It calls the normal nD-interpolation, but then cuts off the small non-triangular part to make it triangular.
    
    Parameters
    ----------
    poly : Polynomial
        This is one of the given polynomials in projective space. So it must be projected into proejective space first.
    hyperplace : int
        Which hyperplance we want to approximate in. Between 0 and n inclusive when the original polynomials are n-1
        dimensional. So the projective space polynomials are n dimensional. n is the original space.
    inv_rotation : numpy array
        The inverse of the rotation of the projective space.
    dim : int
        The dimension of the chebysehv polynomial we want.
    deg : int
        The degree of the chebyshev polynomial we want.

    Returns
    -------
    triangular_cheb_approx : MultiCheb
        The chebyshev polynomial we want.
    '''
    cheb = cheb_interpND(poly, hyperplane, inv_rotation, dim, deg)
    clean_zeros_from_matrix(cheb)
    return MultiCheb(cheb)    

def getChebNodes(dim, cheb_deg):
    '''Gets the chebyshev nodes to approximate a chebshev polynomial of the given dimension and degree.
        
    Parameters
    ----------
    dim : int
        The dimension of the chebysehv polynomial we want.
    cheb_deg : int
        The degree of the chebyshev polynomial we want.

    Returns
    -------
    getChebNodes : numpy arrary
        The array has dimension dim+1. Iterating through along the first dimension gives the tuples that are
        the Chebyshev node coordinates. So each Chebyshev node has one point in each of the matrixes in the
        final stack.
    '''
    cheb_nodes = np.cos((np.pi*np.arange(2*cheb_deg))/cheb_deg)
    nodes = np.array(list(itertools.product(cheb_nodes, repeat=dim)))
    stacks = list()
    for i in range(dim):
        stacks.append(nodes[:,i].reshape([2*cheb_deg]*dim))
    return np.stack(stacks)

def evaluate(x,poly):
    '''Evaluates a polynomial at a point. Useful so we can use it in a np.apply_along_axis.
        
    Parameters
    ----------
    x : tuple
        The point at which we want to evaluate the polynomial.
    poly : Polynomial
        The polynomial we are evaluating on.

    Returns
    -------
    evaluate : float
        The evaluated value.
    '''
    return poly.evaluate_at(x)

def mult(x,A):
    '''Multiplies a vector by a matrix A. Useful so we can use it in a np.apply_along_axis.
        
    Parameters
    ----------
    x : numpy array
        A one-dimensional vector.
    A : numpy array
        The matrix we multiply the vector by.

    Returns
    -------
    mult : float
        The evaluated value.
    '''
    return A@x

def project_poly_list(poly_list):
    '''Projects the polynomials in a list into projective space.
    
    Parameters
    ----------
    poly_list : list
        A list of polynomials.

    Returns
    -------
    projected_list : list
        The same polynomails in projective space.
    '''
    projected_list = list()
    for poly in poly_list:
        projected_list.append(project(poly))
    return projected_list

def project(poly):
    '''Projects a polynomial into projective space.
    
    Parameters
    ----------
    poly : Polynomial
        The polynomial to project.

    Returns
    -------
    project : Polynomial
        The same polynomail in projective space.
    '''
    Pcoeff = np.zeros([poly.degree+1]*(poly.dim+1))
    for spot in zip(*np.where(poly.coeff != 0)):
        new_spot = list(spot)
        new_spot = new_spot + [poly.degree - np.sum(spot)]
        Pcoeff[tuple(new_spot)] = poly.coeff[spot]
    if isinstance(poly,MultiPower):
        return MultiPower(Pcoeff)
    else:
        return MultiCheb(Pcoeff)

def getZeroSet(zeros, polys):
    '''Finds the number of distinct zeros given a list of possibly repeating zeros.
    
    Parameters
    ----------
    zeros: list
        A list of zeros of the polynomias system.
    polys : list
        A list of the polynomials we want the common zeros of.

    Returns
    -------
    zeroSet : set
        A set of the common zeros, rounded to 5 decimal places.
    '''
    dim = polys[0].dim
    zeroSet = set()
    for zero in zeros:
        
        #good = True
        #for poly in polys:
        #    if abs(poly.evaluate_at(zero)) > 1:
        #        good = False
        #        break
        #if not good:
        #    continue
        
        inf = False
        try:
            polished = newton_polish(polys,zero)
        except LinAlgError as e:
            inf = True
        if inf or isnan(polished[0].real):
            continue
        good = True
        for poly in polys:
            if abs(poly.evaluate_at(polished)) > 1.e-3:
                good = False
                break
        if not good:
            continue
        rounded = list([0]*dim)
        for i in range(dim):
            rounded[i] = complex(round(polished[i].real,8),round(polished[i].imag,8))
        rounded = tuple(rounded)
        zeroSet.add(rounded)
    return zeroSet