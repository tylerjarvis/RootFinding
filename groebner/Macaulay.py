from operator import itemgetter
import itertools
import numpy as np
from groebner import maxheap
import math
from groebner.multi_cheb import MultiCheb
from groebner.multi_power import MultiPower
from groebner.polynomial import Polynomial
from scipy.linalg import lu, qr, solve_triangular, inv, solve, svd
from numpy.linalg import cond
from scipy.sparse import csc_matrix, vstack
from groebner.maxheap import Term
import matplotlib.pyplot as plt
import time
from collections import defaultdict

def Macaulay(initial_poly_list, global_accuracy = 1.e-10):
    """
    Macaulay will take a list of polynomials and use them to construct a Macaulay matrix.

    parameters
    --------
    initial_poly_list: A list of polynomials
    global_accuracy: How small we want a number to be before assuming it is zero.
    --------

    Returns
    -----------
    Reduced Macaulay matrix that can be passed into the root finder.
    -----------
    """
    times = {}
    startTime = time.time()
    MultiCheb.clearTime()
    MultiPower.clearTime()
    Polynomial.clearTime()

    Power = bool
    if all([type(p) == MultiPower for p in initial_poly_list]):
        Power = True
    elif all([type(p) == MultiCheb for p in initial_poly_list]):
        Power = False
    else:
        print([type(p) == MultiPower for p in initial_poly_list])
        raise ValueError('Bad polynomials in list')

    poly_coeff_list = []
    degree = find_degree(initial_poly_list)
    
    startAdding = time.time()
    for i in initial_poly_list:
        poly_coeff_list = add_polys(degree, i, poly_coeff_list)
    endAdding = time.time()
    times["adding polys"] = (endAdding - startAdding)
    
    
    startCreate = time.time()
    matrix, matrix_terms = create_matrix(poly_coeff_list)
    endCreate = time.time()
    times["create matrix"] = (endCreate - startCreate)
    #print(matrix.shape)
    
    return matrix, matrix_terms   #Take this out when done with Testing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    #plt.matshow([i==0 for i in matrix])
    
    original = matrix
    
    startReduce = time.time()
    #rrqr_reduce2 and rrqr_reduce same pretty matched on stability, though I feel like 2 should be better.
    matrix = rrqr_reduce2(matrix, global_accuracy = global_accuracy)
    matrix = clean_zeros_from_matrix(matrix)
    non_zero_rows = np.sum(abs(matrix),axis=1) != 0
    matrix = matrix[non_zero_rows,:] #Only keeps the non_zero_polymonials
    endReduce = time.time()
    times["reduce matrix"] = (endReduce - startReduce)
    
    #plt.matshow([i==0 for i in matrix])
    
    startTri = time.time()
    triangle, order = triangular_solve(matrix)
    #matrix = clean_zeros_from_matrix(matrix)
    endTri = time.time()
    times["triangular solve"] = (endTri - startTri)
    '''
    #plt.matshow([i==0 for i in matrix])
    
    #return original, triangle, matrix_terms, order #This is for notebook testing
    
    P = inverse_P(order)
    original = original[:,P]
    x = original.shape[0]
    M1 = original[:x,:x]
    M2 = original[:,x:]
    
    #Shift the rows and columns to try and make it better
    length = M1.shape[1]
    rowChange = np.eye(length)
    #diag = np.eye(length)
    #diag = varDiag(M1)
    print(cond(M1),cond(M2))
    diag = np.diag(M1.diagonal())
    #rowChange = varDiag2(M1@diag)
    #diag = inv(np.diag(triangle.diagonal()))
    M1 = rowChange@M1@diag
    M2 = rowChange@M2
    
    #inverse = inv(M1)
    #M1 = inverse@M1
    #M2 = inverse@M2
    
    print(cond(M1),cond(M2))
    
    #newR = solve(M1,M2)
    newR = np.linalg.solve(M1,M2)
    
    #rows = get_good_rows(matrix, matrix_terms)
    
    #i = rows[::-1][0]
    #j = np.where(M1[:,i:i+1] != 0)[0][::-1][0] + 1
    #M1new = M1[j:,j:]
    #M2new = M2[j:]
    #newRnew = solve(M1new,M2new)
    #newR[j:] = newRnew
    
    triangle[:,x:] = newR
    matrix = triangle[:,order]
    #return M1,newR,M2
    #matrix = np.hstack((inv(diag), newR))[:,order]

    #matrix = clean_zeros_from_matrix(matrix)
    '''
    matrix = triangle[:,order]
    
    startGetPolys = time.time()
    rows = get_good_rows(matrix, matrix_terms)
    final_polys = get_poly_from_matrix(rows,matrix,matrix_terms,Power)
    endGetPolys = time.time()
    times["get polys"] = (endGetPolys - startGetPolys)
    
    #return M1,newR,M2,rows
    
    endTime = time.time()
    #print("Macaulay run time is {} seconds".format(endTime-startTime))
    #print(times)
    #MultiCheb.printTime()
    #MultiPower.printTime()
    #Polynomial.printTime()
    #for poly in final_polys:
    #    print(poly.lead_term)
    return final_polys

def varDiag2(matrix, inc = 1.e-1):
    '''
    Tries to reduce the cond of a matrix by multiplying it by a diagonal of varied coefficients.
    This one scales the rows.
    '''
    values = np.ones(matrix.shape[0])
    C = cond(matrix)
    print(C)
    change = True
    while change:
        change = False
        for i in range(len(values)):
            #print(i)
            improve = True
            while improve:
                improve = False
                values[i] += inc
                if abs(values[i]) < inc/2:
                    values[i] += inc
                newC = cond(np.diag(values)@matrix)
                if newC < C:
                    C = newC
                    #print(C)
                    improve = True
                    #change = True
                else:
                    values[i] -= inc
                    if abs(values[i]) < inc/2:
                        values[i] -= inc
            improve = True
            while improve:
                improve = False
                values[i] -= inc
                if abs(values[i]) < inc/2:
                    values[i] -= inc
                newC = cond(np.diag(values)@matrix)
                if newC < C and values[i] != 0:
                    C = newC
                    #print(C)
                    improve = True
                    #change = True
                else:
                    values[i] += inc
                    if abs(values[i]) < inc/2:
                        values[i] += inc
    print(C)
    return np.diag(values)

def varDiag(matrix, inc = 1.e-1):
    '''
    Tries to reduce the cond of a matrix by multiplying it by a diagonal of varied coefficients.
    This one scales the columns.
    '''
    values = np.ones(matrix.shape[0])
    C = cond(matrix)
    print(C)
    for i in range(len(values)):
        improve = True
        while improve:
            improve = False
            values[i] += inc
            newC = cond(matrix@np.diag(values))
            if newC < C:
                C = newC
                improve = True
            else:
                values[i] -= inc
        improve = True
        while improve:
            improve = False
            values[i] -= inc
            newC = cond(matrix@np.diag(values))
            if newC < C and values[i] != 0:
                C = newC
                improve = True
            else:
                values[i] += inc
    print(C)
    return np.diag(values)

def triangular_solve(matrix):
    " Reduces the upper block triangular matrix. "
    m,n = matrix.shape
    j = 0  # The row index.
    k = 0  # The column index.
    c = [] # It will contain the columns that make an upper triangular matrix.
    d = [] # It will contain the rest of the columns.
    order_c = [] # List to keep track of original index of the columns in c.
    order_d = [] # List to keep track of the original index of the columns in d.

    # Checks if the given matrix is not a square matrix.
    if m != n:
        # Makes sure the indicies are within the matrix.
        while j < m and k < n:
            if matrix[j,k]!= 0:
                c.append(matrix[:,k])
                order_c.append(k)
                # Move to the diagonal if the index is non-zero.
                j+=1
                k+=1
            else:
                d.append(matrix[:,k])
                order_d.append(k)
                # Check the next column in the same row if index is zero.
                k+=1
        # C will be the square matrix that is upper triangular with no zeros on the diagonals.
        C = np.vstack(c).T
        # If d is not empty, add the rest of the columns not checked into the matrix.
        if d:
            D = np.vstack(d).T
            D = np.hstack((D,matrix[:,k:]))
        else:
            D = matrix[:,k:]
        # Append the index of the rest of the columns to the order_d list.
        for i in range(n-k):
            order_d.append(k)
            k+=1

        # Solve for the CX = D
        #return np.hstack((C,D)), inverse_P(order_c+order_d) #Just for testing?
        X = solve_triangular(C,D)

        # Add I to X. [I|X]
        solver = np.hstack((np.eye(X.shape[0]),X))

        # Find the order to reverse the columns back.
        order = inverse_P(order_c+order_d)

        # Reverse the columns back.
        
        
        #solver = solver[:,order] #PUT THIS BACK IN WHEN DONE TESTING
        
        
        # Temporary checker. Plots the non-zero part of the matrix.
        #plt.matshow(~np.isclose(solver,0))
        return solver, order #JUST RETURN SOLVER WHEN DONE TESTING

    else:
        # The case where the matrix passed in is a square matrix
        return np.eye(m)
    pass

def get_poly_from_matrix(rows,matrix,matrix_terms,power):
    '''
    Takes a list of indicies corresponding to the rows of the reduced matrix and
    returns a list of polynomial objects
    '''
    shape = []
    p_list = []
    matrix_term_vals = [i.val for i in matrix_terms]

    # Finds the maximum size needed for each of the poly coeff tensors
    for i in range(len(matrix_term_vals[0])):
        # add 1 to each to compensate for constant term
        shape.append(max(matrix_term_vals, key=itemgetter(i))[i]+1)
    # Grabs each polynomial, makes coeff matrix and constructs object
    for i in rows:
        p = matrix[i]
        coeff = np.zeros(shape)
        for j,term in enumerate(matrix_term_vals):
            coeff[term] = p[j]

        if power:
            poly = MultiPower(coeff)
        else:
            poly = MultiCheb(coeff)

        if poly.lead_term != None:
            p_list.append(poly)
    return p_list

def divides(a,b):
    '''
    Takes two terms, a and b. Returns True if b divides a. False otherwise.
    '''
    diff = tuple(i-j for i,j in zip(a.val,b.val))
    return all(i >= 0 for i in diff)

def get_good_rows(matrix, matrix_terms):
    '''
    Gets the rows in a matrix whose leading monomial is not divisible by the leading monomial of any other row.
    Returns a list of rows.
    This function could probably be improved, but for now it is good enough.
    '''
    rowLMs = dict()
    already_looked_at = set()
    #Finds the leading terms of each row.
    for i, j in zip(*np.where(matrix!=0)):
        if i in already_looked_at:
            continue
        else:
            already_looked_at.add(i)
            rowLMs[i] = matrix_terms[j]
    keys= list(rowLMs.keys())
    keys = keys[::-1]
    spot = 0
    #Uses a sieve to find which of the rows to keep.
    while spot != len(keys):
        term1 = rowLMs[keys[spot]]
        toRemove = list()
        for i in range(spot+1, len(keys)):
            term2 = rowLMs[keys[i]]
            if divides(term2,term1):
                toRemove.append(keys[i])
        for i in toRemove:
            keys.remove(i)
        spot += 1
    return keys

def find_degree(poly_list):
    """
    Takes a list of polynomials and finds the degree needed for a Macaulay matrix.
    Adds the degree of each polynomial and then subtracts the total number of polynomials and adds one.

    Example:
        For polynomials [P1,P2,P3] with degree [d1,d2,d3] the function returns d1+d2+d3-3+1

    """
    degree_needed = 0
    for poly in poly_list:
        degree_needed += poly.degree
    return ((degree_needed - len(poly_list)) + 1)

def mon_combos(mon, numLeft, spot = 0):
    '''
    This function finds all the monomials up to a given degree (here numLeft) and returns them.
    mon is a tuple that starts as all 0's and gets changed as needed to get all the monomials.
    numLeft starts as the dimension, but as the code goes is how much can still be added to mon.
    spot is the place in mon we are currently adding things too.
    Returns a list of all the possible monomials.
    '''
    answers = list()
    if len(mon) == spot+1: #We are at the end of mon, no more recursion.
        for i in range(numLeft+1):
            mon[spot] = i
            answers.append(mon.copy())
        return answers
    if numLeft == 0: #Nothing else can be added.
        answers.append(mon.copy())
        return answers
    temp = mon.copy() #Quicker than copying every time inside the loop.
    for i in range(numLeft+1): #Recursively add to mon further down.
        temp[spot] = i
        answers += mon_combos(temp, numLeft-i, spot+1)
    return answers

def add_polys(degree, poly, poly_coeff_list):
    """
    Take each polynomial and adds it to a poly_list
    Then uses monomial multiplication and adds all polynomials with degree less than
        or equal to the total degree needed.
    Returns a list of polynomials.
    """
    poly_coeff_list.append(poly.coeff)
    deg = degree - poly.degree
    dim = poly.dim
    mons = mon_combos(np.zeros(dim, dtype = int),deg)
    mons = mons[1:]
    for i in mons:
        poly_coeff_list.append(poly.mon_mult(i, returnType = 'Matrix'))
    return poly_coeff_list

def row_swap_matrix(matrix):
    '''
    Rearange the rows of the matrix so it starts close to upper traingular and return it.
    '''
    rows, columns = np.where(matrix != 0)
    lms = {}
    last_i = -1
    lms = list()
    #Finds the leading column of each row and adds it to lms.
    for i,j in zip(rows,columns):
        if i == last_i:
            continue
        else:
            lms.append(j)
            last_i = i
    #Get the list by which we sort the matrix, first leading columns first.
    argsort_list = sorted(range(len(lms)), key=lms.__getitem__)[::]
    return matrix[argsort_list]

def fill_size(bigShape,smallPolyCoeff):
    '''
    Pads the smallPolyCoeff so it has the same shape as bigShape. Does this by making a matrix with the shape of
    bigShape and then dropping smallPolyCoeff into the top of it with slicing.
    Returns the padded smallPolyCoeff.
    '''
    if (smallPolyCoeff.shape == bigShape).all():
        return smallPolyCoeff
    matrix = np.zeros(bigShape)

    slices = list()
    for i in smallPolyCoeff.shape:
        s = slice(0,i)
        slices.append(s)
    matrix[slices] = smallPolyCoeff
    return matrix

def sort_matrix(matrix, matrix_terms):
    '''
    Takes a matrix and matrix_terms (holding the terms in each column of the matrix), and sorts them both
    by term order.
    Returns the sorted matrix and matrix_terms.
    '''
    #argsort_list gives the ordering by which the matrix should be sorted.
    argsort_list = sorted(range(len(matrix_terms)), key=matrix_terms.__getitem__)[::-1]
    matrix_terms.sort()
    matrix = matrix[:,argsort_list]
    return matrix, matrix_terms[::-1]

def clean_matrix(matrix, matrix_terms):
    '''
    Gets rid of columns in the matrix that are all zero and returns it and the updated matrix_terms.
    '''
    non_zero_monomial = np.sum(abs(matrix), axis=0) != 0
    matrix = matrix[:,non_zero_monomial] #Only keeps the non_zero_monomials
    matrix_terms = matrix_terms[non_zero_monomial] #Only keeps the non_zero_monomials
    return matrix, matrix_terms

def create_matrix(polys_coeffs):
    '''
    Takes a list of polynomial objects (polys) and uses them to create a matrix. That is ordered by the monomial
    ordering. Returns the matrix and the matrix_terms, a list of the monomials corresponding to the rows of the matrix.
    '''
    #Gets an empty polynomial whose lm all other polynomial divide into.
    bigShape = np.maximum.reduce([coeff.shape for coeff in polys_coeffs])
    #Gets a list of all the flattened polynomials.
    flat_polys = list()
    for coeff in polys_coeffs:
        #Gets a matrix that is padded so it is the same size as biggest, and flattens it. This is so
        #all flattened polynomials look the same.
        newMatrix = fill_size(bigShape, coeff)
        flat_polys.append(newMatrix.ravel())
    
    #Make the matrix
    matrix = np.vstack(flat_polys[::-1])

    #Makes matrix_terms, a list of all the terms in the matrix.
    #startTerms = time.time()
    terms = np.zeros(bigShape, dtype = Term)
    for i,j in np.ndenumerate(terms):
        terms[i] = Term(i)
    matrix_terms = terms.ravel()
    #endTerms = time.time()
    #print(endTerms - startTerms)
    
    #Gets rid of any columns that are all 0.
    matrix, matrix_terms = clean_matrix(matrix, matrix_terms)

    #Sorts the matrix and matrix_terms by term order.
    #matrix, matrix_terms = sort_matrix(matrix, matrix_terms)

    #Sorts the rows of the matrix so it is close to upper triangular.
    matrix = row_swap_matrix(matrix)
    return matrix, matrix_terms

def create_matrix2(polys):
    '''
    Takes a list of polynomial objects (polys) and uses them to create a matrix. That is ordered by the monomial
    ordering. Returns the matrix and the matrix_terms, a list of the monomials corresponding to the rows of the matrix.
    '''
    #Gets an empty polynomial whose lm all other polynomial divide into.
    termSet = set()
    for poly in polys:
        for i in zip(*np.where(poly.coeff != 0)):
            termSet.add(i)

    matrix_terms = np.zeros(len(termSet), dtype = Term)
    spot = 0
    for term in termSet:
        matrix_terms[spot] = Term(term)
        spot += 1
    matrix_terms = np.sort(matrix_terms)[::-1]

    termSpots = {}

    for i in range(len(matrix_terms)):
        termSpots[matrix_terms[i].val] = i
    matrix = np.random.rand(0,len(matrix_terms))
    
    start = time.time()
    for poly in polys:
        matrix = np.vstack((matrix, np.zeros(matrix.shape[1])))
        for term in zip(*np.where(poly.coeff != 0)):
            matrix[-1,termSpots[term]] = poly.coeff[term]
    end = time.time()
    print(end-start)
    
    #Sorts the rows of the matrix so it is close to upper triangular.
    matrix = row_swap_matrix(matrix)
    return matrix, matrix_terms


def rrqr_reduce(matrix, clean = True, global_accuracy = 1.e-10): #Appears to work best when clean = True
    '''
    Recursively reduces the matrix using rrqr reduction so it returns a reduced matrix, where each row has
    a unique leading monomial.
    '''
    if matrix.shape[0]==0 or matrix.shape[1]==0:
        return matrix
    height = matrix.shape[0]
    A = matrix[:height,:height] #Get the square submatrix
    B = matrix[:,height:] #The rest of the matrix to the right
    Q,R,P = qr(A, pivoting = True) #rrqr reduce it
    PT = inverse_P(P)
    diagonals = np.diagonal(R) #Go along the diagonals to find the rank
    rank = np.sum(np.abs(diagonals)>global_accuracy)
    if rank == height: #full rank, do qr on it
        Q,R = qr(A)
        A = R #qr reduce A
        B = Q.T.dot(B) #Transform B the same way
    else: #not full rank
        A = R[:,PT] #Switch the columns back
        if clean:
            Q[np.where(abs(Q) < global_accuracy)]=0
        B = Q.T.dot(B) #Multiply B by Q transpose
        if clean:
            B[np.where(abs(B) < global_accuracy)]=0
        #sub1 is the top part of the matrix, we will recursively reduce this
        #sub2 is the bottom part of A, we will set this all to 0
        #sub3 is the bottom part of B, we will recursively reduce this.
        #All submatrices are then put back in the matrix and it is returned.
        sub1 = np.hstack((A[:rank,],B[:rank,])) #Takes the top parts of A and B
        result = rrqr_reduce(sub1) #Reduces it
        A[:rank,] = result[:,:height] #Puts the A part back in A
        B[:rank,] = result[:,height:] #And the B part back in B

        sub2 = A[rank:,]
        zeros = np.zeros_like(sub2)
        A[rank:,] = np.zeros_like(sub2)

        sub3 = B[rank:,]
        B[rank:,] = rrqr_reduce(sub3)

    reduced_matrix = np.hstack((A,B))
    return reduced_matrix


def inverse_P(p):
    P = np.eye(len(p))[:,p]
    return np.where(P==1)[1]

def clean_zeros_from_matrix(matrix, global_accuracy = 1.e-10):
    '''
    Sets all points in the matrix less than the gloabal accuracy to 0.
    '''
    matrix[np.where(np.abs(matrix) < global_accuracy)]=0
    return matrix

def fullRank(matrix, global_accuracy = 1.e-10):
    '''
    Finds the full rank of a matrix.
    Returns independentRows - a list of rows that have full rank, and 
    dependentRows - rows that can be removed without affecting the rank
    Q - The Q matrix used in RRQR reduction in finding the rank
    '''
    height = matrix.shape[0]
    Q,R,P = qr(matrix, pivoting = True)
    diagonals = np.diagonal(R) #Go along the diagonals to find the rank
    #print(diagonals)
    rank = np.sum(np.abs(diagonals)>global_accuracy)
    numMissing = height - rank
    if numMissing == 0: #Full Rank. All rows independent
        return [i for i in range(height)],[],None
    else:
        #Find the rows we can take out. These are ones that are non-zero in the last rows of Q transpose, as QT*A=R.
        #To find multiple, we find the pivot columns of Q.T
        QMatrix = Q.T[-numMissing:]
        Q1,R1,P1 = qr(QMatrix, pivoting = True)
        independentRows = P1[R1.shape[0]:] #Other Columns
        dependentRows = P1[:R1.shape[0]] #Pivot Columns
        return independentRows, dependentRows, QMatrix
    pass

def rrqr_reduce2(matrix, clean = False, global_accuracy = 1.e-10): #Appears to work best when clean = False
    '''
    This function does the same thing as rrqr_reduce. It is an attempt at higher stability, appears slighlty more stable.
    '''
    if matrix.shape[0] <= 1 or matrix.shape[0]==1 or  matrix.shape[1]==0:
        return matrix
    height = matrix.shape[0]
    A = matrix[:height,:height] #Get the square submatrix
    B = matrix[:,height:] #The rest of the matrix to the right
    independentRows, dependentRows, QMatrix = fullRank(A, global_accuracy = global_accuracy)
    nullSpaceSize = len(dependentRows)
    if nullSpaceSize == 0: #A is full rank
        Q,R = qr(matrix)
        return R
    else: #A is not full rank
        #sub1 is the independentRows of the matrix, we will recursively reduce this
        #sub2 is the dependentRows of A, we will set this all to 0
        #sub3 is the dependentRows of Q.T@B, we will recursively reduce this.
        #We then return sub1 stacked on top of sub2+sub3
        if clean:
            QMatrix[np.where(abs(QMatrix) < global_accuracy)]=0
        bottom = matrix[dependentRows]
        sub3 = bottom[:,height:]
        sub3 = QMatrix@B
        
        sub3 = rrqr_reduce2(sub3)

        sub1 = matrix[independentRows]
        sub1 = rrqr_reduce2(sub1)            

        sub2 = bottom[:,:height]
        sub2[:] = np.zeros_like(sub2)

        reduced_matrix = np.vstack((sub1,np.hstack((sub2,sub3))))
        return reduced_matrix
    pass
