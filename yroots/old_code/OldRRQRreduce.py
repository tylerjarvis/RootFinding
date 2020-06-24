"""Old Macaulay RRQR reduction for eigenvectors"""

def rrqr_reduceMacaulay(matrix, matrix_terms, cuts, max_cond_num, macaulay_zero_tol, return_perm=False):
    ''' Reduces a Macaulay matrix, BYU style.

    The matrix is split into the shape
    A B C
    D E F
    Where A is square and contains all the highest terms, and C contains all the x,y,z etc. terms. The lengths
    are determined by the matrix_shape_stuff tuple. First A and D are reduced using rrqr without pivoting, and then the rest of
    the matrix is multiplied by Q.T to change it accordingly. Then E is reduced by rrqr with pivoting, the rows of B are shifted
    accordingly, and F is multipled by Q.T to change it accordingly. This is all done in place to save memory.

    Parameters
    ----------
    matrix : numpy array.
        The Macaulay matrix, sorted in BYU style.
    matrix_terms: numpy array
        Each row of the array contains a term in the matrix. The i'th row corresponds to
        the i'th column in the matrix.
    cuts : tuple
        When the matrix is reduced it is split into 3 parts with restricted pivoting. These numbers indicate
        where those cuts happen.
    max_cond_num : float
        Throws an error if the condition number of the backsolve is more than max_cond_num.
    macaulay_zero_tol : float
        What is considered to be 0 after the reduction. Specifically, rows where every element has
        magnitude less that macaulay_zero_tol are removed.
    return_perm : bool
        If True, also returns the permutation done by the pivoting.
    Returns
    -------
    matrix : numpy array
        The reduced matrix.
    matrix_terms: numpy array
        The resorted matrix_terms.
    perm : numpy array
        The permutation of the rows from the original. Returned only if return_perm is True.
    Raises
    ------
    ConditioningError if the conditioning number of the Macaulay matrix after
    QR is greater than max_cond_num.
    '''
    #controller variables for each part of the matrix
    AD = matrix[:,:cuts[0]]

    BCEF = matrix[:,cuts[0]:]
    # A = matrix[:cuts[0],:cuts[0]]
    B = matrix[:cuts[0],cuts[0]:cuts[1]]
    # C = matrix[:cuts[0],cuts[1]:]
    # D = matrix[cuts[0]:,:cuts[0]]
    E = matrix[cuts[0]:,cuts[0]:cuts[1]]
    F = matrix[cuts[0]:,cuts[1]:]

    #RRQR reduces A and D without pivoting sticking the result in its place.
    Q1,matrix[:,:cuts[0]] = qr(AD)
    #Conditioning check
    cond_num = np.linalg.cond(matrix[:,:cuts[0]])
    if cond_num > max_cond_num:
        raise ConditioningError("Conditioning number of the Macaulay matrix "\
                                + "after first QR is: " + str(cond_num))

    #Multiplying BCEF by Q.T
    BCEF[...] = Q1.T @ BCEF
    del Q1 #Get rid of Q1 for memory purposes.

    #Check to see if E exists
    if cuts[0] != cuts[1] and cuts[0] < matrix.shape[0]:
        #RRQR reduces E sticking the result in it's place.
        Q,E[...],P = qr(E, pivoting = True)

        #Multiplies F by Q.T.
        F[...] = Q.T @ F
        del Q #Get rid of Q for memory purposes.

        #Permute the columns of B
        B[...] = B[:,P]

        #Resorts the matrix_terms.
        matrix_terms[cuts[0]:cuts[1]] = matrix_terms[cuts[0]:cuts[1]][P]

    #use the numerical rank to determine how many rows to keep
    matrix = row_swap_matrix(matrix)[:cuts[1]]
    s = svd(matrix,compute_uv=False)
    tol = max(matrix.shape)*s[0]*macheps
    rank = len(s[s>tol])
    matrix = matrix[:rank]

    #find the condition number of the backsolve
    s = svd(matrix[:,:rank],compute_uv=False)
    cond_num = s[0]/s[-1]
    if cond_num > max_cond_num:
        raise ConditioningError("Conditioning number of backsolving the Macaulay is: " + str(cond_num))

    #backsolve
    height = matrix.shape[0]
    matrix[:,height:] = solve_triangular(matrix[:,:height],matrix[:,height:])
    matrix[:,:height] = np.eye(height)

    if return_perm:
        perm = np.arange(matrix.shape[1])
        perm[cuts[0]:cuts[1]] = perm[cuts[0]:cuts[1]][P]
        return matrix, matrix_terms, perm

    return matrix, matrix_terms
