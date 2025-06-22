import numpy as np

def swap_rows(A, i, j):
    '''
    Inputs:
    - A: numpy.ndarray, matrix
    - i: int, index of the first row
    - j: int, index of the second row

    Outputs:
    - numpy.ndarray, matrix with swapped rows
    '''
    A[[i, j]] = A[[j, i]]
    return A

def multiply_row(A, i, scalar):
    '''
    Inputs:
    - A: numpy.ndarray, matrix
    - i: int, index of the row
    - scalar: float, scalar to multiply the row with

    Outputs:
    - numpy.ndarray, matrix with multiplied row
    '''
    A[i] = A[i] * scalar
    return A

def add_row(A, i, j, scalar=1):
    '''
    Inputs:
    - A: numpy.ndarray, matrix
    - i: int, index of the row to be added to
    - j: int, index of the row to be added

    Outputs:
    - numpy.ndarray, matrix with added rows
    '''
    A[i] = A[i] + A[j]*scalar
    return A

def perform_gaussian_elemination(A):
    '''
    Inputs:
    - A: numpy.ndarray, matrix of shape (dim, dim)

    Outputs:
    - ops: List[Tuple[str,int,int]], sequence of elementary operations
    - A_inv: numpy.ndarray, inverse of A
    '''
    dim = A.shape[0]
    A_inv = np.eye(dim)
    ops = []
    A = A.astype(float)
    ########################################################################
    # TODO:                                                                #
    # Implement the Gaussian elemination algorithm.                        #
    # Return the sequence of elementary operations and the inverse matrix. #
    #                                                                      #
    # The sequence of the operations should be in the following format:    #
    # • to swap to rows                                                    #
    #   ("S",<row index>,<row index>)                                      #
    # • to multiply the row with a number                                  #
    #   ("M",<row index>,<number>)                                         #
    # • to add multiple of one row to another row                          #
    #   ("A",<row index i>,<row index j>, <number>)                        #
    # Be aware that the rows are indexed starting with zero.               #
    # Output sufficient number of significant digits for numbers.          #
    # Output integers for indices.                                         #
    #                                                                      #
    # Append to the sequence of operations                                 #
    # • "DEGENERATE" if you have successfully turned the matrix into a     #
    #   form with a zero row.                                              #
    # • "SOLUTION" if you turned the matrix into the $[I|A −1 ]$ form.     #
    #                                                                      #
    # If you found the inverse, output it as a second element,             #
    # otherwise return None as a second element                            #
    ########################################################################
    for i in range(dim):
        
        # find the rows in [i: dim-1] which has the largest i-th item
        mx_row = i
        for j in range(i, dim):
            if abs(A[j][i]) >= abs(A[mx_row][i]):
                mx_row = j
        if A[mx_row][i] == 0: continue # if the largest one is still 0

        # get row enchelon form 
        
        if i != dim-1:  # not the last row
            A = swap_rows(A, i, mx_row)
            A_inv = swap_rows(A_inv, i, mx_row)
            ops.append(("S", i, mx_row))
            # print(A, "S")

        
        ## scale enchelon to be 1
        r = 1/A[i][i]
        A = multiply_row(A, i, r)
        A_inv = multiply_row(A_inv, i, r)
        ops.append(("M", i, r))
        # print(A, "M")

        ## minus the row below
        for j in range(i+1, dim):
            # minus the row-i with scalar
            r = -A[j][i]
            A = add_row(A, j, i, r)
            A_inv = add_row(A_inv, j, i, r)
            ops.append(("A",j, i, r))
        # print(A, "A")

    # backward 
    if A[dim-1][dim-1] != 0:
        for i in range(dim-1, -1, -1):
            for j in range(i):
                r = -A[j][i]
                # print(r)
                A = add_row(A, j, i, r)
                A_inv = add_row(A_inv, j, i, r)
                ops.append(("A", j, i,r))
                # print(A, "A")
        ops.append("SOLUTION")
    else :
        ops.append("DEGENERATE")
    return ops, A_inv

    

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
