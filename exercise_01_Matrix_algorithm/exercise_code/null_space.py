import numpy as np

def get_null_vector(D):
    '''
    Inputs:
    - D: numpy.ndarray, matrix of shape (m,n)
    Outputs:
    - null_vector: numpy.ndarray, matrix of shape (dim_kern,n)
    '''
    
    ########################################################################
    # TODO:                                                                #
    # Get the kernel of the matrix D.                                      #
    # the kernel should consider the numerical errors.                     #
    ########################################################################

    # in principal, one can do row operation on A^T | I and we get the x^T such that it's in nullspace
    D = D.astype(np.float64)

    m = D.shape[0]
    n = D.shape[1]

    Dt = D.T
    I = np.eye(n)

    # do guassian elimination
    k = min(m,n)
    for i in range(k):

        # find the largest leading entry
        Mi = i
        for j in range(i, n):
            if abs(Dt[j][i]) >= abs(Dt[Mi][i]):
                Mi = j
        if Dt[Mi][i] == 0: continue

        # swap the rows
        
        Dt[[Mi, i]] = Dt[[i, Mi]]
        I[[Mi, i]] = I[[i, Mi]]

        # normalize to 1
        I[i] = I[i]/Dt[i][i]
        Dt[i] = Dt[i]/Dt[i][i]

        # row ops
        for j in range(i+1, n):
            I[j] = I[j] - I[i]*Dt[j][i]
            Dt[j] = Dt[j] - Dt[i]*Dt[j][i]
        
    # backward
    for i in range(k-1, -1, -1):
        for j in range(i):
            I[j] = I[j] - I[i]*Dt[j][i]
            Dt[j] = Dt[j] - Dt[i]*Dt[j][i]
    

    # collect the zero rows
    nulls = []
    for i in range(n): 
        all0 = True
        for j in range(m):
            if Dt[i][j] != 0:
                all0 = False
                break
        if all0 == True:
            nulls.append(np.array(I[i])/np.linalg.norm(I[i]))

    null_vector = np.array(nulls)

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return null_vector 


# D = np.array([[1,2,3,4], [5,3,2,1]])
# null_vector = get_null_vector(D)
# print(null_vector)
# print(np.matmul(D, null_vector.T))


