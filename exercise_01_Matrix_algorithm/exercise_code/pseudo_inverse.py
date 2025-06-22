import numpy as np

v = [[-1.19528616,  1.,         -0.87154047,  0.32968552]]
delta_x = [ 0.65821884 ,-0.55073623,  0.48007027, -0.18157669]


def solve_linear_equation_SVD(D, b):
    '''
    Inputs:
    - D: numpy.ndarray, matrix of shape (m,n)
    - b: numpy.ndarray, vector of shape (m,)
    Outputs:
    - x: numpy.ndarray, solution of the linear equation D*x = b
    - D_inv: numpy.ndarray, pseudo-inverse of D of shape (n,m)
    '''

    ########################################################################
    # TODO:                                                                #
    # Solve the linear equation D*x = b using the pseudo-inverse and SVD.  #
    # Your code should be able to tackle the case where D is singular.     # 
    ########################################################################
    D = D.astype(np.float64)
    # U, S, Vh= np.linalg.svd(D, True, True)
    
    # S_inv = np.zeros_like(D.T)
    # for i, s in enumerate(S):
    #     S_inv[i][i] = 1/s 
    
    # D_inv = np.matmul(np.matmul(Vh.T, S_inv), U.T)
    D_inv = np.linalg.pinv(D)
    x = np.matmul(D_inv, b)

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return x, D_inv


def generate_matrix(x_star, b, eps=1e-4):
    '''
    Inputs:
    - x_star: numpy.ndarray, vector of shape (n,)
    - b: numpy.ndarray, vector of shape (m,)
    - eps: float, noise level
    Outputs:
    - D: numpy.ndarray, matrix of shape (m,n)
    
    '''
    m = len(b)
    n = len(x_star)
    D = np.random.randn(m, n)

    ########################################################################
    # TODO:                                                                #
    # Generate a matrix D such that D @ x_star = b.                        #
    #                                                                      #
    # Construct D[:,-1] such that D @ x_star = b.                          #
    ########################################################################


    for i in range(m):
        sum = 0
        for j in range(n-1):
            sum += D[i][j]*x_star[j]
        D[i][n-1] = (b[i]-sum)/x_star[n-1]

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    # add some noise
    D = D + eps * np.random.randn(m, n)

    
    return D # shape (m,n)



# D = generate_matrix(x_star, b, 0.)
# print(D)
# x, _ = solve_linear_equation_SVD(D, b)
# print(x)