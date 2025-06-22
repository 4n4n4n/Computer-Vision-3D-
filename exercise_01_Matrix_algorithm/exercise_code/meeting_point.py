
import numpy as np
from scipy.linalg import null_space


def swap_rows(A, i, j):
    A[[i,j]] = A[[j,i]]
    return A
def multiply_row(A, i, scalar):
    A[i] = A[i]*scalar
    return A
def add_row(A, i, j, scalar = 1):
    A[i] = A[i] + A[j] * scalar
    return A

def rref(A):
    A = A.astype(float)
    r = A.shape[0]
    c = A.shape[1]
    pivots = []
    n = min(r,c)
    for i in range(n):
        mr = i
        for j in range(i, n):
            if abs(A[mr][i]) <= abs(A[j][i]):
                mr = j
        if A[mr][i] == 0: continue
        A = swap_rows(A, mr, i)
        A = multiply_row(A, i, 1/A[i][i])
        for j in range(i+1, n):
            A =  add_row(A, j, i, -A[j][i])
        
    
    # collecting pivots
    for i in range(n):
        if A[i][i] != 0:
            pivots.append(i)
    return pivots
    

# from null_space import get_null_vector

def normalize_col(A):
    n = A.shape[1]
    for i in range(n):
        A[:, i] = A[:, i]/np.linalg.norm(A[:, i])
    return A

def meeting_point_linear(pts_list):
    '''
    Inputs:
    - pts_list: List[numpy.ndarray], list of each persons points in the space
    Outputs:
    - numpy.ndarray, meeting point or vectors spanning the possible meeting points of shape (m, dim_intersection)
    '''
    A = pts_list[0] # person A's points of shape (m,num_pts_A)
    B = pts_list[1] # person B's points of shape (m,num_pts_B)
    m = A.shape[0]
    ########################################################################
    # TODO:                                                                #
    # Implement the meeting point algorithm.                               #
    #                                                                      #
    # As an input, you receive                                             #
    # - for each person, you receive a list of landmarks in their subspace.#
    #   It is guaranteed that the landmarks span each person’s whole       #
    #   subspace.                                                          #
    #                                                                      #
    # As an output,                                                        #
    # - If such a point exist, output it.                                  #
    # - If there is more than one such point,                              # 
    #   output vectors spanning the space.                                 #
    ########################################################################

    Na = null_space(A.T)
    # print(Na)
    Nb = null_space(B.T)
    # print(Nb)
    N = np.hstack([Na,Nb])
    # print(N)
    basis = null_space(N.T)
    
    
    return np.array([[0],[0],[0]]) if len(basis[0])==0 else basis
    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

c = np.array([0, 0, 0])
a = np.array([0, 1, 0])
b = np.array([0, 0, 1])
PTS_a = [2*a+2*c, 3*a+c, 4*a+9*c, 5*a+c]
PTS_b = [2*b+c, 3*b+4*c, 4*b+c, 5*b+2*c]
PTS_a, PTS_b = np.array(PTS_a).T, np.array(PTS_b).T
basis = meeting_point_linear([PTS_a, PTS_b])
print(basis)


