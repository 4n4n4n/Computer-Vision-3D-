import numpy as np

from .utils import skewMat


def transformImgCoord(x1, x2, y1, y2, K1, K2):
    # transform the image coordinates
    # assume the image plane is at z = 1
    # what should 3D points be in camera coordinates?
    # input: 2D points in two images (x1, x2, y1, y2), intrinsics K1, K2
    # output: normalized camera coords x1, x2, y1, y2 (each of shape (n_pts,))

    ########################################################################
    # TODO: Implement the transformation with                              #
    # the given camera intrinsic matrices                                  #
    ########################################################################


    pts1 = np.matmul(np.linalg.inv(K1), np.stack([x1, y1, np.ones_like(x1)]))
    pts2 = np.matmul(np.linalg.inv(K2), np.stack([x2, y2, np.ones_like(x2)]))

    x1 = pts1[0]
    y1 = pts1[1]
    x2 = pts2[0]
    y2 = pts2[1]

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return x1, x2, y1, y2


def constructChiMatrix(x1, x2, y1, y2):
    # construct the chi matrix using the kronecker product
    # input: normalized camera coords x1, y1 in image1 and x2, y2 in image2 
    # output: chi matrix of shape (n_pts, 9)
    n_pts = x1.shape[0]
    chi_mat = np.zeros((n_pts, 9))
    for i in range(n_pts):
        ########################################################################
        # TODO: construct the chi matrix by kronecker product                  #
        ########################################################################
        

        chi_mat[i] = np.array([x1[i]*x2[i], x1[i]*y2[i], x1[i], y1[i]*x2[i], y1[i]*y2[i], y1[i], x2[i], y2[i], 1])

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
    return chi_mat


def solveForEssentialMatrix(chi_mat):
    # project the essential matrix onto the essential space
    # input: chi matrix - shape (n_pts, 9)
    # output: essential matrix E - shape (3, 3), U, Vt - shape (3, 3),  S - shape (3, 3) diagonal matrix with E = U @ S @ Vt

    ########################################################################
    # TODO: solve the minimization problem to get the solution of E here.  #
    ########################################################################

    # Perform SVD on chi_mat
    _, S, vh = np.linalg.svd(chi_mat)
    
    # Find the vector with the smallest singular value (last row of vh)
    E = vh[-1]  

    # Reshape E to 3x3 by filling columns first
    E = np.reshape(E, (3,3), 'F')

    # Perform SVD on E
    U, S, Vt = np.linalg.svd(E)

    # Ensure the determinants of U and Vt are positive
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt

    # Set the singular values to enforce the essential matrix constraint
    # Essential matrix should have two equal non-zero singular values and one zero singular value
    S = np.array([1.0, 1.0, 0.0])

    # Reconstruct E with the corrected singular values
    E = U @ np.diag(S) @ Vt




    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return E, U, Vt, np.diag(S)


def constructEssentialMatrix(x1, x2, y1, y2, K1, K2):
    # compute an approximate essential matrix
    # input: 2D points in two images (x1, x2, y1, y2), camera intrinsic matrix K1, K2
    # output: essential matrix E - shape (3, 3),
    #         singular vectors of E: U, Vt - shape (3, 3),
    #         singular values of E: S - shape (3, 3) diagonal matrix, with E = U @ S @ Vt.

    # you need to finish the following three functions
    x1, x2, y1, y2 = transformImgCoord(x1, x2, y1, y2, K1, K2)
    chi_mat = constructChiMatrix(x1, x2, y1, y2)
    E, U, Vt, S = solveForEssentialMatrix(chi_mat)
    return E, U, Vt, S


def recoverPose(U, Vt, S):
    # recover the possible poses from the essential matrix
    # input: singular vectors of E: U, Vt - shape (3, 3),
    #        singular values of E: S - shape (3, 3) diagonal matrix, with E = U @ S @ Vt.
    # output: possible rotation matrices R1, R2 - each of shape (3, 3),
    #         possible translation vectors T1, T2 - each of shape (3,)

    ########################################################################
    # TODO: 1. implement the R_z rotation matrix.                          #
    #          There should be two of them.                                #
    #       2. recover the rotation matrix R                               #
    #          with R_z, U, Vt. (two of them).                             #
    #       3. recover \hat{T} with R_z, U, S                              #
    #          and extract T. (two of them).                               #
    #       4. return R1, R2, T1, T2.                                      #
    ########################################################################


    Rz1 = np.array([[0,1,0], [-1,0,0], [0,0,1]])
    Rz2 = np.array([[0,-1,0], [1,0,0], [0,0,1]])
    R1 = U @ Rz1 @ Vt
    R2 = U @ Rz2 @ Vt
    # be careful of the rotation direction
    T1_hat = U @ Rz2 @ S @ U.T
    T2_hat = U @ Rz1 @ S @ U.T
    # Extract translation vectors from the skew-symmetric matrices
    T1 = np.array([T1_hat[2,1], T1_hat[0,2], T1_hat[1,0]])
    T2 = np.array([T2_hat[2,1], T2_hat[0,2], T2_hat[1,0]])

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return R1, R2, T1, T2


def reconstruct(x1, x2, y1, y2, R, T):
    # reconstruct the 3D points from the 2D correspondences and (R, T)
    # input:  normalized camera coords in two images (x1, x2, y1, y2), rotation matrix R - shape (3, 3), translation vector T - shape (3,)
    # output: 3D points X1, X2

    n_pts = x1.shape[0]
    X1, X2 = None, None

    ########################################################################
    # TODO: implement the structure reconstruction matrix M.               #
    #  1. construct the matrix M -shape (3 * n_pts, n_pts + 1)             #
    #    which is defined as page18, chapter 5.                            #
    #  2. find the lambda and gamma as explained on the same page.         #
    #     make sure that gamma is positive                                 #
    #  3. generate the 3D points X1, X2 with lambda and (R, T).            #
    #  4. check the number of points with positive depth,                  #
    #     it should be n_pts                                               #
    ########################################################################
    
    # 1. Construct M
    pts1 = np.stack([x1, y1, np.ones_like(x1)], axis = 1) # (n,3)
    pts2 = np.stack([x2, y2, np.ones_like(x1)], axis = 1)
    M = np.zeros((3*n_pts, n_pts+1)) 
    for i in range(n_pts):
        skew_p2 = skewMat(pts2[i])
        M[3*i: 3*(i+1), i] = (skew_p2 @ R @ pts1[i])
        M[3*i: 3*(i+1), n_pts] = (skew_p2 @ T)
    # 2. find the lmabda and gamma
    _, _, Vh = np.linalg.svd(M)
    
    solution = Vh[-1, :]
    lmbda = solution[:-1] #(n, )
    gamma = solution [-1] # constant
    # print("Vh\n", Vh)

    # normalize gamma to 1
    lmbda /= gamma
    gamma /= gamma


    # 3. Get back X1 and X2
    X1 = pts1.T * lmbda # (3, n)
    X2 = R @ X1 + (gamma * T)[:, np.newaxis]

    # 4. Count how many points have positive depth (z-coordinate > 0)
    n_positive_depth1 = np.sum(X1[2, :] > 0)
    n_positive_depth2 = np.sum(X2[2, :] > 0)

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################


    
    if n_positive_depth1 == n_pts and n_positive_depth2 == n_pts:
        return X1, X2
    else:
        return None, None


def allReconstruction(x1, x2, y1, y2, R1, R2, T1, T2, K1, K2):
    # reconstruct the 3D points from the 2D correspondences and the possible poses
    # input: 2D points in two images (x1, x2, y1, y2), possible rotation matrices R1, R2 - each of shape (3, 3),
    #        possible translation vectors T1, T2 - each of shape (3,), intrinsics K1, K2
    # output: the correct rotation matrix R, translation vector T, 3D points X1, X2

    num_sol = 0
    #transform to camera coordinates
    x1, x2, y1, y2 = transformImgCoord(x1, x2, y1, y2, K1, K2)
    # first check (R1, T1)
    X1, X2 = reconstruct(x1, x2, y1, y2, R1, T1)
    if X1 is not None:
        num_sol += 1
        R = R1
        T = T1
        X1_res = X1
        X2_res = X2

    # check (R1, T2)
    X1, X2 = reconstruct(x1, x2, y1, y2, R1, T2)
    if X1 is not None:
        num_sol += 1
        R = R1
        T = T2
        X1_res = X1
        X2_res = X2

    # check (R2, T1)
    X1, X2 = reconstruct(x1, x2, y1, y2, R2, T1)
    if X1 is not None:
        num_sol += 1
        R = R2
        T = T1
        X1_res = X1
        X2_res = X2

    # check (R2, T2)
    X1, X2 = reconstruct(x1, x2, y1, y2, R2, T2)
    if X1 is not None:
        num_sol += 1
        R = R2
        T = T2
        X1_res = X1
        X2_res = X2

    if num_sol == 0:
        print('No valid solution found')
        return None, None, None, None
    elif num_sol == 1:
        print('Unique solution found')
        return R, T, X1_res, X2_res
    else:
        print('Multiple solutions found')
        return R, T, X1_res, X2_res
