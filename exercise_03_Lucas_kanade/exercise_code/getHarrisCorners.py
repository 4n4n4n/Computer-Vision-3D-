import numpy as np

def getHarrisCorners(M, kappa, theta):
    # Compute Harris corners
    # Input:
    # M: structure tensor of shape (H, W, 2, 2)
    # kappa: float (parameter for Harris corner score) 
    # theta: float (threshold for corner detection)
    # Output:
    # score: numpy.ndarray (Harris corner score) of shape (H, W)
    # points: numpy.ndarray (detected corners) of shape (N, 2)

    ########################################################################
    # TODO:                                                                #
    # Compute the Harris corner score and find the corners.               #
    #                                                                      #
    # Hints:                                                               #
    # - The Harris corner score is computed using the determinant and      #
    #   trace of the structure tensor.                                     #
    # - Use the threshold theta to find the corners.                       #
    # - Use non-maximum suppression to find the corners.                   #
    ########################################################################


    score = np.linalg.det(M) - kappa* (np.trace(M, axis1 = 2, axis2 = 3)**2) # shape (H, W)
    padded = np.pad(score, 1, 'constant', constant_values= -np.inf) # pad the score matrix
    # construct 4 matrices to represent the 4 neighbors and the centers
    center = padded[1:-1, 1:-1] # the original entries (excluding the first and the last rows and column)
    up = padded[0:-2, 1:-1] 
    down = padded[2:, 1:-1]
    left = padded[1:-1, 0:-2]
    right = padded[1:-1, 2:] 
    is_max = (center > theta) & (center > up) & (center > down) & (center > left) & (center > right)

    points = np.argwhere(is_max)


    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return score, points

