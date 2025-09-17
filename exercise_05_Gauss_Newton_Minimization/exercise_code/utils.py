import numpy as np
from PIL import Image
from scipy.linalg import expm, logm

def se3Exp(twist):
    # twist: 6x1 vector
    ########################################################################
    # TODO: Implement the function to compute the mat exp of the twist     #
    # construct the corresponding mat in se3                               #
    # For details see slide 19/20 in Chapter 2 of the lecture slides       #
    ########################################################################


    v = twist[0:3]
    w = twist[3:]
    
    w_hat = np.array([
        [0,     -w[2],  w[1]],
        [w[2],   0,    -w[0]],
        [-w[1],  w[0],  0]
    ])
    M = np.zeros((4,4))
    M[0:3, 0:3] = w_hat
    M[0:3, 3] = v


    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return expm(M)

def se3Log(T):
    # T: 4x4 matrix
    lg = logm(T)
    ########################################################################
    # TODO: Implement the function to compute the mat log of the SE3 mat   #
    # extract the twist vector from the corresponding se3 mat              #
    ########################################################################

    twist = np.zeros(6)
    twist[0:3] = lg[0:3, 3]

    twist[3] = lg[2,1]
    twist[4] = lg[0,2]
    twist[5] = lg[1,0]    

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return twist


def load_dataset():
    K = np.array([[517.3, 0, 318.6],
                  [0, 516.5, 255.3],
                  [0, 0, 1]]) 
    img1 = np.array(Image.open('./rgb/1305031102.275326.png').convert('L')) / 255.
    img2 = np.array(Image.open('./rgb/1305031102.175304.png').convert('L')) / 255.
    depth1 = np.array(Image.open('./depth/1305031102.262886.png')) / 5000.
    depth2 = np.array(Image.open('./depth/1305031102.160407.png')) / 5000.

    return img1, img2, depth1, depth2, K

def down_scale(img, depth, K, lvl):
    # Recursively downscale image, depth, and intrinsics by factor 2 per level
    # Inputs:
    #   img, depth : H×W grayscale image and depth map (2D numpy arrays)
    #   K          : 3×3 intrinsics matrix
    #   lvl        : number of downscaling steps (int)
    # Outputs:
    #   img_down, depth_down : downscaled image and depth map
    #   K_down               : adjusted intrinsics

    # if we are at the lowest level, we don't need to downscale
    if lvl == 1:
        return img, depth, K
    ########################################################################
    # TODO: Implement the downscale of the image, depth and camera         #
    # for a detailed description see exercise sheet 8                      #
    # average only over valid (non-zero) depth pixels                      #
    ########################################################################

    H, W =  img.shape[0], img.shape[1]
    
    img_downscale = np.zeros((H//2, W//2))
    depth_downscale = np.zeros((H//2, W//2))
    for i in range(H//2):
        for j in range(W//2):
            cnt = 0
            arr = [[2*i, 2*j], [2*i+1, 2*j], [2*i, 2*j+1], [2*i+1, 2*j+1]]
            for idx in arr:
                if(depth[idx[0], idx[1]] != 0):
                    cnt += 1
                    depth_downscale[i,j] += depth[idx[0], idx[1]]
                img_downscale[i,j] += img[idx[0], idx[1]]
            img_downscale[i,j] /= 4
            if cnt > 0:
                depth_downscale[i,j] /= cnt
    K_downscale = np.zeros_like(K)
    K_downscale[0,0] = 0.5*K[0,0]
    K_downscale[1,1] = 0.5*K[1,1]
    K_downscale[0,2] = 0.5*K[0,2]-0.25
    K_downscale[1,2] = 0.5*K[1,2]-0.25
    K_downscale[2,2] = 1.0
    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    
    # recursively downscale
    return down_scale(img_downscale, depth_downscale, K_downscale, lvl - 1)
    
