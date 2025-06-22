import numpy as np
from math import tan, atan
from abc import ABC, abstractmethod

def compute_relative_pose(pose_1,pose_2):
    '''
    Inputs:
    - pose_i transform from cam_i to world coordinates, matrix of shape (3,4)
    Outputs:
    - pose transform from cam_1 to cam_2 coordinates, matrix of shape (3,4)
    '''

    ########################################################################
    # TODO:                                                                #
    # Compute the relative pose, which transform from cam_1 to cam_2       #
    # coordinates.                                                         #
    ########################################################################

    R1 = pose_1[:,  0:3]
    T1 = pose_1[:, 3]
    R2 = pose_2[:, 0:3]
    T2 = pose_2[:, 3]
    R = np.matmul(R2.T, R1)
    T = np.matmul(R2.T, T1-T2)
    
    pose = np.zeros((3,4))
    pose[:, 0:3] = R
    pose[:, 3] = T

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return pose



class Camera(ABC):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    @abstractmethod
    def project(self, pt):
        """Project the point pt onto a pixel on the screen"""
        
    @abstractmethod
    def unproject(self, pix, d):
        """Unproject the pixel pix into the 3D camera space for the given distance d"""


class Pinhole(Camera):

    def __init__(self, w, h, fx, fy, cx, cy):
        Camera.__init__(self, w, h)
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def project(self, pt):
        '''
        Inputs:
        - pt, vector of size 3
        Outputs:
        - pix, projection of pt using the pinhole model, vector of size 2
        '''
        ########################################################################
        # TODO:                                                                #
        # project the point pt, considering the pinhole model.                 #
        ########################################################################

        pix = np.matmul(self.K, pt)
        pix = pix[0:2] / pt[2]


        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return pix

    def unproject(self, pix, d):
        '''
        Inputs:
        - pix, vector of size 2
        - d, scalar 
        Outputs:
        - final_pt, obtained by unprojecting pix with the distance d using the pinhole model, vector of size 3
        '''
        ########################################################################
        # TODO:                                                                #
        # Unproject the pixel pix using the distance d, considering the pinhole#
        # model. Be careful: d is the distance between the camera origin and   #
        # the desired point, not the depth.                                    #
        ########################################################################

        ray = np.array([(pix[0]-self.K[0,2])/self.K[0,0], (pix[1]-self.K[1,2])/self.K[1,1], 1.0])
        r = np.linalg.norm(ray)
        final_pt = ray*d/r
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return final_pt


class Fov(Camera):

    def __init__(self, w, h, fx, fy, cx, cy, W):
        Camera.__init__(self, w, h)
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.W = W

    def project(self, pt):
        '''
        Inputs:
        - pt, vector of size 3
        Outputs:
        - pix, projection of pt using the Fov model, vector of size 2
        '''
        ########################################################################
        # TODO:                                                                #
        # project the point pt, considering the Fov model.                     #
        ########################################################################

        pix = pt
        pix = pix/pix[2]
        r = np.linalg.norm(pix[0:2]) # note that only the first two entries 
        g = 1/(self.W*r)*atan(2*r*tan(self.W/2))
        pix[0] *= g
        pix[1] *= g
        pix = np.matmul(self.K, pix)
        pix = pix[0:2]

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return pix

    def unproject(self, pix, d):
        '''
        Inputs:
        - pix, vector of size 2
        - d, scalar 
        Outputs:
        - final_pt, obtained by unprojecting pix with the distance d using the Fov model, vector of size 3
        '''
        ########################################################################
        # TODO:                                                                #
        # Unproject the pixel pix using the distance d, considering the FOV    #
        # model. Be careful: d is the distance between the camera origin and   #
        # the desired point, not the depth.                                    #
        ########################################################################

        pt = np.matmul(np.linalg.inv(self.K), np.append(pix, 1.0))
        rd = np.linalg.norm(pt[0:2])
        f = tan(self.W*rd)/(2*rd*tan(self.W/2))
        ray = pt
        ray[0] *= f
        ray[1] *= f
        final_pt = ray*d/np.linalg.norm(ray)

        

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################        
        return final_pt
