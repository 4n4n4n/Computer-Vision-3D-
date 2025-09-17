import numpy as np
from .utils import se3Exp, se3Log
from scipy.interpolate import RegularGridInterpolator

def derive_jac_and_residual(img, depth, img_tag, xi, K):
    # compute the jacobian and residual corresponding to the given xi.

    ########################################################################
    # TODO: get the rotation and translation from the xi                   #
    ########################################################################


    xi_exp = se3Exp(xi)
    R = xi_exp[0:3, 0:3]
    T = xi_exp[0:3, 3]

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    # the arrays to save the intermediate results
    # these two contain the x, y, image coordinates of the reference-pixel, 
    # transformed & projected into the new image
    Imgx = np.zeros(img.shape) - 10
    Imgy = np.zeros(img.shape) - 10

    # these arrays save the 3d position of the transformed point
    xp = np.zeros(img.shape) + np.float64('nan')
    yp = np.zeros(img.shape) + np.float64('nan')
    zp = np.zeros(img.shape) + np.float64('nan')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
    ########################################################################
    # TODO: 1) get the point in 3D space using the depth map               #
    # 2) rotate the point and translate it to target camera                # 
    # 3) if it is valid (depth >0 in both), project and save results       #
    ########################################################################
            # get back the 3D point X
            X = np.linalg.pinv(K) @  (np.array([j,i,1]) * depth[i,j])
            # transformed into another coordinates
            X = R @ X + T
            # store the coordinates
            xp[i,j] = X[0]
            yp[i,j] = X[1]
            zp[i,j] = X[2]
            # projected if the depth > 0
            if(zp[i,j] > 0):
                X_proj = K @ X 
                Imgx[i,j] = X_proj[0] / X_proj[2]
                Imgy[i,j] = X_proj[1] /X_proj[2]
    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    # compute the image derivative
    dxI = np.zeros(img.shape) + np.float64('nan')
    dyI = np.zeros(img.shape) + np.float64('nan')
    dxI[:, 1:-1] = (img_tag[:, 2:] - img_tag[:, :-2]) / 2
    dyI[1:-1, :] = (img_tag[2:, :] - img_tag[:-2, :]) / 2
    interpx = RegularGridInterpolator((np.arange(img.shape[0]), np.arange(img.shape[1])), dxI, bounds_error=False)
    interpy = RegularGridInterpolator((np.arange(img.shape[0]), np.arange(img.shape[1])), dyI, bounds_error=False)
    # this allows us to query (x,y) for x, y non-integer, (Imgx[i,j], Imgy[i,j]) 
    Ixfx = K[0, 0] * interpx(np.stack([Imgy.flatten(), Imgx.flatten()], axis=1))
    Iyfy = K[1, 1] * interpy(np.stack([Imgy.flatten(), Imgx.flatten()], axis=1))

    # get warped 3d points
    xp = xp.flatten()
    yp = yp.flatten()
    zp = zp.flatten()
    
    # implement the gradient
    Jac = np.zeros((img.shape[0] * img.shape[1], 6))
    interpI = RegularGridInterpolator((np.arange(img.shape[0]), np.arange(img.shape[1])), img_tag, bounds_error=False)
    ########################################################################
    # TODO: compute the Jacobian and residual                              #
    # check the theoretical exericse sheet 8 to get the Jacobian formula   #
    ########################################################################

    valid = (zp > 0) 

    # # set invalid entries to nan so they will be zeroed outside this box
    Ixfx[~valid] = np.nan
    Iyfy[~valid] = np.nan
    xp  [~valid] = np.nan
    yp  [~valid] = np.nan
    zp  [~valid] = np.nan

    # 2. pre-compute factors that appear repeatedly
    inv_z = 1.0 / zp                         # 1/z
    x_on_z = xp * inv_z                      # x/z
    y_on_z = yp * inv_z                      # y/z

    # 3. Jacobian columns according to slide formula
    Jac[:, 0] = inv_z * Ixfx                              # d r / d v_x
    Jac[:, 1] = inv_z * Iyfy                              # d r / d v_y
    Jac[:, 2] = inv_z * (-Ixfx * x_on_z  - Iyfy * y_on_z) # d r / d v_z

    Jac[:, 3] = inv_z * (-Ixfx * xp * y_on_z
                         -Iyfy * (zp + yp * y_on_z))     # d r / d ω_x

    Jac[:, 4] = inv_z * ( Ixfx * ( zp + xp * x_on_z)
                         +Iyfy * xp * y_on_z)             # d r / d ω_y

    Jac[:, 5] = inv_z * (-Ixfx * yp + Iyfy * xp)          # d r / d ω_z
  
    # 4. residual: warped intensity minus reference intensity
    I_warped = interpI(np.stack([Imgy.flatten(),
                                 Imgx.flatten()], axis=1))
    residual = I_warped - img.flatten()
    


    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    
    # remove not valid pixels
    notvalid = np.isnan(np.sum(Jac, axis=1))
    residual[notvalid] = 0
    Jac[notvalid, :] = 0

    return Jac, residual


def update_step(jac, residual, xi):
    # compute the update step using the Gauss-Newton method
    # input: jac: the Jacobian J, num_pixel x 6
    #        residual: the residual r, num_pixel x 1
    #        xi: the current xi, 6 x 1
    # output: xi: the updated xi

    ########################################################################
    # TODO: compute the update step using the Gauss-Newton method          #
    # 1. compute the descent setp for \xi                                  #
    # 2. update the \xi using the descent step                             #
    # For details check theoretical exercise sheet 9                       #
    ########################################################################

    JTJ = jac.T @ jac
    JTr = jac.T @ residual
    
  
    # Solve  normal equations
    delta = -np.linalg.solve(JTJ, JTr)
    

    # get back motion
    dg = se3Exp(delta)
    g = se3Exp(xi)
    g_new = dg @ g
    xi = se3Log(g_new)
    # Control step size with a learning rate
    
    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################

    return xi