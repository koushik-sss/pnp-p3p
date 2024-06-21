import numpy as np

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordinates of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camera intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels
    """

    ##### STUDENT CODE START #####
    # Convert pixel coordinates to homogeneous coordinates
    pixels_homogeneous = np.hstack((pixels, np.ones((pixels.shape[0], 1))))

    # Calculate the inverse of the rotation matrix
    Rw_inv = np.linalg.inv(R_wc)

    # Calculate the camera translation in the world frame
    tw_inv = -Rw_inv @ t_wc

    # Create the [R | t] matrix
    Rt = np.hstack((Rw_inv[:, :2], tw_inv.reshape(3, 1)))

    # Calculate the world coordinates
    Pw = np.linalg.inv(K @ Rt) @ pixels_homogeneous.T

    # Normalize the world coordinates
    Pw = Pw / Pw[2]

    # Set the z-coordinate to zero (assuming points lie on the x-y plane)
    Pw[2] = 0

    # Transpose the result to get an N x 3 matrix
    Pw = Pw.T
    ##### STUDENT CODE END #####

    return Pw