import numpy as np
from helpers import *


def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic
    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)
    """
    world_points = [Pw[i] for i in range(4)]
    normalized_pixels = [np.dot(np.linalg.inv(K), np.append(Pc[i], 1).T) for i in range(4)]

    distances = list(map(lambda i: np.linalg.norm(world_points[i[0]] - world_points[i[1]]), [(2, 3), (1, 3), (1, 2)]))

    cosines = list(map(lambda i: np.dot(normalized_pixels[i[0]], normalized_pixels[i[1]]) / (np.linalg.norm(normalized_pixels[i[0]]) * np.linalg.norm(normalized_pixels[i[1]])), [(2, 3), (1, 3), (1, 2)]))

    coefficients = compute_coefficients(distances, cosines)
    real_roots = find_real_roots(coefficients)

    min_error = float('inf')
    best_R, best_t = None, None

    best_R, best_t = None, None
    min_error = float('inf')

    for v in real_roots:
        u = compute_u(v, distances, cosines)
        s12 = compute_s1_square(u, v, distances, cosines)
        s = compute_scales(s12, u, v)
        Pc_3d = compute_3d_points(s, normalized_pixels)

        R_wc, t_wc = Procrustes(Pc_3d, Pw[1:4])

        R_cw, t_cw = R_wc.T, -np.dot(R_wc.T, t_wc.reshape((-1, 1)))
        p0_world_homogeneous = np.append(world_points[0], 1)
        p0_camera_homogeneous = np.dot(np.hstack((R_cw, t_cw)), p0_world_homogeneous)
        p0_pr = np.dot(K, p0_camera_homogeneous)
        p0_p = p0_pr[:2] / p0_pr[2]

        error = np.linalg.norm(Pc[0] - p0_p)

        if error < min_error:
            min_error = error
            best_R, best_t = R_wc, t_wc

    return best_R, best_t


def Procrustes(X, Y):
    """
    Solve the Procrustes problem: Y = RX + t
    Input:
    - X: Nx3 numpy array representing N points in the camera coordinate system (obtained from P3P)
    - Y: Nx3 numpy array representing N points in the world coordinate system
    Returns:
    - R: 3x3 numpy array representing the camera orientation in the world coordinate system (R_wc)
    - t: (3,) numpy array representing the camera translation in the world coordinate system (t_wc)
    """
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean
    covariance_matrix = np.dot(X_centered.T, Y_centered)
    U, _, Vt = np.linalg.svd(covariance_matrix)
    rotation_matrix = np.dot(Vt.T, np.dot(np.diag([1, 1, np.linalg.det(np.dot(Vt.T, U.T))]), U.T))
    translation_vector = Y_mean - np.dot(rotation_matrix, X_mean)
    return rotation_matrix, translation_vector.flatten()
