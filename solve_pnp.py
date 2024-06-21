from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)
        R@Pc + t = Pw
    """

    ##### STUDENT CODE START #####
    H = est_homography(Pw[:, :2], Pc)
    
    H = H / H[2, 2]
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    
    K_inv = np.linalg.inv(K)
    lambda_ = 1 / np.linalg.norm(K_inv @ h1)
    
    r1 = lambda_ * K_inv @ h1
    r2 = lambda_ * K_inv @ h2
    r3 = np.cross(r1, r2)
    t = lambda_ * K_inv @ h3
    
    R = np.stack([r1, r2, r3], axis=1)
    
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    ##### STUDENT CODE END ##### 

    return R.T, -R.T@t