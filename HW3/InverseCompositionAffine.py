import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    p = np.zeros(6)
    height, width = It.shape

    # Create spline representations of the images
    s_template = RectBivariateSpline(np.arange(height), np.arange(width), It)
    s_current = RectBivariateSpline(np.arange(height), np.arange(width), It1)

    x, y = np.mgrid[0:width, 0:height]
    homogenous_coords = np.vstack((x.ravel(), y.ravel(), np.ones(width * height)))

    dx_template = s_template.ev(y, x, dy=1).ravel()
    dy_template = s_template.ev(y, x, dx=1).ravel()

    # Precompute the matrix A and b for the optimization
    A = np.vstack((x.ravel() * dx_template, y.ravel() * dx_template, dx_template,
                   x.ravel() * dy_template, y.ravel() * dy_template, dy_template)).T

    precomputed_matrix = np.linalg.pinv(A.T @ A) @ A.T

    iteration, change = 0, np.inf
    while change > threshold and iteration < num_iters:
        warp_matrix = np.array([[1 + p[0], p[1], p[2]],
                                [p[3], 1 + p[4], p[5]],
                                [0, 0, 1]])
        
        warped_coords = warp_matrix @ homogenous_coords
        warped_image = s_current.ev(warped_coords[1], warped_coords[0]).ravel()

        error = s_template.ev(y, x).ravel() - warped_image
        delta_p = precomputed_matrix @ error[:, np.newaxis]
        
        change = np.linalg.norm(delta_p)
        p += delta_p.ravel()
        iteration += 1

    M = np.array([[1 + p[0], p[1], p[2]], [p[3], 1 + p[4], p[5]],[0, 0, 1]])
    return M