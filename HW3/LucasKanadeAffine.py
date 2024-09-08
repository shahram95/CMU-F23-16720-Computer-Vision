import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    
    params = np.zeros(6)
    height_template, width_template = It.shape
    height_current, width_current = It1.shape

    template_spline = RectBivariateSpline(
        np.arange(height_template), 
        np.arange(width_template), 
        It)
    
    current_spline = RectBivariateSpline(
        np.arange(height_current), 
        np.arange(width_current), 
        It1)
    
    iteration = 1
    change_magnitude = 1
    
    x, y = np.mgrid[0:width_template, 0:height_template]
    x_flat = x.ravel()
    y_flat = y.ravel()
    coords = np.vstack((x_flat, y_flat, np.ones_like(x_flat)))

    while change_magnitude > threshold and iteration < num_iters:
        affine_matrix = np.array([
            [1 + params[0], params[1], params[2]], 
            [params[3], 1 + params[4], params[5]]])
        
        warped_coords = np.dot(affine_matrix, coords)
        warped_x = warped_coords[0]
        warped_y = warped_coords[1]
        
        x_mask = np.logical_or(warped_x >= width_template, warped_x < 0)
        y_mask = np.logical_or(warped_y >= height_template, warped_y < 0)

        mask_combined = np.logical_or(x_mask, y_mask)
        
        valid_x = np.delete(x_flat, mask_combined)
        valid_y = np.delete(y_flat, mask_combined)
        warped_x = np.delete(warped_x, mask_combined)
        warped_y = np.delete(warped_y, mask_combined)
        
        grad_x = current_spline.ev(warped_y, warped_x, dy=1)
        grad_y = current_spline.ev(warped_y, warped_x, dx=1)
        current_values = current_spline.ev(warped_y, warped_x)
        template_values = template_spline.ev(valid_y, valid_x)

        A_matrix = np.vstack((
            valid_x * grad_x, valid_y * grad_x, grad_x,
            valid_x * grad_y, valid_y * grad_y, grad_y)).T

        b_vector = (template_values - current_values).reshape(-1, 1)

        delta_params = np.linalg.lstsq(A_matrix, b_vector, rcond=None)[0]
        change_magnitude = np.linalg.norm(delta_params)
        params += delta_params.ravel()

        iteration += 1

    M = np.array([
        [1 + params[0], params[1], params[2]], 
        [params[3], 1 + params[4], params[5]]])        
    return M