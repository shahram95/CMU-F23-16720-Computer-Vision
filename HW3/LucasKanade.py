import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    p = p0

    # Extract image dimensions
    template_height, template_width = It.shape
    current_height, current_width = It1.shape

    # Extract rectangle coordinates and compute its dimensions
    x1, y1, x2, y2 = rect
    width = int(x2 - x1)
    height = int(y2 - y1)

    # Helper function to create a RectBivariateSpline object for each image
    def create_spline(image, height, width):
        return RectBivariateSpline(
            np.linspace(0, height, num=height, endpoint=False),
            np.linspace(0, width, num=width, endpoint=False),
            image
        )
    
    # Create splines for template and current images
    spline_template = create_spline(It, template_height, template_width)
    spline_current = create_spline(It1, current_height, current_width)

    change_magnitude = np.inf
    iteration = 0
    x, y = np.mgrid[x1:x2+1:width*1j, y1:y2+1:height*1j]

    # Iterative optimization to compute vector p
    while (change_magnitude > threshold) and (iteration < num_iters):
        # Warp coordinates using p
        warped_y, warped_x = y + p[1], x + p[0]
        
        # Compute gradients and error between template and current image
        grad_x = spline_current.ev(warped_y, warped_x, dy=1).flatten()
        grad_y = spline_current.ev(warped_y, warped_x, dx=1).flatten()
        error = spline_template.ev(y, x).flatten() - spline_current.ev(warped_y, warped_x).flatten()

        # Create the A matrix
        I = np.eye(2)
        A = np.column_stack((grad_x, grad_y))
        #A_kron = np.kron(np.eye(width*height), I)
        #A_full = A @ A_kron
        
        # Compute the update dp using the pseudo-inverse
        #dp = np.linalg.pinv(A_full).dot(error)
        dp = np.linalg.pinv(A).dot(error)
        change_magnitude = np.linalg.norm(dp)
        
        # Update p
        p = (p + dp.T).ravel()
        
        iteration += 1
    
    return p
