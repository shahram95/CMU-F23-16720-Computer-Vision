import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
import scipy.ndimage as ndi
from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)

    # Calculate the affine transformation matrix using LucasKanadeAffine method
    M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    #M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    
    # Warp image1 using the derived transformation matrix
    warpim1 = ndi.affine_transform(image1, -M)
    
    # Compute the intensity difference between the warped image1 and image2
    diff = np.abs(warpim1 - image2)
    
    # Generate binary mask based on the tolerance value
    mask = diff > tolerance

    # Refine the mask using morphological operations
    mask = ndi.binary_erosion(mask, iterations=1)
    mask = ndi.binary_dilation(mask, iterations=1)

    return mask


'''
import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
import scipy.ndimage
from scipy.interpolate import RectBivariateSpline


def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """

    # put your implementation here
    mask = np.zeros(image1.shape, dtype=bool)
    M = LucasKanadeAffine(image1, image2, threshold, num_iters)

    imH, imW = image1.shape

    warpim1 = scipy.ndimage.affine_transform(image1, -M, offset=0.0, output_shape=None)
    diff = abs(warpim1 - image2)
    mask[diff > tolerance] = 1
    mask[diff < tolerance] = 0

    mask = scipy.ndimage.morphology.binary_erosion(mask)
    mask = scipy.ndimage.morphology.binary_dilation(mask, iterations=1)

    return mask
'''