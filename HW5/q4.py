import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
import matplotlib.pyplot as plt

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    ##### your code here #####
    ##########################
    
     Denoise the image using a Gaussian filter
    denoised_image = skimage.filters.gaussian(image, sigma=2, multichannel=True)

    # Convert the image to grayscale
    gray_image = skimage.color.rgb2gray(denoised_image)

    # Apply Otsu's method to find an optimal threshold value
    threshold = skimage.filters.threshold_otsu(gray_image)

    # Perform morphological closing to close small holes in the foreground
    closed_image = skimage.morphology.closing(
        gray_image < threshold, skimage.morphology.square(10))

    # Remove artifacts connected to image border
    cleared_image = skimage.segmentation.clear_border(closed_image)

    # Label the image regions and extract region properties
    labeled_image, num = skimage.measure.label(
        cleared_image, background=0, return_num=True, connectivity=2)

    # Filter out small regions based on area
    for region in skimage.measure.regionprops(labeled_image):
        if region.area >= 200:
            bboxes.append(region.bbox)

    # Create a black and white image for output
    bw = 1.0 - closed_image

    return bboxes, bw
