import numpy as np
import cv2

#Import necessary functions
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac, compositeH

#Write script for Q4.2x

def load_images():
    #Load left and right images.
    left_img = cv2.imread('../data/study_space_left.jpg')
    right_img = cv2.imread('../data/study_space_right.jpg')
    print("Images loaded successfully.")
    return left_img, right_img


def adjust_image_dimension(left_img, right_img):
    #Rectifying dimension mismatch by adding zero-padding
    left_height, left_width, _ = left_img.shape
    right_height, right_width, _ = right_img.shape
    
    extra_width = round(max(left_width, right_width) * 0.2)
    top_padding = 0
    bottom_padding = left_height - right_height

    adjusted_img = cv2.copyMakeBorder(right_img, top_padding, bottom_padding, extra_width, 0, cv2.BORDER_CONSTANT, 0)
    print("Padding adjusted.")
    
    return adjusted_img


def compute_panorama(left_img, adjusted_right_img, options):
    # Computing panorama using homography
    matches, locs1, locs2 = matchPics(left_img, adjusted_right_img, options)
    locs1_matched = locs1[matches[:, 0], 0:2]
    locs2_matched = locs2[matches[:, 1], 0:2]
    print("Calculating homography.")
    bestH2to1, _ = computeH_ransac(locs1_matched, locs2_matched, options)
    panorama = compositeH(bestH2to1, left_img, adjusted_right_img)
    
    # Ensure all parts of the images are retained in the panorama
    panorama = np.maximum(adjusted_right_img, panorama)
    print("Panorama successfully created!")
    return panorama


if __name__ == '__main__':
    # Get options for matching and RANSAC
    options = get_opts()

    # Load images and adjust dimensions
    left, right = load_images()
    adjusted_right = adjust_image_dimension(left, right)
    
    # Compute the panorama
    panorama_img = compute_panorama(left, adjusted_right, options)

    # Save the resultant panorama image
    cv2.imwrite('../result/test.png', panorama_img)