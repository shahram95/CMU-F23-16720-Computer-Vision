import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts


#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH

#Write script for Q2.2.4
opts = get_opts()

# Load the images
cv_cover =cv2.imread('../data/cv_cover.jpg')
cv_desk=cv2.imread('../data/cv_desk.png')
hp_cover=cv2.imread('../data/hp_cover.jpg')

# Resize the Harry Potter cover to match the CV cover dimensions
hp_cover_resize = cv2.resize(hp_cover,(cv_cover.shape[1],cv_cover.shape[0]))

# Get matches and locations using matchPics function
matches,locs1,locs2=matchPics(cv_cover,cv_desk,opts)

# Extract matched feature points
x1=locs1[matches[:,0],0:2]
x2=locs2[matches[:,1],0:2]

# Compute homography using RANSAC
bestH2to1, inliers=computeH_ransac(x1,x2,opts)

# Get composite image using computed homography
composite_img = compositeH(bestH2to1, hp_cover_resize , cv_desk)
cv2.imwrite('../results/hp_warped.png',composite_img)