import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts

import scipy.ndimage
import matplotlib.pyplot as plt
from helper import plotMatches

opts = get_opts()
#Q2.1.6
#Read the image and convert to grayscale, if necessary

match_bin_count = []
rotation_angles = []
angle_increment = 10

cover_image = cv2.imread("../data/cv_cover.jpg")
opt = get_opts()

for i in range(36):
	#Rotate Image
	rotation_angle = i * angle_increment
	print("Processing rotation: {}".format(rotation_angle))
	rotated_cover = scipy.ndimage.rotate(cover_image, rotation_angle, reshape=False)
	#Compute features, descriptors and Match features

	#Update histogram
	matches, locs_original, locs_rotated = matchPics(cover_image, rotated_cover, opts)
	match_bin_count.append(len(matches))
	rotation_angles.append(rotation_angle)
	# Visualization for specific angles
	if rotation_angle in [30, 120, 210]:
		plotMatches(cover_image, rotated_cover, matches, locs_original, locs_rotated)


#Display histogram

plt.figure()
plt.hist(rotation_angles, bins=np.arange(0, 360, angle_increment), weights=match_bin_count, color='blue', alpha=0.75)
plt.xlabel('Rotation Angles (degrees)')
plt.ylabel('Number of Matches')
plt.title('Number of Matches for Different Angles')
plt.show()