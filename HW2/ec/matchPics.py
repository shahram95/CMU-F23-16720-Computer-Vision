import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

def matchPics(I1, I2, opts):
	#I1, I2 : Images to match
	#opts: input opts
	ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
	sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'
	
	assert I1 is not None and I2 is not None, "Input images cannot be None"
	assert hasattr(opts, 'ratio') and hasattr(opts, 'sigma'), "opts must have 'ratio' and 'sigma' attributes"

	#Convert Images to GrayScale
	I1Gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
	I2Gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
	
	#Detect Features in Both Images
	locs1 = corner_detection(I1Gray,sigma)
	locs2 = corner_detection(I2Gray,sigma)
	
	#Obtain descriptors for the computed feature locations
	desc1, locs1 = computeBrief(I1Gray,locs1)
	desc2, locs2 = computeBrief(I2Gray,locs2)

	#Match features using the descriptors
	matches = briefMatch(desc1, desc2, ratio)
	

	'''# Initialize ORB detector
	orb = cv2.ORB_create(n_features=5000)

	# Use ORB to detect keypoints and compute the descriptors for both images
	kp1, desc1 = orb.detectAndCompute(I1Gray, None)
	kp2, desc2 = orb.detectAndCompute(I2Gray, None)

	# Convert keypoints to numpy array format for compatibility
	locs1 = np.array([kp.pt for kp in kp1])
	locs2 = np.array([kp.pt for kp in kp2])

	# Match features using the descriptors
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(desc1, desc2)
	matches = sorted(matches, key=lambda x: x.distance)  # Sort matches based on feature distances'''

	return matches, locs1, locs2
