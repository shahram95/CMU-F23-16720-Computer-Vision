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
	
	return matches, locs1, locs2
