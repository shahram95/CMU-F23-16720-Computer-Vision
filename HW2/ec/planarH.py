import numpy as np
import cv2


def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points
	
	# Check: Ensure the input matrices have correct dimensions
	assert x1.shape[0] >= 4 and x2.shape[0] >= 4, "There should be at least 4 point correspondences."
	assert x1.shape[1] == 2 and x2.shape[1] == 2, "Input point matrices should be N x 2 in size."

	A = []
	for i in range(x1.shape[0]):
		u1, v1 = x1[i]
		u2, v2 = x2[i]
		A.append([-u2, -v2, -1, 0, 0, 0, u2*u1, v2*u1, u1])
		A.append([0, 0, 0, -u2, -v2, -1, u2*v1, v2*v1, v1])

	A = np.array(A)
	_, _, V_t = np.linalg.svd(A)
	H2to1 = V_t[-1].reshape(3, 3)

	return H2to1

def computeH_norm(x1, x2):
	#Q2.2.2
	#Compute the centroid of the points
	x1_centroid = np.mean(x1, axis=0)
	x2_centroid = np.mean(x2, axis=0)

	#Shift the origin of the points to the centroid
	x1_shifted = x1 - x1_centroid
	x2_shifted = x2 - x2_centroid

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	scale_1 = np.sqrt(2) / np.max(np.sqrt(np.sum(x1_shifted**2, axis=1)))
	scale_2 = np.sqrt(2) / np.max(np.sqrt(np.sum(x2_shifted**2, axis=1)))


	#Similarity transform 1
	T1 = np.array([
		[scale_1, 0, -scale_1 * x1_centroid[0]],
		[0, scale_1, -scale_1 * x1_centroid[1]],
		[0, 0, 1]
		])

	#Similarity transform 2
	T2 = np.array([
		[scale_2, 0, -scale_2 * x2_centroid[0]],
		[0, scale_2, -scale_2 * x2_centroid[1]],
		[0, 0, 1]
		])
        # Apply similarity transforms
	x1_normalized = (T1 @ np.vstack((x1.T, np.ones(x1.shape[0])))).T
	x2_normalized = (T2 @ np.vstack((x2.T, np.ones(x2.shape[0])))).T

	# Strip the homogenous coordinate
	x1_normalized = x1_normalized[:, :2]
	x2_normalized = x2_normalized[:, :2]
	
	#Compute homography
	H_norm = computeH(x1_normalized, x2_normalized)

	#Denormalization
	H2to1 = np.linalg.inv(T1) @ H_norm @ T2

	return H2to1


def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

        # Convert points to homogenous coordinates
	x1_homo = np.hstack((locs1, np.ones((locs1.shape[0], 1)))).T
	x2_homo = np.hstack((locs2, np.ones((locs2.shape[0], 1)))).T
	
	best_inliers = np.zeros(locs1.shape[0], dtype=bool)  # Initialize the best inliers set

	for i in range(max_iters):
		random_idx = np.random.choice(locs1.shape[0], 4, replace=False) # Random selection of 4 point correspondences
		randSet1 = locs1[random_idx, :]
		randSet2 = locs2[random_idx, :]
		
		# Compute the homography using computeH_norm function
		est_homo = computeH_norm(randSet1, randSet2)
		
		# Apply the homography to locs2
		x1_est = np.dot(est_homo, x2_homo)
		x1_est /= x1_est[2, :]  # Normalize the estimated points

		# Compute the squared error
		error = np.sum((x1_homo[:2, :] - x1_est[:2, :])**2, axis=0)

		# Identify the inliers
		inliers = error < inlier_tol**2	

		# Update the best homography if the number of inliers is maximized
		if np.sum(inliers) > np.sum(best_inliers):
			best_inliers = inliers
	
	# Re-compute the homography using all inliers
	bestH2to1 = computeH_norm(locs1[best_inliers, :], locs2[best_inliers, :])
	inliers = best_inliers
	return bestH2to1, inliers

def compositeH(H2to1, template, img):
 	
 	#Create a composite image after warping the template image on top
 	#of the image using the homography

 	#Note that the homography we compute is from the image to the template;
 	#x_template = H2to1*x_photo
 	#For warping the template to the image, we need to invert it.
       
    #Create mask of same size as template
    transposed_template = cv2.transpose(template)
    template_mask = np.ones_like(transposed_template)
    warped_mask=cv2.warpPerspective(template_mask,np.linalg.inv(H2to1),(img.shape[0],img.shape[1]))
    transposed_warped_mask=cv2.transpose(warped_mask)
    mask_idx = np.nonzero(transposed_warped_mask)
	
    #Warp template by appropriate homography
    warped_template=cv2.warpPerspective(transposed_template,np.linalg.inv(H2to1),(img.shape[0],img.shape[1]))
    transposed_warped_template = cv2.transpose(warped_template)
	
    #Use mask to combine the warped template and the image
    composite_img = img.copy()
    composite_img[mask_idx]=transposed_warped_template[mask_idx]
    
    
    return composite_img

