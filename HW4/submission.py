"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import util
import pdb
import os
import scipy
import scipy.ndimage as ndimage
from scipy import optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    # Initialize the matrix A with the right shape
    num_points = pts1.shape[0]
    matrix_A = np.empty((num_points, 9))

    # Normalization transformation matrix
    normalization_transform = np.diag([1/M, 1/M, 1])

    # Normalize the points
    pts1 /= M
    pts2 /= M

    # Extract x and y coordinates
    x1, y1 = pts1[:, 0], pts1[:, 1]
    x2, y2 = pts2[:, 0], pts2[:, 1]

    # Construct the matrix A based on the outer product of point correspondences
    matrix_A = np.vstack((x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones(num_points))).T

    # Apply Singular Value Decomposition (SVD)
    _, _, vh = np.linalg.svd(matrix_A)

    # The fundamental matrix F is the last row of V (V^H in numpy's SVD)
    fundamental_matrix = vh[-1].reshape(3, 3)

    # Refine the fundamental matrix using the utility functions
    fundamental_matrix = util.refineF(fundamental_matrix, pts1, pts2)
    fundamental_matrix = util._singularize(fundamental_matrix)

    # Denormalize the fundamental matrix
    fundamental_matrix = normalization_transform.T @ fundamental_matrix @ normalization_transform
    if not (os.path.isfile('q2_1.npz')):
        np.savez('q2_1.npz',F = fundamental_matrix, M = M)
    return fundamental_matrix


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = K2.T @ F @ K1
    if not (os.path.isfile('q3_1.npz')):
        np.savez('q3_1.npz',E=E)
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    num_points, _ = pts1.shape
    points_3d = np.zeros((num_points, 3))
    points_homogeneous = np.zeros((num_points, 4))
    total_error = 0

    # Construct A matrix and perform SVD for each point
    for i in range(num_points):
        A = np.vstack([
            pts1[i, 0] * C1[2, :] - C1[0, :],
            pts1[i, 1] * C1[2, :] - C1[1, :],
            pts2[i, 0] * C2[2, :] - C2[0, :],
            pts2[i, 1] * C2[2, :] - C2[1, :]
        ])
        _, _, vh = np.linalg.svd(A)
        point_3d_homogeneous = vh[-1, :]
        point_3d_homogeneous /= point_3d_homogeneous[3]
        points_3d[i, :] = point_3d_homogeneous[:3]
        points_homogeneous[i, :] = point_3d_homogeneous

    # Project points back into each camera and calculate error
    projected_pts1 = (C1 @ points_homogeneous.T)[:-1]
    projected_pts1 /= projected_pts1[-1, :]
    
    projected_pts2 = (C2 @ points_homogeneous.T)[:-1]
    projected_pts2 /= projected_pts2[-1, :]
    
    error_1 = np.sum((projected_pts1.T - pts1) ** 2)
    error_2 = np.sum((projected_pts2.T - pts2) ** 2)
    total_error = error_1 + error_2

    return points_3d, total_error


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    
    point1 = np.array([[x1], [y1], [1]])
    fundamentalMatrix = F
    epipolarLine = fundamentalMatrix.dot(point1)
    height, width, _ = im1.shape
    searchRangeY = np.arange(max(0, y1-30), min(height, y1+30))
    searchRangeX = (-(epipolarLine[1] * searchRangeY + epipolarLine[2]) / epipolarLine[0]).astype(int)

    filterSize = 5
    filteredIm1 = ndimage.gaussian_filter(im1, sigma=1, output=np.float64)
    filteredIm2 = ndimage.gaussian_filter(im2, sigma=1, output=np.float64)

    minError = np.inf
    bestMatchIndex = 0

    for i, (x2, y2) in enumerate(zip(searchRangeX, searchRangeY)):
        if filterSize <= x2 < width - filterSize and filterSize <= y2 < height - filterSize:
            patch1 = filteredIm1[y1-filterSize:y1+filterSize+1, x1-filterSize:x1+filterSize+1, :]
            patch2 = filteredIm2[y2-filterSize:y2+filterSize+1, x2-filterSize:x2+filterSize+1, :]
            error = np.sum((patch1 - patch2) ** 2)

            if error < minError:
                minError = error
                bestMatchIndex = i

    matchedX, matchedY = searchRangeX[bestMatchIndex], searchRangeY[bestMatchIndex]
    if not os.path.isfile('q4_1.npz'):
        np.savez('q4_1.npz', F=fundamentalMatrix, pts1=points1, pts2=points2)
    return matchedX, matchedY

'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=0.9):
    # Replace pass by your implementation
    max_inliers = -1
    best_F = None
    best_inliers = None
    
    # Convert points to homogeneous coordinates
    pts1_homo = np.vstack((pts1.T, np.ones(pts1.shape[0])))
    pts2_homo = np.vstack((pts2.T, np.ones(pts2.shape[0])))

    for i in range(nIters):
        print("idx: {}".format(i))
        # Randomly sample 8 points for the fundamental matrix calculation
        random_indices = np.random.choice(pts1.shape[0], 8, replace=False)
        sample_pts1 = pts1[random_indices]
        sample_pts2 = pts2[random_indices]
        
        # Compute the fundamental matrix using the eight-point algorithm
        F_candidate = eightpoint(sample_pts1, sample_pts2, M)
        
        # Compute the epipolar lines for pts1
        epipolar_lines = F_candidate @ pts1_homo
        
        # Compute the distances from pts2 to the epipolar lines
        distances = np.sum((pts2_homo * epipolar_lines), axis=0)
        distances /= np.linalg.norm(epipolar_lines[:2, :], axis=0)
        inliers_mask = np.abs(distances) < tol
        
        # Count inliers
        inlier_count = np.sum(inliers_mask)

        # Update best F and inliers if current F has more inliers
        if inlier_count > max_inliers:
            max_inliers = inlier_count
            best_F = F_candidate
            best_inliers = inliers_mask
    print(np.sum(best_inliers))
    return best_F, best_inliers

'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    angle = np.linalg.norm(r)
    if angle == 0:
        return np.eye(3)

    # Normalized axis of rotation
    axis = r / angle
    axis_x, axis_y, axis_z = axis.flatten()

    # Cross-product matrix of the normalized axis
    cross_product_matrix = np.array([[0, -axis_z, axis_y],
                                     [axis_z, 0, -axis_x],
                                     [-axis_y, axis_x, 0]])

    # Outer product of the axis with itself
    outer_product_axis = np.outer(axis, axis)

    # Rodrigues' rotation formula
    rotation_matrix = (np.eye(3) * np.cos(angle) +
                       (1 - np.cos(angle)) * outer_product_axis +
                       np.sin(angle) * cross_product_matrix)

    return rotation_matrix

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    # Skew-symmetric part of R
    skew_sym_part = (R - R.T) / 2
    rho = np.array([[skew_sym_part[2, 1]], [skew_sym_part[0, 2]], [skew_sym_part[1, 0]]])
    s = np.linalg.norm(rho)
    c = (np.trace(R) - 1) / 2

    # Handling the special cases
    if s == 0 and c == 1:
        # No rotation
        return np.zeros((3, 1))
    elif s == 0 and c == -1:
        # Pi rotation, direction of rotation axis is ambiguous
        R_plus_eye = R + np.eye(3)
        for i in range(3):
            if np.linalg.norm(R_plus_eye[:, i]) != 0:
                v = R_plus_eye[:, i]
                break
        u = v / np.linalg.norm(v)
        r = np.pi * u
        r = r.reshape(3, 1)
        # Adjust the sign of the rotation vector
        r = np.where(np.logical_and(r == 0, np.any(r < 0)), -r, r)
        return r
    elif s != 0:
        # General case
        u = rho / s
        theta = np.arctan2(s, c)
        return theta * u

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass
