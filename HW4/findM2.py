'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
from submission import essentialMatrix, triangulate, eightpoint
import helper
import matplotlib.pyplot as plt
import os.path
import sys
import numpy as np
import os

def findM2(pts1, pts2, F, K1, K2):
    # Compute the essential matrix
    E = essentialMatrix(F, K1, K2)

    # Initialize the first camera matrix
    M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    C1 = np.matmul(K1, M1)

    # Compute the second camera matrices
    M2s = helper.camera2(E)
    _, _, num = M2s.shape

    # Initialize variables to store the best results
    err_best = float('inf')
    M2_best, C2_best, P_best = None, None, None

    # Iterate over possible second camera matrices
    for i in range(num):
        M2 = M2s[:, :, i]
        C2 = np.matmul(K2, M2)
        P, err = triangulate(C1, pts1, C2, pts2)

        # Check if all points are in front of both cameras
        if np.all(P[:, 2] > 0) and err < err_best:
            err_best = err
            P_best = P
            M2_best = M2
            C2_best = C2

    # Debug output (optional)
    # print('Best M2 (min error): \n', M2_best)

    # Save results if file doesn't exist
    if not os.path.isfile('q3_3.npz'):
        np.savez('q3_3.npz', M2=M2_best, C2=C2_best, P=P_best)

    return M1, C1, M2_best, C2_best, F



if __name__ == '__main__':
    data = np.load('../data/some_corresp.npz')
    Ks = np.load('../data/intrinsics.npz')
    K1 = Ks['K1']
    K2 = Ks['K2']
    pts1 = data['pts1']
    pts2 = data['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    M = np.max(np.shape(im1))
    F = eightpoint(data['pts1'].astype(np.float64), data['pts2'].astype(np.float64), M)
    findM2(pts1, pts2, F, K1, K2)
