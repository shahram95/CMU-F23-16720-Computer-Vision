# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals, normalize
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability
import matplotlib.pyplot as plt
import cv2
def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """

    U, S, Vt = np.linalg.svd(I, full_matrices=False)
    S_3x3 = np.diag(S[:3])
    B = np.dot(np.sqrt(S_3x3), Vt[:3, :])
    L = np.dot(U[:, :3], np.sqrt(S_3x3)).T

    return B, L


if __name__ == "__main__":
    
    
    '''# Question b
    I, originalL, s = loadData()
    B, L = estimatePseudonormalsUncalibrated(I)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    print(L)

    # Albedos
    plt.imshow(albedoIm, cmap='gray')
    cv2.imwrite('../results/q1e_albedo.png', (albedoIm*255))
    plt.show()

    # Normals Image
    normalIm = normalize(normalIm)
    plt.imshow(normalIm, cmap='rainbow')
    plt.savefig('../results/q1e_normal.png')
    plt.show()
    '''
    
    '''
    # Question d
    I, L0, s = loadData()
    B, L = estimatePseudonormalsUncalibrated(I)
    albedos, normals = estimateAlbedosNormals(B)
    surface = estimateShape(normals, s)    
    min_v, max_v = np.min(surface), np.max(surface)
    surface = (surface - min_v) / (max_v - min_v)   
    surface = (surface * 255.).astype('uint8')
    plotSurface(surface)
    '''

    '''
    # Question e
    I, L0, s = loadData()
    B, L = estimatePseudonormalsUncalibrated(I)
    Nt = enforceIntegrability(B, s)
    albedos, normals = estimateAlbedosNormals(Nt)
    surface = estimateShape(normals, s)    
    min_v, max_v = np.min(surface), np.max(surface)
    surface = (surface - min_v) / (max_v - min_v)  
    surface = (surface * 255.).astype('uint8')
    plotSurface(surface)
    '''
    
    # Question f
    I, L0, s = loadData()
    B, L = estimatePseudonormalsUncalibrated(I)
    mu= 0.1
    v=0.1
    lambd=100
    G=np.array([[1,0,0],[0,1,0],[mu,v,lambd]])
    print(G)
    G_invT=np.linalg.inv(G).T
    Nt = enforceIntegrability(B, s)
    GTB=np.dot(G_invT,Nt)
    albedos, normals = estimateAlbedosNormals(GTB)
    surface = estimateShape(normals, s)
    
    min_v, max_v = np.min(surface), np.max(surface)
    surface = (surface - min_v) / (max_v - min_v)
    
    surface = (surface * 255.).astype('uint8')
    plotSurface(surface)
    