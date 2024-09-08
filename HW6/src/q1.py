# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
import numpy as np
from matplotlib import pyplot as plt
from utils import integrateFrankot
import cv2
from skimage.color import rgb2xyz
from PIL import Image
from matplotlib import cm
from scipy.ndimage import gaussian_filter

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """
    image = np.zeros((res[1], res[0]))
    cx, cy = res[0] / 2, res[1] / 2

    light = light / np.linalg.norm(light)
    j, i = np.meshgrid(np.arange(res[0]), np.arange(res[1]))

    x = (j - cx) * pxSize + center[0]
    y = (i - cy) * pxSize + center[1]

    mask = (x - center[0])**2 + (y - center[1])**2 <= rad**2

    z = np.sqrt(rad**2 - (x - center[0])**2 - (y - center[1])**2) + center[2]
    z[~mask] = 0

    normals = np.dstack(((x - center[0]), (y - center[1]), z - center[2]))
    norms = np.linalg.norm(normals, axis=2)
    normals /= norms[:,:,np.newaxis]

    intensity = np.einsum('ijk,k->ij', normals, light)
    intensity[intensity < 0] = 0
    intensity[~mask] = 0

    image = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))

    return image


def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """

    vectorized_images = []

    for i in range(1, 8):
        img = cv2.imread(path + f"input_{i}.tif", cv2.IMREAD_UNCHANGED)
        img_xyz = rgb2xyz(img)
        luminance = img_xyz[:, :, 1]
        vectorized_images.append(luminance.reshape(-1))

    I = np.stack(vectorized_images)
    L = np.load(path + "sources.npy").T
    s = img_xyz.shape[:2]


    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    B, _, _, _ = np.linalg.lstsq(L.T, I, rcond=None)
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normalfrom sklearn.preprocessing import normalizes
    '''

    albedos = np.linalg.norm(B, axis=0)
    epsilon = 1e-6
    albedos_safe = albedos + epsilon
    normals = B / albedos_safe
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):
    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `gray` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = np.reshape((albedos/np.max(albedos)), s)
    normalIm = np.reshape(((normals+1.)/2.).T, (s[0], s[1], 3))

    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    epsilon = 1e-6
    zx = np.reshape(normals[0, :] / (-normals[2, :] + epsilon), s)
    zy = np.reshape(normals[1, :] / (-normals[2, :] + epsilon), s)
    surface = integrateFrankot(zx, zy)
    return surface

def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """

    h, w = surface.shape
    y, x = np.arange(h), np.arange(w)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, gaussian_filter(surface, sigma=1), edgecolor='none', cmap=cm.coolwarm)
    ax.set_title('Surface Plot')
    plt.show()

def normalize(img):
    # If the image is multi-channel (e.g., RGB), process each channel independently
    if len(img.shape) == 3:
        # Normalize each channel
        channels = [normalize(channel) for channel in np.rollaxis(img, axis=-1)]
        return np.stack(channels, axis=-1)

    # Single-channel image normalization
    min_v, max_v = np.min(img), np.max(img)
    # Prevent division by zero and handle the case where all pixel values are equal
    if min_v == max_v:
        return img - min_v
    else:
        normalized_img = (img - min_v) / (max_v - min_v)

    # Convert back to original datatype if it's an integer type
    if img.dtype.kind in 'ui':  # Unsigned and signed integer types
        # Scale to the max value of the original datatype
        max_dtype_value = np.iinfo(img.dtype).max
        normalized_img = (normalized_img * max_dtype_value).astype(img.dtype)

    return normalized_img


if __name__ == '__main__':
    # Q1(b)
    # Define the parameters
    center = np.array([0, 0, 0])
    rad = 0.75
    pxSize = 7e-4
    res = np.array([3840, 2160])

    # Lighting directions
    lights = [
        np.array([1, 1, 1]) / np.sqrt(3),
        np.array([1, -1, 1]) / np.sqrt(3),
        np.array([-1, -1, 1]) / np.sqrt(3)
    ]

    # Loop through each lighting direction and render the sphere
    for i, light in enumerate(lights, start=1):
        image = renderNDotLSphere(center, rad, light, pxSize, res)

        # Display the image
        plt.imshow(image, cmap='gray')
        plt.title(f'Rendered Sphere with Light Direction {i}')
        plt.axis('off')
        plt.show()

        # Optionally, save the image
        plt.imsave(f'rendered_sphere_light_{i}.png', image, cmap='gray')
    
    # Q1(c)
    I, L, s = loadData()
    print(L)
    # Q1(d)
    U, S, Vt = np.linalg.svd(I, full_matrices=False)
    print(S)
    
    # Q1(e)
    B = estimatePseudonormalsCalibrated(I, L)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    
    # Q1(f)
    # Albedos Image
    plt.imshow(albedoIm, cmap='gray')
    plt.show()

    # Normals Image
    normalIm = normalize(normalIm)
    plt.imshow(normalIm, cmap='rainbow')
    plt.show()

    # Q1(i)
    surface = normalize(estimateShape(normals, s))
    surface *= 255
    plotSurface(surface)
