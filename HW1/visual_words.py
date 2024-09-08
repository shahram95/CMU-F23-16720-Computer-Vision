import os, multiprocessing
from multiprocessing import Pool
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''

    filter_scales = opts.filter_scales
    # ----- TODO -----
    '''
    The following code follows the following structure:
    1. Checks whether the data type of the pixel values is float, if not type cast it
    2. Checks whether the the pixel values are normalized between 0 and 1, if not provisined to be handled; although this is not required because as a safe practice I have normalized the images before passing them to this function
    3. Checks if the image is grayscale (or possibily have additional channels) and then type cast it to 3 channel RGB
    4. Convert RGB to LAB; from what I read on wikipedia, LAB has a better representation for luminance and chrominance along with texture discrimination thus bolstering consistent filter responses
    5. Initialized a black canvas of H X W X 3F for storing stacked filter responses
    6. Nested loop through the feature scales and indiviual LAB channels for applying gaussian filter, LoG, and directional gaussian filters, and stacked through indexing on the canvas placeholder
    7. The padding is taken place using the flag offered within the scipy.ndimage. module, i.e. mode = ["reflect", "nearest", "constant"]
    8. Return stacked and aligned filters H X W X 3F i.e. for default parameters there are 4 filters and 2 feature scales so F= 4 * 2 = 8 i.e. H X W X 24
    '''
    # Check: dtype is float

    if img.dtype != np.float32:
        img = img.astype(np.float32)
    
    # Check: pixel values Normalized between 0 and 1 
    if np.min(img) < 0.0 or np.max(img) > 1.0:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    
    # Check: 3 channel image
    if img.ndim == 2:
        img = skimage.color.gray2rgb(img)
    elif img.shape[2] > 3:
        img = img[:, :, 3]
    
    num_channels = 3

    # Convert RGB to LAB
    imgLab = skimage.color.rgb2lab(img)

    # Number of filters in the filter bank
    numFilter = 4

    # Initialize the filter responses array for H X W X 3*4*2
    filter_responses = np.zeros((imgLab.shape[0], imgLab.shape[1], 3 * len(filter_scales) * numFilter))
    
    # Looping through feature scales provided through opt, also looping through each channel individually
    for idx, filter_scale in enumerate(filter_scales):
        for channelIdx in range(3):
            # Applying filter on each channel individually and stacking the channel responses accordingly; Padding added through flag in scipy.ndimage
            # [Referenced from util.py]
            filter_responses[:, :, idx*numFilter*num_channels+channelIdx+0] = scipy.ndimage.gaussian_filter( imgLab[:, :,channelIdx], filter_scale, mode='reflect')
            filter_responses[:, :, idx*numFilter*num_channels+channelIdx+3] = scipy.ndimage.gaussian_laplace(imgLab[:, :, channelIdx], filter_scale, mode='reflect')
            filter_responses[:, :, idx*numFilter*num_channels+channelIdx+6] = scipy.ndimage.gaussian_filter(imgLab[:, :, channelIdx], filter_scale, [0, 1], mode='reflect')
            filter_responses[:, :, idx*numFilter*num_channels+channelIdx+9] = scipy.ndimage.gaussian_filter(imgLab[:, :, channelIdx], filter_scale, [1, 0], mode='reflect')
    
    return filter_responses

def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''
    from opts import get_opts
    # ----- TODO -----
    '''
    1. Only a single argument required from the location of function invocation, which is image path. Alpha and data_dir are referenced using the opt module.
    2. As a standard practice, the loaded image is normalized 
    3. The normalized image is passed to receive filter responses
    4. Storing complete filter responses is computationally expensive and memory inefficient. That's why sampling non-repeating random indices without replacement
    5. The question although said to store the responses locally on disk, but shortcircuiting this step to make it computationally more efficient and instead returning the randomly sampled filtered response back to the location of invocation to be stacked
    '''
    im_pth = args
    print("{}".format(im_pth))
    opts = get_opts()
    alpha = opts.alpha

    # Referenced from main.py Q1.1.1
    img = Image.open(join(opts.data_dir, im_pth))
    img = np.array(img, dtype=np.float32)/255.

    # Extracting filter responses for each image
    filter_responses = extract_filter_responses(opts,img)

     # Random sampling without sampling for "alpha" non-repeating indices
    H,W,F = filter_responses.shape
    idx = np.random.choice(H * W, alpha, replace=False)

    '''
    # Increasing runtime efficiency by shortcircuiting writing the file to the disk, instead returning the filter responses back


    if not os.path.exists():
        os.makedirs(feat_dir)
    
    np.save(join(feat_dir,"{}".format(i)),filter_responses.reshape((-1, F))[idx,:])
    '''

    return filter_responses.reshape((-1, F))[idx,:]


def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''
    import sklearn.cluster

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
        
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    # ----- TODO -----


    '''
    1. Calling the computer_dictionary_one_image took ~10 minutes wall time, so parallelized the processing using multiprocessing.pool
    2. Once each pool is finished processing, the filtered sampled response is stacked
    3. Used the scikit-learn documentation to make sense of the clustering algorithms around K centroids
    4. Save the generated cluster centers / centroids into a dictionary and saved it to disk
    '''
    # Creating a pool for parallel processing; executing compute_dictionary_one_image on each thread
    with Pool(processes=n_worker) as pool:
        filter_response = pool.map(compute_dictionary_one_image, train_files)
    
    stacked_responses = np.vstack(filter_response)
    print(stacked_responses.shape)
    # Note: n_jobs deprecated and removed from scikit-learn >= 0.25

    kmeans = sklearn.cluster.KMeans(n_clusters=K, n_jobs =n_worker).fit(stacked_responses)
    dictionary = kmeans.cluster_centers_

    ## example code snippet to save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----

    '''
    1. For the referenced image, receive an filtered response
    2. Reshape the response from H X W X 3F to 
    '''
    # Extract filtered image from filter banks
    img_filter = extract_filter_responses(opts, img)
    
    try:
        H, W, _ = img.shape
    except ValueError:
        H, W = img.shape
        
    # Calculate Euclidean distances between filtered image and dictionary
    eucl_dist = scipy.spatial.distance.cdist(img_filter.reshape(H * W, -1), dictionary, 'euclidean')

    return np.argmin(eucl_dist, axis=1).reshape(H, W)