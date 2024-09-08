import os, math, multiprocessing
from multiprocessing import Pool
from os.path import join
from copy import copy
from functools import partial
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
import visual_words


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    # ----- TODO -----

    # A more efficient way to create word histogram without explicity assigning/forming bins

    word_hist = np.bincount(wordmap.ravel(), minlength=K)
    
    # L1-normalized output return
    return word_hist/np.sum(word_hist)

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    # ----- TODO -----

    # Load the saved dictionary of K clustered centroids
    dictionary = np.load("dictionary.npy")
    dict_size = len(dictionary)

    # Initialize a storage reference for histogram of visual words using spatial pyramid matching (SPM)
    hist_all = list()

    # l adjusted through zero-indexing of python; creating submaps at each level of the Spatial Pyramid
    for l in range(L):
        layer = 2 ** l
        weight = 2 ** (l-L)

        submaps = np.array_split(wordmap, layer, axis=0)
        submaps = [np.array_split(submap, layer, axis=1) for submap in submaps]

        for row in submaps:
            for submap in row:
                hist = get_feature_from_wordmap(opts, submap)
                hist_all.extend(hist*weight)
        
    hist_all = np.array(hist_all)

    return hist_all/np.sum(hist_all)
    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K*(4^L-1)/3)
    '''

    # ----- TODO -----

    # Referenced from main.py() Q1.1.1
    img = Image.open(join('../data',img_path))
    img = np.array(img).astype(np.float32)/255
    print(str(join('../data',img_path)))       # For troubleshooting

    # Calling function for visual wordmap, and SPM feature extraction
    wordmap = visual_words.get_visual_words(opts,img,dictionary)
    extracted_feature = get_feature_from_wordmap_SPM(opts,wordmap)
    return extracted_feature


def parallelization_wrapper(args):
    opts, img_path, dictionary = args
    return get_image_feature(opts, img_path, dictionary)

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    #para_list = list(zip(img_list, train_files, train_labels))
    # ----- TODO -----
    
    train_data = np.array(train_files)
    train_labels = np.array(train_labels)
    K = opts.K
    nPts_train = len(train_files)
    
    total_spatial_bins = 4 ** SPM_layer_num
    total_spatial_bins_excluding_root = total_spatial_bins - 1
    total_histogram_bins = K * total_spatial_bins_excluding_root
    numHBM = total_histogram_bins // 3
    stacked_features=np.empty((0, numHBM))
    
    for idx in range(nPts_train):
        img_feat = get_image_feature(opts,train_data[idx],dictionary) 
        stacked_features= np.vstack([stacked_features, img_feat])
    
    '''args_list = [(opts, img_path, dictionary) for img_path in train_data]

    with Pool(processes=n_worker) as pool:    
        results = pool.map(partial(get_image_feature, dictionary=dictionary), args_list)
        
    features = np.vstack(results)'''

    ## example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=stacked_features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    
    return np.sum(np.minimum(word_hist, histograms), axis=1)
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # ----- TODO -----
    features = trained_system['features']
    train_labels = trained_system['labels']
    test_data  = np.array(test_files).astype(str)

    test_features = [get_image_feature(opts, img_path, dictionary) for img_path in test_data]
    similarities = [distance_to_set(test_feature, features) for test_feature in test_features]
    pred_labels = [train_labels[np.argmax(similarity)] for similarity in similarities]

    conf_matrix = confusion_matrix(test_labels, pred_labels)
    accuracy = accuracy_score(test_labels, pred_labels)

    return(conf_matrix,accuracy)