import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################
    
    # Convert list of bounding boxes to numpy array for vectorized operations
    bbox_array = np.array(bboxes)

    # Calculate heights and mean height
    heights = bbox_array[:, 2] - bbox_array[:, 0]
    mean_height = np.mean(heights)

    # Calculate center positions of bounding boxes
     centers = np.column_stack(((bbox_array[:, 2] + bbox_array[:, 0]) // 2,
                               (bbox_array[:, 3] + bbox_array[:, 1]) // 2,
                               heights, bbox_array[:, 3] - bbox_array[:, 1]))

    # Sort centers by y-coordinate
    centers_sorted = centers[centers[:, 0].argsort()]

    # Cluster rows based on y-coordinate
    rows = []
    row = [centers_sorted[0]]
    for center in centers_sorted[1:]:
        # Check if the center is part of the current row or a new row
        if center[0] > row[-1][0] + mean_height:
            rows.append(sorted(row, key=lambda x: x[1]))  # Sort row by x-coordinate
            row = [center]
        else:
            row.append(center)
    rows.append(sorted(row, key=lambda x: x[1]))  # Sort last row

    # Convert rows back to list of lists, if required
    rows = [row.tolist() for row in rows]

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    import skimage.transform
    
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
data = []

    for row in rows:
        row_data = []
        for y, x, h, w in row:
            # Crop out the character
            crop = bw[y - h // 2:y + h // 2, x - w // 2:x + w // 2]

            # Calculate padding to make the crop square
            padding = max(h, w)
            h_pad = (padding - h) // 2
            w_pad = (padding - w) // 2

            # Pad and resize the crop
            crop_padded = np.pad(crop, ((h_pad, h_pad), (w_pad, w_pad)), 'constant', constant_values=1)
            crop_resized = skimage.transform.resize(crop_padded, (32, 32))

            # Apply erosion and transpose
            crop_processed = skimage.morphology.erosion(crop_resized, kernel)
            crop_transposed = np.transpose(crop_processed)

            # Flatten and add to row data
            row_data.append(crop_transposed.flatten())

        # Add processed row data to main data list
        data.append(np.array(row_data))
        
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    ##########################
    ##### your code here #####
    ##########################
    # Mapping from index to character
    ind2c = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3',30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9'}

    for row_data in data:
        h1 = forward(row_data, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)

        # Using numpy's argmax for faster and more concise indexing
        predicted_indices = np.argmax(probs, axis=1)
        predicted_chars = [ind2c[idx] for idx in predicted_indices]

        # Join the characters to form the string
        row_s = ''.join(predicted_chars)
        print(row_s)

    print("--------------------------------------------------------")

