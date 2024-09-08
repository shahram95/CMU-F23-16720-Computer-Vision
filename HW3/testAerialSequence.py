import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from SubtractDominantMotion import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation

# Function to visualize the motion in a video sequence
def visualize_motion(seq, num_iters, threshold, tolerance):
    """
    Visualizes the subtracted dominant motion on specific frames of the sequence.

    Args:
        seq (numpy.ndarray): The video sequence.
        num_iters (int): Number of iterations for Lucas-Kanade.
        threshold (float): Termination threshold for Lucas-Kanade.
        tolerance (float): Binary threshold of intensity difference when computing the mask.
    """
    num_frames = seq.shape[2]

    for i in range(num_frames-1):
        print(i)
        current_frame = seq[:, :, i]
        next_frame = seq[:, :, i+1]
        
        mask = SubtractDominantMotion(current_frame, next_frame, threshold, num_iters, tolerance)
        
        static_objects = np.where(mask == 0)

        if (i+1) in [30,60,90,120]:
            visualize(next_frame, static_objects, i+1)

# Function to visualize static objects in a frame
def visualize(frame, objects, frame_num):
    """
    Visualizes the static objects in a frame.

    Args:
        frame (numpy.ndarray): The frame to visualize.
        objects (tuple): Coordinates of static objects.
        frame_num (int): Frame number for saving the image.
    """

    # Calculate the width and height bounds
    height, width = frame.shape
    h_bound = int(0.1 * height)
    w_bound = int(0.1 * width)
    
    # Filter out the edge coordinates
    mask = (objects[0] >= h_bound) & (objects[0] < height - h_bound) & \
           (objects[1] >= w_bound) & (objects[1] < width - w_bound)
    
    filtered_y = objects[0][mask]
    filtered_x = objects[1][mask]
    
    plt.figure()
    plt.imshow(frame, cmap='gray')
    
    # Plot the filtered coordinates
    plt.scatter(filtered_x, filtered_y, color='blue', edgecolor='None', alpha=0.5)
    
    plt.axis('off')
    plt.savefig(f'../results/aerialseqICA{frame_num}.png', bbox_inches='tight')

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=5e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.2, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/aerialseq.npy')

visualize_motion(seq, num_iters, threshold, tolerance)