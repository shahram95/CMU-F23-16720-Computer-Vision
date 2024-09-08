import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

def track_and_visualize(video_sequence, initial_rectangle, max_iterations, convergence_threshold):
    """
    Tracks the object in the sequence and visualizes the tracking at specific frames.

    Args:
        video_sequence (numpy.ndarray): The video sequence.
        initial_rectangle (list): Initial rectangle coordinates.
        max_iterations (int): Number of iterations for Lucas-Kanade.
        convergence_threshold (float): Termination threshold for Lucas-Kanade.

    Returns:
        numpy.ndarray: Array of rectangles for all frames.
    """
    total_frames = video_sequence.shape[2]
    tracked_rectangles = [initial_rectangle]

    for frame_index in range(total_frames - 1):
        current_frame = video_sequence[:, :, frame_index]
        next_frame = video_sequence[:, :, frame_index + 1]
        
        displacement = LucasKanade(current_frame, next_frame, initial_rectangle, convergence_threshold, max_iterations)
        
        # Update the rectangle
        updated_rectangle = [initial_rectangle[0] + displacement[0], 
                             initial_rectangle[1] + displacement[1], 
                             initial_rectangle[2] + displacement[0], 
                             initial_rectangle[3] + displacement[1]]
        tracked_rectangles.append(updated_rectangle)
        initial_rectangle = updated_rectangle  # Update the initial_rectangle for the next iteration

        # Visualize tracking at specific frames (e.g., every 20 frames and the first frame)
        if (frame_index + 1) in [0, 20, 40, 60, 80]:
            visualize_tracking(next_frame, updated_rectangle, frame_index + 1)

    return np.array(tracked_rectangles, dtype=int)

def visualize_tracking(frame, rectangle, frame_number):
    """
    Visualizes the tracking on a frame.

    Args:
        frame (numpy.ndarray): The frame to visualize.
        rectangle (list): The rectangle coordinates.
        frame_number (int): Frame number for saving the image.
    """
    plt.figure()
    plt.imshow(frame, cmap='gray')
    plt.axis('off')
    plt.axis('tight')
    
    tracking_patch = patches.Rectangle((rectangle[0], rectangle[1]), 
                                       rectangle[2] - rectangle[0], 
                                       rectangle[3] - rectangle[1], 
                                       edgecolor='r', facecolor='none', linewidth=2)
    
    plt.gca().add_patch(tracking_patch)
    plt.savefig(f'../results/girlseq{frame_number}.png', bbox_inches='tight')

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
    
seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

rectAll = track_and_visualize(seq, rect, num_iters, threshold)
np.save('girlseqrects.npy', rectAll)