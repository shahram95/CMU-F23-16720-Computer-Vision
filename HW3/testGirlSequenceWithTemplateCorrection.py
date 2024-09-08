import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from LucasKanade import LucasKanade
# write your script here, we recommend the above libraries for making your animation

def track_and_visualize(video_sequence, initial_rectangle, max_iterations, convergence_threshold, template_update_threshold):
    """
    Tracks the object in the sequence and visualizes the tracking at specific frames using a corrected rect approach.

    Args:
        video_sequence (numpy.ndarray): The video sequence.
        initial_rectangle (list): Initial rectangle coordinates.
        max_iterations (int): Number of iterations for Lucas-Kanade.
        convergence_threshold (float): Termination threshold for Lucas-Kanade.

    Returns:
        numpy.ndarray: Array of rectangles for all frames.
    """
    tracking_rectangle = initial_rectangle[:]
    tracked_rectangles = [initial_rectangle]
    image_height, image_width, total_frames = video_sequence.shape
    initial_frame = video_sequence[:, :, 0]
    initial_parameters = np.zeros(2)

    for frame_index in range(total_frames-1):
        next_frame = video_sequence[:, :, frame_index+1]
        displacement = LucasKanade(initial_frame, next_frame, tracking_rectangle, convergence_threshold, max_iterations, p0=initial_parameters)
        total_displacement = displacement + [tracking_rectangle[0] - initial_rectangle[0], tracking_rectangle[1]-initial_rectangle[1]]
        corrected_displacement = LucasKanade(video_sequence[:, :, 0], next_frame, initial_rectangle, convergence_threshold, max_iterations, p0=total_displacement)
        displacement_change = np.linalg.norm(total_displacement - corrected_displacement)

        if displacement_change < template_update_threshold:
            rect_displacement = (corrected_displacement - [tracking_rectangle[0]-initial_rectangle[0], tracking_rectangle[1] - initial_rectangle[1]])
            update_rectangle(tracking_rectangle, rect_displacement)
            initial_frame = video_sequence[:, :, frame_index+1]
            tracked_rectangles.append(tracking_rectangle)
            initial_parameters = np.zeros(2)
        else:
            new_rectangle = [tracking_rectangle[0]+displacement[0], tracking_rectangle[1]+displacement[1], tracking_rectangle[2]+displacement[0], tracking_rectangle[3]+displacement[1]]
            tracked_rectangles.append(new_rectangle)
            initial_parameters = displacement
        
        if (frame_index+1) in [1, 20, 40, 60, 80]:
            visualize_tracking(video_sequence[:, :, frame_index+1], tracked_rectangles[-1], frame_index+1)

    return np.array(tracked_rectangles, dtype=int)

def update_rectangle(rectangle, displacement):
    rectangle[0] += displacement[0]
    rectangle[1] += displacement[1]
    rectangle[2] += displacement[0]
    rectangle[3] += displacement[1]

def visualize_tracking(frame, rectangle, frame_number):
    """
    Visualizes the tracking on a frame for both the original and corrected rectangle tracking.

    Args:
        frame (numpy.ndarray): The frame to visualize.
        rectangle (list): The rectangle coordinates.
        frame_number (int): Frame number for saving the image.
    """
    saved_rectangles = np.load('girlseqrects.npy')
    saved_rectangle = saved_rectangles[frame_number, :]
    
    plt.figure()
    plt.imshow(frame, cmap='gray')
    plt.axis('off')
    
    original_patch = patches.Rectangle((saved_rectangle[0], saved_rectangle[1]), (saved_rectangle[2]-saved_rectangle[0]), 
                                       (saved_rectangle[3]-saved_rectangle[1]), edgecolor='b', facecolor='none', linewidth=2)
    
    tracking_patch = patches.Rectangle((rectangle[0], rectangle[1]), (rectangle[2]-rectangle[0]), 
                                       (rectangle[3]-rectangle[1]), edgecolor='r', facecolor='none', linewidth=2)
    
    plt.gca().add_patch(original_patch)
    plt.gca().add_patch(tracking_patch)
    plt.savefig(f'../results/girlseq_wrct{frame_number}.png', bbox_inches='tight')


parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold
    
seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

rects = track_and_visualize(seq, rect, num_iters, threshold, template_threshold)
np.save('girlseqrects-wcrt.npy', rects)