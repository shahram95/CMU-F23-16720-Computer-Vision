import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from LucasKanade import LucasKanade
# write your script here, we recommend the above libraries for making your animation

def track_and_visualize(video_sequence, initial_rect, max_iterations, convergence_threshold, template_update_threshold):
    """
    Tracks the car in the sequence and visualizes the tracking at specific frames using a corrected rect approach.

    Args:
        video_sequence (numpy.ndarray): The video sequence.
        initial_rect (list): Initial rectangle coordinates.
        max_iterations (int): Number of iterations for Lucas-Kanade.
        convergence_threshold (float): Termination threshold for Lucas-Kanade.

    Returns:
        numpy.ndarray: Array of rectangles for all frames.
    """
    tracking_rect = initial_rect[:]
    tracked_rectangles = [initial_rect]
    _, _, total_frames = video_sequence.shape
    initial_frame = video_sequence[:, :, 0]
    initial_parameters = np.zeros(2)

    for frame_index in range(total_frames-1):
        next_frame = video_sequence[:, :, frame_index+1]
        displacement = LucasKanade(initial_frame, next_frame, tracking_rect, convergence_threshold, max_iterations, p0=initial_parameters)
        total_displacement = displacement + [tracking_rect[0] - initial_rect[0], tracking_rect[1]-initial_rect[1]]
        corrected_displacement = LucasKanade(video_sequence[:, :, 0], next_frame, initial_rect, convergence_threshold, max_iterations, p0=total_displacement)
        displacement_change = np.linalg.norm(total_displacement - corrected_displacement)

        if displacement_change < template_update_threshold:
            rect_displacement = (corrected_displacement - [tracking_rect[0]-initial_rect[0], tracking_rect[1] - initial_rect[1]])
            update_rectangle(tracking_rect, rect_displacement)
            initial_frame = video_sequence[:, :, frame_index+1]
            tracked_rectangles.append(tracking_rect)
            initial_parameters = np.zeros(2)
        else:
            new_rect = [tracking_rect[0]+displacement[0], tracking_rect[1]+displacement[1], tracking_rect[2]+displacement[0], tracking_rect[3]+displacement[1]]
            tracked_rectangles.append(new_rect)
            initial_parameters = displacement
        
        if (frame_index+1) in [1, 100, 200, 300, 400]:
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
    saved_rectangles = np.load('carseqrects.npy')
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
    plt.savefig(f'../results/carseq_wrct{frame_number}.png', bbox_inches='tight')


parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]

rects = track_and_visualize(seq, rect, num_iters, threshold, template_threshold)
np.save('carseqrects-wcrt.npy', rects)