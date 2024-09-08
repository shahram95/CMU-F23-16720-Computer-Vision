import numpy as np
import cv2

#Import necessary functions
import time
import random

#Write script for Q4.1x
MIN_MATCHES = 20
detector = cv2.ORB_create(nfeatures=1800)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

frame_counter = 0

fps = 0
frame_time = time.time()

def load_input():
    # Use the book template image
    input_image = cv2.imread('../data/cv_cover.jpg')
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray_image, None)

    return gray_image, keypoints, descriptors

def compute_matches(descriptors_input, descriptors_output):
    if len(descriptors_output) != 0 and len(descriptors_input) != 0:
        matches = flann.knnMatch(np.asarray(descriptors_input, np.float32),
                                 np.asarray(descriptors_output, np.float32), k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append(m)
        return good
    else:
        return None

if __name__ == '__main__':
    input_image, input_keypoints, input_descriptors = load_input()

    cap = cv2.VideoCapture('../data/book.mov')
    kungfu_video = cv2.VideoCapture('../data/ar_source.mov')
    
    cap.set(cv2.CAP_PROP_FPS, 60)
    kungfu_video.set(cv2.CAP_PROP_FPS, 60)
    ret, frame = cap.read()
    aug_ret, aug_image = kungfu_video.read()
    
    prev_M = None
    global_start = time.time()

    while ret and aug_ret:
        start = time.time()
        ret, frame = cap.read()
        aug_ret, aug_image = kungfu_video.read()
        
        try:
            # Remove black patch from top
            aug_image = aug_image[52:307, :, :]

            # Resizing to maintain aspect ratio
            width = int(aug_image.shape[1] / aug_image.shape[0]) * input_image.shape[0]
            resized_kfp = cv2.resize(aug_image, (width, input_image.shape[0]), interpolation=cv2.INTER_LINEAR)

            # Taking a central crop
            h, w, d = resized_kfp.shape
            cropped_kfp = resized_kfp[:, int(w/2) - int(input_image.shape[1]/2):int(w/2) + int(input_image.shape[1]/2), :]
            aug_image = cropped_kfp
        except TypeError:
                break
        if frame_counter % 10 == 0:
            if len(input_keypoints) < MIN_MATCHES:
                continue
            
            frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            output_keypoints, output_descriptors = detector.detectAndCompute(frame_bw, None)
            matches = compute_matches(input_descriptors, output_descriptors)

            if matches is not None and len(matches) > 10:
                src_pts = np.float32([input_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([output_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                prev_M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)
            else:
                prev_M = None

        if prev_M is not None:  # If we have a valid homography
            dst = cv2.perspectiveTransform(np.array([
                [0, 0],
                [0, input_image.shape[0] - 1],
                [input_image.shape[1] - 1, input_image.shape[0] - 1],
                [input_image.shape[1] - 1, 0]
            ], dtype='float32').reshape(-1, 1, 2), prev_M)

            M_aug = cv2.warpPerspective(aug_image, prev_M, (frame.shape[1], frame.shape[0]))
            frameb = cv2.fillConvexPoly(frame, dst.astype(int), 0)
            Final = frameb + M_aug
        else:
            Final = frame  # If no homography, show the raw frame
        
        cv2.imshow('16-720 HW2: AR Extra Credit', Final)

        key = cv2.waitKey(1)
        if key == 27:
            break

        frame_counter += 1  # Increment frame counter
    global_fps = frame_counter/(time.time() - global_start)
    print("Global FPS is {}".format(global_fps))
    cap.release()
    kungfu_video.release()
    cv2.destroyAllWindows()