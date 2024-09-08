import numpy as np
import cv2
#Import necessary functions
from loadVid import loadVid
from matchPics import matchPics
from opts import get_opts
from planarH import computeH_ransac, compositeH

#Write script for Q3.1

opts = get_opts()

# Loading video frames for book and Kungfu panda AR source
book_frames = loadVid('../data/book.mov')
kfp_frames = loadVid('../data/ar_source.mov')
cv_cover = cv2.imread('../data/cv_cover.jpg')

# Get dimensions and initialize video writer for 25 FPS
h,w,_=book_frames[0].shape
out=cv2.VideoWriter('../result/ar.avi',cv2.VideoWriter_fourcc('X','V','I','D'),25,(w,h))

H2to1 = None
coverW = cv_cover.shape[1]
coverH = cv_cover.shape[0]

# Iterate through each frame, using video with the shortest length as termination
for i in range(min(len(book_frames), len(kfp_frames))):
    print("Processing frame # {}".format(i))
    book_frame = book_frames[i]
    kfp_frame = kfp_frames[i]

    # Match features using BRIEF
    matches, locs1, locs2 = matchPics(cv_cover, book_frame, opts)
    
    # Extract matched points
    x1 = locs1[matches[:,0], 0:2]
    x2 = locs2[matches[:,1], 0:2]
    
    # Computing homography using RANSAC
    H2to1, inliers = computeH_ransac(x1, x2, opts)
    
    # Pre-processing kungfu panda video frame i.e. black padding removal + central crop
    kfp_frame = kfp_frame[52:307,:,:]
    
    width = int(kfp_frame.shape[1]/kfp_frame.shape[0]) * coverH

    resized_kfp = cv2.resize(kfp_frame, (width,coverH), interpolation = cv2.INTER_LINEAR)
    h, w, d = resized_kfp.shape
    cropped_ar= resized_kfp[:,int(w/2)-int(coverW/2):int(w/2)+int(coverW/2),:]
    
    # Composite the the kungfu panda frame onto the CV cover book image
    warped_out = compositeH(H2to1, cropped_ar, book_frame)
    out.write(warped_out)