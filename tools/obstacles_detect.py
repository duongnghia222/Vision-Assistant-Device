import cv2
import numpy as np


def obstacles_detect(depth_frame, roi, distance_threshold, size_threshold, color_frame, visual=False):
    xmin, ymin, xmax, ymax = roi
    # extract roi from depth frame
    roi_depth_frame = depth_frame[ymin:ymax, xmin:xmax]
    obstacles = []
    # Threshold the depth values, if depth is less than distance_threshold then label it 1, otherwise 0
    mask = np.where(roi_depth_frame < distance_threshold, 1, 0).astype(np.uint8)

