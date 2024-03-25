import cv2
import numpy as np


def segment_object(depth_frame, roi, threshold_value=10):
    xmin, ymin, xmax, ymax = roi

    # Calculate the center of the bounding box
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2

    # Extract the center depth value
    center_depth = depth_frame[center_y, center_x]

    # Extract ROI from the depth image
    roi = depth_frame[ymin:ymax, xmin:xmax]

    # Create a mask based on the threshold criteria
    mask = np.abs(roi - center_depth) < threshold_value

    # Calculate the average depth where the mask is True
    if np.any(mask):
        average_depth = np.mean(roi[mask])
    else:
        average_depth = None

    return mask.astype(np.uint8) * 255, int(average_depth)  # Convert mask to 8-bit format
