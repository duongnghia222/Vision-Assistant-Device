import cv2
import numpy as np

def remove_outliers(data, m=1.5):
    d = np.abs(data - np.mean(data))
    mdev = np.mean(d)
    s = d / mdev if mdev else 0.
    return data[s < m]

def segment_object(depth_frame, roi, threshold_value=10):
    xmin, ymin, xmax, ymax = roi

    # Calculate the center of the bounding box
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2

    half_window_size = 5
    region_xmin = max(center_x - half_window_size, xmin)
    region_ymin = max(center_y - half_window_size, ymin)
    region_xmax = min(center_x + half_window_size + 1, xmax)
    region_ymax = min(center_y + half_window_size + 1, ymax)
    center_region = depth_frame[region_ymin:region_ymax, region_xmin:region_xmax]
    center_region_flat = center_region.flatten()
    center_region_filtered = remove_outliers(center_region_flat)

    if len(center_region_filtered) > 0:
        center_depth = np.mean(center_region_filtered)
    else:
        center_depth = np.mean(center_region_flat)

    # Extract ROI from the depth image
    roi = depth_frame[ymin:ymax, xmin:xmax]

    # Create a mask based on the threshold criteria
    mask = np.abs(roi - center_depth) < threshold_value

    # Calculate the average depth where the mask is True
    if np.any(mask):
        average_depth = int(np.mean(roi[mask]))
    else:
        average_depth = 0

    return mask.astype(np.uint8) * 255, average_depth  # Convert mask to 8-bit format
