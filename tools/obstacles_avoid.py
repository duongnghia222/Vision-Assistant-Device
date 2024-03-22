import cv2
import numpy as np

def obstacles_detect(depth_frame, anotated_frame, roi, base_height, threshold, visual=False):
    xmin, ymin, xmax, ymax = roi

    depth_range_mask = np.logical_and(depth_frame >= base_height, depth_frame <= base_height + threshold)
    roi_mask = np.zeros_like(depth_range_mask)
    roi_mask[ymin: ymax, xmin: xmax] = depth_range_mask[ymin: ymax, xmin: xmax]
    if visual:
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow("Depth Frame", depth_colormap)
        anotated_frame[roi_mask] = [0, 0, 0]
    segmented_objects = None
    return segmented_objects