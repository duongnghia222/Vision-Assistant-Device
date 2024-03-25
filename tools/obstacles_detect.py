import cv2
import numpy as np
import json
import time

def obstacles_detect(depth_frame, roi, distance_threshold, size_threshold, color_frame, visual=False):
    xmin, ymin, xmax, ymax = roi
    # extract roi from depth frame
    roi_depth_frame = depth_frame[ymin:ymax, xmin:xmax]
    obstacles = []
    # Threshold the depth values
    mask = (roi_depth_frame < distance_threshold).astype(np.uint8) * 255

    # # Show disparity map of depth frame
    # if visual:
    #     depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(roi_depth_frame, alpha=0.03), cv2.COLORMAP_JET)
    #     cv2.imshow('Disparity Map', depth_colormap)
    #     cv2.waitKey(0)
    # # Only Show disparity map of depth frame with pixel values where mask is 1
    # if visual:
    #     depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(roi_depth_frame * mask, alpha=0.03), cv2.COLORMAP_JET)
    #     cv2.imshow('Disparity Map With Mask', depth_colormap)
    #     cv2.waitKey(0)

    # Clustering mask into different obstacles
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > size_threshold:
            print(cv2.contourArea(contour))
            x, y, w, h = cv2.boundingRect(contour)
            obstacles.append([x+xmin, y+ymin, x + w, y + h])
    # draw obstacles on depth frame
    if visual:
        for obstacle in obstacles:
            x1, y1, x2, y2 = obstacle
            cv2.rectangle(color_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return obstacles

# # Load json file
# with open('../depth_frame_values.json', 'r') as file:
#     depth_frame = np.array(json.load(file))
# obstacles = obstacles_detect(depth_frame, [0, 0, 640, 480], 500, 10000, None, visual=True)
# print(obstacles)

