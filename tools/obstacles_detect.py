import cv2
import numpy as np
import json
import time

def obstacles_detect(depth_frame, roi, distance_threshold, size_threshold, color_frame=None):
    xmin, ymin, xmax, ymax = roi
    if color_frame is not None:
        cv2.rectangle(color_frame, (xmin, ymin), (xmax, ymax), (233, 100, 255), 2)
    # extract roi from depth frame
    roi_depth_frame = depth_frame[ymin:ymax, xmin:xmax]
    obstacles = []
    # Threshold the depth values
    mask = ((roi_depth_frame < distance_threshold) & (roi_depth_frame != 0)).astype(np.uint8) * 255

    # Clustering mask into different obstacles
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # visualizing the contours
    # if color_frame is not None: pass
        # cv2.drawContours(color_frame, contours, -1, (0, 255, 0), 2)
    for contour in contours:
        if cv2.contourArea(contour) > size_threshold:
            print(cv2.contourArea(contour))
            # Calculate the distance of the obstacle
            distance = np.mean(roi_depth_frame[mask == 255])
            x, y, w, h = cv2.boundingRect(contour)
            # Append the obstacle coordinates, area, and distance to the list
            obstacles.append({
                'coordinates': (x + xmin, y + ymin, x + w + xmin, y + h + ymin),
                'area': int(cv2.contourArea(contour)),
                'distance': int(distance)
            })

    return obstacles

# # Load json file
# with open('../depth_frame_values.json', 'r') as file:
#     depth_frame = np.array(json.load(file))
# obstacles = obstacles_detect(depth_frame, [0, 0, 640, 480], 500, 10000, None, visual=True)
# print(obstacles)

