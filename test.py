# count = 0
# for i in range(10000):
#     if 2**(i - 1) % 7 == 0:
#         count+=1
#         print(i)
# print(count)

import json
from tools.realsense_camera import RealsenseCamera
import cv2
import numpy as np

# rs_camera = RealsenseCamera(width=640, height=480) # This is max allowed
# print("Starting RealSense camera. Press 'q' to quit.")
# while True:
#     ret, color_frame, depth_frame = rs_camera.get_frame_stream()
#     if not ret:
#         print("Error: Could not read frame.")
#         break
#
#     # Display the resulting frame
#     cv2.imshow('Color Image', color_frame)
#     depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
#     cv2.imshow('Disparity Map', depth_colormap)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         # open json file to save depth_frame values
#         with open('depth_frame_values.json', 'w') as file:
#             json.dump(depth_frame.tolist(), file)
#         break


# Open json file to read depth_frame as numpy array
with open('depth_frame_values.json', 'r') as file:
    depth_frame = np.array(json.load(file))

# Viualize depth_frame values with disparity map
depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
while True:
    cv2.imshow('Disparity Map', depth_colormap)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


