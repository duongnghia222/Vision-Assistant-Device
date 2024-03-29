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
#     ret, color_frame, depth_frame, infrared_frame, frame_number = rs_camera.get_frame_stream()
#     if not ret:
#         print("Error: Could not read frame.")
#         break
#
#     # Display the resulting frame
#     cv2.imshow('Color Image', color_frame)
#     # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
#     # cv2.imshow('Disparity Map', depth_colormap)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         # open json file to save color frame
#         with open('infrared_frame_values.json', 'w') as file:
#             json.dump(infrared_frame.tolist(), file)
#         break


# # Open json file to read depth_frame as numpy array
# with open('depth_frame_values.json', 'r') as file:
#     depth_frame = np.array(json.load(file))
#
# # Viualize depth_frame values with disparity map
# depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
# while True:
#     cv2.imshow('Disparity Map', depth_colormap)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# import nltk
# import os
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
#
# def download_nltk_resources():
#     nltk.data.path.append("./nltk_data")
#     if not os.path.exists("./nltk_data/corpora/stopwords"):
#         nltk.download('stopwords', download_dir="./nltk_data")
#     if not os.path.exists("./nltk_data/tokenizers/punkt"):
#         nltk.download('punkt', download_dir="./nltk_data")
#
# def remove_stopwords(text):
#     words = word_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#     filtered_words = [word for word in words if word.lower() not in stop_words]
#     filtered_text = ' '.join(filtered_words)
#     return filtered_text
#
# # Example text
# text = "ok okay k"
#
# # Download NLTK resources
# download_nltk_resources()
#
# # Remove stopwords from the example text
# filtered_text = remove_stopwords(text)
#
# # Print the filtered text
# print("Original Text:", text)
# print("Filtered Text:", filtered_text)
#
# from ultralytics import YOLO
#
# # Load a model
# model = YOLO('yolov8x-cls.pt')  # load an official model
#
# # Predict with the model
# results = model('pictures/1.png', conf=0.1)  # predict on an image
# for result in results:
#     print(result)


import numpy as np
import cv2

def detect_covering(color_frame, depth_frame, visualize=False):
    # Extract ROI
    middle_x = 1280 // 2
    middle_y = 720 // 2
    square_size = middle_x // 2
    roi_depth_frame = depth_frame[middle_y - square_size:middle_y + square_size,
                      middle_x - square_size:middle_x + square_size]

    covered_pixels = np.count_nonzero(roi_depth_frame < 30)
    # print(roi_depth_frame < 100)

    # Calculate the total number of pixels in the ROI
    total_pixels = roi_depth_frame.size
    # Calculate the percentage of covered pixels
    coverage_percentage = covered_pixels / total_pixels
    print(coverage_percentage)
    # Check if the coverage percentage exceeds the threshold
    if visualize:
        cv2.rectangle(color_frame, (middle_x - square_size, middle_y - square_size),
                      (middle_x + square_size, middle_y + square_size), (0, 0, 0), 2)
    if coverage_percentage > 0.9:
        return True
    else:
        return False

# Assuming you have already initialized the RealSenseCamera instance
rs_camera = RealsenseCamera(width=1280, height=720)

# Main loop
while True:
    ret, color_frame, depth_frame, frame_number = rs_camera.get_frame_stream()
    if ret:
        # Detect covering
        if detect_covering(color_frame, depth_frame, visualize=True):
            print("Something is covering your camera lens!")

        # Display frames (optional)
        cv2.imshow("Color Frame", color_frame)
        cv2.imshow("Depth Frame", depth_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera
rs_camera.release()
cv2.destroyAllWindows()




