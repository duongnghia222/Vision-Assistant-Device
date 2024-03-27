import cv2
import numpy as np
import pyrealsense2 as rs


# Mouse callback function
def show_depth_value(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        depth = aligned_depth_frame.get_distance(x, y)
        print(f"Depth at pixel ({x}, {y}): {depth} meters")

height = 480
width = 640
mid = width // 2
# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, 30)
# Start streaming
pipeline.start(config)

base_height = 1500
threshold = 200

def apply_canny(image):
    # Convert color image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def apply_canny_for_infrared(image):
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    return edges

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        aligned_infrared_frame = aligned_frames.get_infrared_frame(1)
        # print(color_frame.get_frame_number())
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        infrared_frame = np.asanyarray(aligned_infrared_frame.get_data())
        color_image_copy = color_image.copy()
        edges = apply_canny(color_image)
        # Apply colormap for better visualizations
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # infrared_frame = cv2.applyColorMap(cv2.convertScaleAbs(infrared_frame, alpha=0.03), cv2.COLORMAP_JET)

        edges_infrared = apply_canny_for_infrared(infrared_frame)
        # Display the resulting frame
        # cv2.imshow('Disparity Map', depth_colormap)
        cv2.imshow('Infrared', infrared_frame)
        cv2.imshow('Original Color Image', color_image_copy)
        cv2.imshow('Canny Edges', edges)
        cv2.imshow('Canny Edges Infrared', edges_infrared)
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()

# Close all OpenCV windows
cv2.destroyAllWindows()
