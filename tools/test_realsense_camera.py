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

cv2.namedWindow('Disparity Map')
cv2.namedWindow('Color Image with Black Pixels')
cv2.namedWindow('Original Color Image')
cv2.setMouseCallback('Disparity Map', show_depth_value)
cv2.setMouseCallback('Color Image with Black Pixels', show_depth_value)
cv2.setMouseCallback('Original Color Image', show_depth_value)
base_height = 1500
threshold = 200
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

        # Apply colormap for better visualizations
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        infrared_frame = cv2.applyColorMap(cv2.convertScaleAbs(infrared_frame, alpha=0.03), cv2.COLORMAP_JET)

        # Show images
        depth_range_mask = np.logical_and(depth_image >= base_height, depth_image <= base_height + threshold)


        # Apply mask only to the bottom half of the image
        bottom_half_mask = np.zeros_like(depth_range_mask)
        bottom_half_mask[height // 2:, mid - threshold: mid + threshold] = depth_range_mask[height // 2:,\
                                                                           mid - threshold:mid + threshold]

        color_image[bottom_half_mask] = [0, 0, 0]

        # Display the resulting frame
        cv2.imshow('Disparity Map', depth_colormap)
        cv2.imshow('Infrared', infrared_frame)
        cv2.imshow('Color Image with Black Pixels', color_image)
        cv2.imshow('Original Color Image', color_image_copy)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()

# Close all OpenCV windows
cv2.destroyAllWindows()
