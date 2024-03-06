import cv2
import numpy as np
import pyrealsense2 as rs

# Mouse callback function
def show_depth_value(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        depth = depth_frame.get_distance(x, y)
        print(f"Depth at pixel ({x}, {y}): {depth:.2f} meters")

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

cv2.namedWindow('Disparity Map')
cv2.setMouseCallback('Disparity Map', show_depth_value)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Apply colormap for better visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Display the resulting frame
        cv2.imshow('Disparity Map', depth_colormap)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()

# Close all OpenCV windows
cv2.destroyAllWindows()
