import cv2
import pyrealsense2 as rs
import numpy as np

# Create a pipeline
pipeline = rs.pipeline()


# Create a config object
config = rs.config()

# Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)

# Start streaming
profile = pipeline.start(config)

# Define a colormap for the heatmap (optional)
heatmap_color_map = cv2.COLORMAP_JET

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Get infrared frame
        ir_frame = frames.get_infrared_frame(1)

        # Convert infrared image to numpy array
        ir_image = np.asanyarray(ir_frame.get_data())

        cv2.imshow('Infrared Image', ir_image)

        # Apply colormap to the infrared image to create a heatmap
        heatmap = cv2.applyColorMap(ir_image, heatmap_color_map)

        # Display the heatmap
        cv2.imshow('Heatmap', heatmap)

        # Check for the 'q' key to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
