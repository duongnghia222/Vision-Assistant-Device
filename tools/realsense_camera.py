#https://pysource.com
import pyrealsense2 as rs
import numpy as np
import cv2

class RealsenseCamera:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        # Configure depth and color streams
        print("Starting RealSense camera. Press 'q' to quit.")
        self.pipeline = rs.pipeline()
        config = rs.config()
        # width -> height
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, 30)

        # Start streaming
        self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)


    def get_frame_stream(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        # infrared_frame = aligned_frames.get_infrared_frame(1)
        frame_number = color_frame.get_frame_number()

        if not depth_frame or not color_frame:
            # If there is no frame, probably camera not connected, return False
            print("Error, impossible to get the frame, make sure that the Intel Realsense camera is correctly connected")
            return False, None, None, None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # infrared_frame = np.asanyarray(infrared_frame.get_data())

        return True, color_image, depth_image, frame_number


    def detect_covering(self, color_frame, depth_frame, visualize=False):
        # Extract ROI
        middle_x = self.width // 2
        middle_y = self.height // 2
        square_size = middle_x // 2
        roi_depth_frame = depth_frame[middle_y - square_size:middle_y + square_size, middle_x - square_size:middle_x + square_size]
        # Depth analysis
        depth_mask = roi_depth_frame < 30
        depth_percentage = np.mean(depth_mask.astype(float))  # Use float instead of np.float

        # Visualization (optional)
        if visualize:
            cv2.rectangle(color_frame, (middle_x - square_size, middle_y - square_size),
                          (middle_x + square_size, middle_y + square_size), (0, 0, 0), 1)

        # Determine if covering is detected
        if depth_percentage > 0.9:  # Adjust threshold as needed
            return True
        else:
            return False


    def release(self):
        self.pipeline.stop()
        print("Realsense Camera released")
        #print(depth_image)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.10), 2)

        # Stack both images horizontally

        #images = np.hstack((color_image, depth_colormap))



