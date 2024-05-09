import pyrealsense2 as rs
import cv2
import numpy as np

# Initialize camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

depth_frame = None

def mouse_event_handler(event, x, y, flags, param):
    global depth_frame
    if event == cv2.EVENT_MOUSEMOVE and depth_frame:
        # Get depth value at mouse position
        depth = depth_frame.get_distance(x, y)
        distance = f"{depth:.2f}"
        position = f"({x}, {y})"
        param['text'] = distance + ", " + position

cv2.namedWindow('Color Frame')
color_image = None
text_info = {'text': ''}

cv2.setMouseCallback('Color Frame', mouse_event_handler, text_info)

while True:
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Display distance info on color image
    cv2.putText(color_image, text_info['text'], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display images
    cv2.imshow('Color Frame', color_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
