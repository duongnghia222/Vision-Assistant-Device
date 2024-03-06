import pyttsx3
import cv2


def navigate_to_object(bbox, depth, color_frame, min_dis=50):
    """
    Navigates the blind user towards the object using audio instructions based on bounding box and depth information.

    Parameters:
    bbox (list): A list containing the bounding box coordinates [xmin, ymin, xmax, ymax].
    depth (float): The depth of the object from the camera. (unit: mm)

    Returns:
    str: Navigation instruction ('turn left', 'turn right', 'move forward', or 'stop').
    """
    xmin, ymin, xmax, ymax = bbox
    box_center_x = int((xmin + xmax) / 2)

    print('depth', depth)
    # Adjust threshold based on depth
    middle_x = color_frame.shape[1] // 2
    scale = 80000
    middle_diff = (1/(depth+1))*scale

    if middle_diff > 250:
        middle_diff = 250
    if middle_diff < 70:
        middle_diff = 70
    left_bound = int(max(min(middle_x - middle_diff, color_frame.shape[1]), 0))
    right_bound = int(max(min(middle_x + middle_diff, color_frame.shape[1]), 0))
    print(middle_diff)
    cv2.line(color_frame, (left_bound, 0), (left_bound, color_frame.shape[0]), (0, 255, 0), 2)  # Left line
    cv2.line(color_frame, (right_bound, 0), (right_bound, color_frame.shape[0]), (0, 255, 0), 2)  # Right line

    # Determine the direction to move
    if box_center_x < middle_x - middle_diff:
        direction = "turn left"
    elif box_center_x > middle_x + middle_diff:
        direction = "turn right"
    else:
        direction = "move forward"

    # Incorporate depth information for distance
    if depth < min_dis:
        instruction = "stop"
    else:
        instruction = f"{direction}"

    return instruction
