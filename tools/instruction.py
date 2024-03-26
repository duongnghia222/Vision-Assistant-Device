# Description: This file contains the function to navigate the blind user towards the object using audio instructions based on bounding box and depth information.
import cv2


def inform_object_location(obstacles, voice, color_frame, visual=False):
    # Sort obstacles based on distance
    obstacles = sorted(obstacles, key=lambda x: x['distance'], reverse=True)
    for obstacle in obstacles:
        x1, y1, x2, y2 = obstacle['coordinates']
        area = obstacle['area']
        distance = obstacle['distance']
        print("Area:", area, "Distance:", distance)
    if voice:
        voice.speak(f"Object at {obstacles[0]['distance']} millimeters")
    direction = ""
    degree = 0
    # draw obstacles on depth frame
    if visual:
        for obstacle in obstacles:
            x1, y1, x2, y2 = obstacle['coordinates']
            cv2.rectangle(color_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return direction, degree


def navigate_to_object(bbox, depth, min_dis, color_frame, voice, visual=False):
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
    if depth > 1000:
        middle_diff = 70
    elif depth < 1000:
        middle_diff = 100
    elif 500 < depth < 1000:
        middle_diff = 150
    else:
        middle_diff = 200

    left_bound = int(max(min(middle_x - middle_diff, color_frame.shape[1]), 0))
    right_bound = int(max(min(middle_x + middle_diff, color_frame.shape[1]), 0))
    if visual:
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

    # Calculate pixel displacement from the center of the image
    pixel_displacement = abs(color_frame.shape[1] / 2 - box_center_x)

    # Calculate degrees per pixel
    degrees_per_pixel = 69 / color_frame.shape[1]  # Horizontal field of view of the camera is 69 degrees

    # Calculate the number of degrees of rotation required
    rotation_degrees = int(pixel_displacement * degrees_per_pixel)
    if voice:
        if instruction == "stop":
            voice.speak(instruction)
        elif direction == "move forward":
            voice.speak(instruction + " " + str(int(depth / 100)) + " meters away")
        else:
            voice.speak(instruction + " " + str(rotation_degrees) + " degrees")
    return instruction, rotation_degrees
