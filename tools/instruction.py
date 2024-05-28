# Description: This file contains the function to navigate the blind user towards the object using audio instructions based on bounding box and depth information.
import cv2


def get_obstacle_info(obstacles, classifier, color_frame, visualize=False, use_classifier=True):
    # If there is no obstacle detected return None
    direction, size, distance, obstacle_class, prob = None, None, None, None, None
    if len(obstacles) == 0:
        return direction, size, distance, obstacle_class, prob
    # Sort obstacles based on distance
    obstacles = sorted(obstacles, key=lambda x: x['distance'])
    obstacle = obstacles[0]

    x1, y1, x2, y2 = obstacle['coordinates']
    distance = obstacle['distance']
    area = obstacle['area']
    obstacle_center_x = (x1 + x2) // 2
    if use_classifier:
        # Classify the obstacle
        obstacle_frame = color_frame[y1:y2, x1:x2]
        obstacle_class, prob = classifier.predict(obstacle_frame)
    pixel_displacement = abs(color_frame.shape[1] / 2 - obstacle_center_x)
    degrees_per_pixel = 69 / color_frame.shape[1]  # Horizontal field of view of the camera is 69 degrees
    degree = int(pixel_displacement * degrees_per_pixel)

    # Determine the direction of the obstacle
    if degree < 10:
        direction = "center"
    elif 10 <= degree < 30 and color_frame.shape[1] / 2 - obstacle_center_x > 0:
        direction = "slightly left"
    elif 10 <= degree < 30 and color_frame.shape[1] / 2 - obstacle_center_x < 0:
        direction = "slightly right"
    elif degree > 30 and color_frame.shape[1] / 2 - obstacle_center_x > 0:
        direction = "left"
    elif degree > 30 and color_frame.shape[1] / 2 - obstacle_center_x < 0:
        direction = "right"

    # Determine the size of the obstacle
    if area < 20000:
        size = "small"
    elif 20000 <= area < 40000:
        size = "medium"
    elif 40000 <= area < 80000:
        size = "large"
    else:
        size = "very large"

    # draw obstacles on depth frame
    if visualize:
        skip = False
        for obstacle in obstacles:
            if skip:
                break
            cv2.rectangle(color_frame, (obstacle['coordinates'][0], obstacle['coordinates'][1]),
                          (obstacle['coordinates'][2], obstacle['coordinates'][3]), (0, 0, 255), 2)
            skip = True
    return direction, size, distance, obstacle_class, prob


def get_object_info(bbox, distance, min_dis, color_frame, visualize=False):
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
    # Adjust threshold based on depth
    middle_x = color_frame.shape[1] // 2
    if distance > 1000:
        middle_diff = 70
    elif distance < 1000:
        middle_diff = 100
    elif 500 < distance < 1000:
        middle_diff = 150
    else:
        middle_diff = 200

    left_bound = int(max(min(middle_x - middle_diff, color_frame.shape[1]), 0))
    right_bound = int(max(min(middle_x + middle_diff, color_frame.shape[1]), 0))

    # Determine the direction to move
    if box_center_x < middle_x - middle_diff:
        direction = "left"
    elif box_center_x > middle_x + middle_diff:
        direction = "right"
    else:
        direction = "straight"

    # Incorporate depth information for distance
    if distance < min_dis:
        instruction = "stop"
    else:
        instruction = f"{direction}"

    # Calculate pixel displacement from the center of the image
    pixel_displacement = abs(color_frame.shape[1] / 2 - box_center_x)

    # Calculate degrees per pixel
    degrees_per_pixel = 69 / color_frame.shape[1]  # Horizontal field of view of the camera is 69 degrees

    # Calculate the number of degrees of rotation required
    rotation_degrees = int(pixel_displacement * degrees_per_pixel)

    if visualize:
        cv2.line(color_frame, (left_bound, 0), (left_bound, color_frame.shape[0]), (0, 255, 0), 2)  # Left line
        cv2.line(color_frame, (right_bound, 0), (right_bound, color_frame.shape[0]), (0, 255, 0), 2)  # Right line
    return instruction, rotation_degrees, distance
