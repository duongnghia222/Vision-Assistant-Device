import cv2
import numpy as np
import json


def display_text(text="Sample text", window_name="Text", screen_width=640, screen_height=480):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 2
    text_color = (0, 255, 0)
    text_color_bg = (0, 0, 0)
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    img[:] = text_color_bg
    # put text on the center of the image:
    x = (screen_width - text_w) // 2
    y = (screen_height + text_h) // 2
    cv2.putText(img, text, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.imshow(window_name, img)
    cv2.waitKey(1)


