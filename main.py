import pathlib
import torch
import os
import time
import cv2
import sys
import pyaudio
import os.path as osp
from tools.yolo_world import YoloWorld
from tools.classifier import Classifier
from tools.finger_count import FingersCount
from tools.tracker import Tracker
from tools.instruction import navigate_to_object, inform_object_location
from tools.virtual_assistant import VirtualAssistant
virtual_assistant = VirtualAssistant("tools/vosk-model-en-us-0.22-lgraph")
from tools.FPS import FPS
from tools.realsense_camera import *
from tools.custom_segmentation import segment_object
from tools.obstacles_detect import obstacles_detect
iou_threshold = 0.1


def run(yolo, classifier):
    rs_camera = RealsenseCamera(width=640, height=480)  # This is max allowed
    print("Starting RealSense camera. Press 'q' to quit.")
    mode = 'BGF'  # For debug, change to disabled after that
    object_to_find = "bottle"
    yolo.set_object_to_find([object_to_find])
    fps = FPS(nsamples=50)

    while True:
        t1 = time.time()
        ret, color_frame, depth_frame, frame_number = rs_camera.get_frame_stream()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Only change gestures if the current mode is disabled or a mode exit gesture is detected
        if mode == 'disabled':
            pass

        # Implement the functionalities for each mode
        if mode == 'BGF':
            if not object_to_find:
                object_to_find = virtual_assistant.recognize_command("What do you want to find?", "find")
                if object_to_find:
                    yolo.set_object_to_find([object_to_find])
                print(object_to_find)
                continue
            print("finding")
            bbox, confidence = yolo.find_object(color_frame)
            print(bbox)
            if bbox:
                object_mask, depth = segment_object(depth_frame, bbox)
                instruction, degree = navigate_to_object(bbox, depth, 50, color_frame, voice=None, visualize=True)
                print(instruction, degree)

        if mode == "SSG":
            obstacles = obstacles_detect(depth_frame, [0, 0, screen_height, screen_width], 1000, 15000)
            direction, size = inform_object_location(obstacles, classifier, voice=None, color_frame=color_frame, visualize=True, use_classifier=True)
            print(direction, size)
            print(frame_number)

        # put text on the frame
        # FPS counter
        t2 = time.time()
        fps.update(1.0 / (t2 - t1))
        avg_fps = fps.accumulate()
        cv2.putText(color_frame, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('RealSense Camera Detection', color_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rs_camera.release()
    cv2.destroyAllWindows()
    fps.reset()



if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)
    screen_width, screen_height = [720, 1280]
    yolo = YoloWorld('yolov8m-world.pt')
    classifier = Classifier(model_path="models/resnet-50", visualize=False)
    # voice.speak("Please wait for system to start")
    run(yolo, classifier)








