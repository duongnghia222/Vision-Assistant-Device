import pathlib
import torch
import os
import time
import cv2
import json
import sys
import os.path as osp
from tools.yolo_world import YoloWorld
from tools.classifier import Classifier
from tools.instruction import get_object_info, get_obstacle_info
from tools.virtual_assistant import VirtualAssistant
virtual_assistant = VirtualAssistant("tools/vosk-model-en-us-0.22-lgraph", None,
                                         words_per_minute=120, volume=0.9)
from tools.FPS import FPS
from tools.realsense_camera import *
from tools.custom_segmentation import segment_object
from tools.obstacles_detect import obstacles_detect
from ultis.draw import display_text
# from tools.finger_count import FingersCount
# from tools.tracker import Tracker


def run():
    # Load settings
    mode = settings.get('mode', 'BGF')  # For debug, change to disabled after that
    object_to_find = settings.get('object_to_find', 'door')
    screen_width = settings.get('screen_width', 640)
    screen_height = settings.get('screen_height', 480)
    fps_n_samples = settings.get('fps_n_samples', 50)
    is_visualize = settings.get('is_visualize', False)
    iou_threshold = settings.get('iou_threshold', 0.1)
    default_conf_threshold = settings.get('conf_threshold', 0.01)
    max_det = settings.get('max_det', 300)
    assistant_volume = settings.get('assistant_volume', 0.5)
    assistant_words_per_minute = settings.get('assistant_words_per_minute', 120)
    min_distance = settings.get('min_distance', 50)
    distance_threshold = settings.get('distance_threshold', 1000)
    size_threshold = settings.get('size_threshold', 15000)
    covering_duration_threshold = settings.get('covering_duration_threshold', 2)
    time_between_inform_obstacle = settings.get('time_between_inform_obstacle', 7)
    time_between_inform_obstacle_finding = settings.get('time_between_inform_obstacle_finding', 7)
    time_between_navigation = settings.get('time_between_navigation', 7)

    yolo.set_object_to_find([object_to_find])  # Delete after debug
    # virtual_assistant = VirtualAssistant("tools/vosk-model-en-us-0.22-lgraph", rs_camera,
    #                                      words_per_minute=assistant_words_per_minute, volume=assistant_volume)
    fps = FPS(nsamples=fps_n_samples)
    first_run = True
    # Runtime variable
    covering_detected = False
    covering_start_time = None

    # Timer variable
    last_navigate_to_object_time = time.time()
    last_inform_obstacle_location_time = time.time()

    while True:
        t1 = time.time()
        ret, color_frame, depth_frame, frame_number = rs_camera.get_frame_stream()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Only change gestures if the current mode is disabled or a mode exit gesture is detected
        if mode == 'disabled':
            pass

        if mode == 'off':
            os.system("sudo shutdown now")

        # Implement the functionalities for each mode
        if mode == 'finding':
            if not object_to_find:
                object_to_find = virtual_assistant.receive_object()
                if object_to_find:
                    yolo.set_object_to_find([object_to_find])
                print(object_to_find)
                continue
            print("finding")
            bbox, confidence = yolo.find_object(color_frame, default_conf_threshold, iou_threshold, max_det,
                                                is_visualize)
            print(bbox)
            if bbox:
                # pass through another classifer to make sure the object is the one we want
                # TODO: Implement classifier to classify the object
                object_mask, depth = segment_object(depth_frame, bbox)
                instruction, rotation_degrees, distance = get_object_info(bbox, depth, min_distance, color_frame,
                                                                          is_visualize)

                if time.time() - last_navigate_to_object_time >= time_between_navigation:
                    # Destroy cv2 window named 'RealSense Camera Detection':
                    # cv2.destroyWindow('RealSense Camera Detection')
                    # display_text(f"{instruction}, {rotation_degrees}, {distance}", "Instruction text",
                    # screen_width, screen_height)

                    virtual_assistant.navigate_to_object(instruction, rotation_degrees, distance)
                    # cv2.destroyWindow("Instruction text")
                    last_navigate_to_object_time = time.time()
                min_distance = 500
                obstacles = obstacles_detect(depth_frame, [screen_width // 4, 0,
                                                           screen_width - screen_width // 4, screen_height], min_distance,
                                             size_threshold, color_frame)
                direction, size, distance, obstacle_class, prob = get_obstacle_info(obstacles, classifier,
                                                                                    color_frame=color_frame,
                                                                                    visualize=is_visualize,
                                                                                    use_classifier=False)
                print(direction, size, distance, obstacle_class, prob)
                if direction and size and distance and time.time() - last_inform_obstacle_location_time >= time_between_inform_obstacle_finding:
                    virtual_assistant.inform_obstacle_location(direction, size, obstacle_class, prob)
                    last_inform_obstacle_location_time = time.time()

        if mode == "walking":
            obstacles = obstacles_detect(depth_frame, [screen_width // 4, 0, screen_width - screen_width // 4,
                                                       screen_height], distance_threshold,
                                         size_threshold, color_frame)
            direction, size, distance, obstacle_class, prob = get_obstacle_info(obstacles, classifier,
                                                                                color_frame=color_frame,
                                                                                visualize=is_visualize,
                                                                                use_classifier=False)
            if direction and size and distance and time.time() - last_inform_obstacle_location_time >= time_between_inform_obstacle:
                virtual_assistant.inform_obstacle_location(direction, size, obstacle_class, prob)
                last_inform_obstacle_location_time = time.time()
            print(direction, size, distance, obstacle_class, prob)

        if mode == "assistant":
            print("Waiting to clean up everything")
            time.sleep(2)
            object_to_find = None
            # command = virtual_assistant.hey_virtual_assistant()
            # print(command)
            # if Window name 'RealSense Camera Detection' is not null, destroy it
            
            # if cv2.getWindowProperty('RealSense Camera Detection', cv2.WND_PROP_VISIBLE) == 1:
            #     cv2.destroyWindow('RealSense Camera Detection')
            mode = virtual_assistant.hey_virtual_assistant(first_run)
            first_run = False
            fps.reset()
            print("Assistant mode")
            continue


        # Check for mode change
        if rs_camera.detect_covering(color_frame, depth_frame, visualize=True):
            if not covering_detected:
                covering_start_time = time.time()
                covering_detected = True
            else:
                covering_duration = time.time() - covering_start_time
                if covering_duration >= covering_duration_threshold:  # 2 seconds threshold
                    mode = 'assistant'
        else:
            covering_detected = False
        # FPS counter
        t2 = time.time()
        fps.update(1.0 / (t2 - t1))
        avg_fps = fps.accumulate()
        cv2.putText(color_frame, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        cv2.imshow('RealSense Camera Detection', color_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rs_camera.release()
    cv2.destroyAllWindows()
    fps.reset()
    virtual_assistant.close()


def load_settings(file_path):
    with open(file_path, 'r') as file:
        s = json.load(file)
    return s


def load_system():
    settings = load_settings('conf.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    yolo_world_path = settings.get('yolo_world_path', 'yolov8m-world.pt')
    classifier_path = settings.get('classifier_path', 'models/resnet-50')
    is_visualize = settings.get('is_visualize', True)
    screen_width = settings.get('screen_width', 640)
    screen_height = settings.get('screen_height', 480)
    yolo = YoloWorld(yolo_world_path)
    rs_camera = RealsenseCamera(width=screen_width, height=screen_height)
    classifier = Classifier(model_path=classifier_path, visualize=False)
    return yolo, classifier, rs_camera, settings


if __name__ == "__main__":
    # voice.speak("Please wait for system to start")
    yolo, classifier, rs_camera, settings = load_system()
    run()
