import pathlib
import torch
import os
import time
import cv2
import threading
import sys
import os.path as osp
from tools.finger_count import FingersCount
from tools.tracker import Tracker
from tools.instruction import navigate_to_object
from tools.voice_navigator import TextToSpeech
voice = TextToSpeech()
from vosk import Model, KaldiRecognizer
import pyaudio
model = Model(r"tools/vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(model, 16000)
from tools.realsense_camera import *
from tools.custom_segmentation import segment_object
from tools.custom_inferer import Inferer, CalcFPS
from yolov6.utils.events import load_yaml


ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def run(fc, yolo, custom_model, voice):
    rs_camera = RealsenseCamera(width=1280, height=720) # This is max allowed
    # rs_camera = RealsenseCamera()
    fps = CalcFPS()
    # hand_cascade = cv2.CascadeClassifier('hand.xml')

    # webcam = cv2.VideoCapture("C:\\Users\\nghia\\Desktop\\WIN_20240112_09_12_34_Pro.mp4")
    # webcam = cv2.VideoCapture(0)
    print("Starting RealSense camera detection. Press 'q' to quit.")
    mode = 'disabled'  # For debug, change to disabled after that
    last_gesture = None
    gesture_start = 0.1
    detection = None
    last_finder_call_time = None
    # object_to_find = {"name": "bottle", "conf_threshold": 0.5} # for debug, change to None after that
    object_to_find = None
    finger_counts = None
    # depth_frame = 0
    while True:
        ret, color_frame, depth_frame = rs_camera.get_frame_stream()
        # ret, color_frame = webcam.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        t1 = time.time()
        # gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        # hands = hand_cascade.detectMultiScale(gray, 1.1, 4)
        # for (x, y, w, h) in hands:
        #     cv2.rectangle(color_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # print(t1)
        finger_counts = fc.infer(color_frame)

        # Only change gestures if the current mode is disabled or a mode exit gesture is detected
        if mode == 'disabled' or finger_counts in [[0, 0], [0, 5]]:
            if finger_counts != last_gesture:
                last_gesture = finger_counts
                gesture_start = time.time()

            # Check if the gesture is held for 2 seconds
            if time.time() - gesture_start >= 2:
                if finger_counts == [0, 0]:
                    mode = 'disabled'
                    object_to_find = None
                    fps.reset()
                    voice.speak("All modes disabled.")
                elif finger_counts == [0, 1]:
                    mode = 'finding'
                    object_to_find = None
                    fps.reset()
                    voice.speak("Finding mode activated.")
                elif finger_counts == [0, 2]:
                    mode = 'detecting'
                    object_to_find = None
                    fps.reset()
                    voice.speak("Detecting mode activated.")
                elif finger_counts == [0, 5]:
                    print("Program stopping...")
                    break

        # Implement the functionalities for each mode
        if mode == 'finding':
            # if not object_to_find:
            #     voice.speak("What you want to find")
            #     mic = pyaudio.PyAudio()
            #
            #     stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
            #     stream.start_stream()
            #     # voice.speak("What you want to find")
            #     while not object_to_find:
            #         data = stream.read(4096)
            #         if recognizer.AcceptWaveform(data):
            #             text = recognizer.Result()
            #             text = text[14:-3]
            #             if object_dictionary(text) is not None:
            #                 object_to_find = object_dictionary(text)
            #                 print(object_to_find)
            #                 stream.stop_stream()
            #                 stream.close()
            #                 mic.terminate()




            # Implement finding functionality
            if finger_counts != last_gesture:
                last_gesture = finger_counts
                gesture_start = time.time()
            elif time.time() - gesture_start >= 2 and not object_to_find:
                # print(finger_counts)
                # print(finger_counts_mapping_obj(finger_counts))
                object_to_find = finger_counts_mapping_obj(finger_counts)
                if object_to_find:
                    voice.speak(f"Looking for: {object_to_find['name']}")
            if object_to_find:
                if last_finder_call_time is None:
                    last_finder_call_time = time.time()
                object_index = yolo.class_names.index(object_to_find["name"])

                conf_threshold = object_to_find["conf_threshold"]


                if detection is None or (time.time() - last_finder_call_time >= 0):
                    last_finder_call_time = time.time()
                    detection = yolo.object_finder(color_frame, object_index, predict_threshold=conf_threshold)
                    if detection is not None:
                        if len(detection) > 1:
                            detection = detection[0]
                        detection = detection.flatten()

                if detection is not None and len(detection):
                    *xyxy, conf, cls = detection
                    #[285, 194, 394, 298]
                    xmin, ymin, xmax, ymax = map(int, xyxy)  # Convert each element to an integer
                    object_mask, depth = segment_object(depth_frame, [xmin, ymin, xmax, ymax])

                    # cv2.imshow("Object Mask", object_mask)
                    # color_roi = color_frame[ymin:ymax, xmin:xmax]
                    # _, binary_mask = cv2.threshold(object_mask, 127, 255, cv2.THRESH_BINARY)
                    #
                    # isolated_object = cv2.bitwise_and(color_roi, color_roi, mask=binary_mask)
                    # color_image_with_object = color_frame.copy()
                    # color_image_with_object[ymin:ymax, xmin:xmax] = isolated_object
                    # cv2.imshow("Color Image with Object", color_image_with_object)
                    #
                    #
                    # yolo.plot_box_and_label(color_frame, max(round(sum(color_frame.shape) / 2 * 0.003), 2), xyxy,\
                    #                         depth, label='Distance', color=(128, 128, 128), txt_color=(255, 255, 255),\
                    #                         font=cv2.FONT_HERSHEY_COMPLEX)
                #     print("distance", depth)
                #
                    instruction = navigate_to_object([xmin, ymin, xmax, ymax], depth, color_frame)
                    print(instruction)
                    # voice.speak(instruction)

        elif mode == 'detecting':
            # Implement detecting functionality
            dangerous_obj = custom_model.dangerous_object_detection(color_frame, conf_threshold=0.5)
            if dangerous_obj is not None:
                if len(dangerous_obj) > 1:
                    dangerous_obj = dangerous_obj[0]
                dangerous_obj = dangerous_obj.flatten()
            if dangerous_obj is not None and len(dangerous_obj):
                *xyxy, conf, cls = dangerous_obj
                if isinstance(cls, torch.Tensor):
                    if cls.nelement() == 1:
                        cls = int(cls.item())


                print(DANGEROUS_CLASS_NAMES[cls])
                # [285, 194, 394, 298]
                xmin, ymin, xmax, ymax = map(int, xyxy)  # Convert each element to an integer
                object_mask, depth = segment_object(depth_frame, [xmin, ymin, xmax, ymax])

                custom_model.plot_box_and_label(color_frame, max(round(sum(color_frame.shape) / 2 * 0.003), 2), xyxy,\
                                        depth, label='Distance', color=(128, 128, 128), txt_color=(255, 255, 255),\
                                        font=cv2.FONT_HERSHEY_COMPLEX)
                instruction = navigate_to_object([xmin, ymin, xmax, ymax], depth, color_frame, 50)
                if instruction == "move forward":
                    instruction = "front"
                elif instruction == "turn left":
                    instruction = "right"
                elif instruction == "turn right":
                    instruction = "left"
                elif instruction == "stop":
                    instruction = "very close"
                guide = custom_model.class_names[cls] + "on the" + instruction + str(depth) + "centimeters away"
                # voice.speak(guide)
        t2 = time.time()
        if t2 - t1 > 1e-9:
            frame_fps = 1.0 / (t2 - t1)
        else:
            frame_fps = 0
        if frame_fps != 0:
            fps.update(frame_fps)
        avg_fps = fps.accumulate()
        yolo.draw_text(
            color_frame,
            f"FPS: {frame_fps:0.1f}",
            pos=(0, 0),
            font_scale=1.0,
            text_color=(204, 85, 17),
            text_color_bg=(255, 255, 255),
            font_thickness=2,
        )

        yolo.draw_text(
            color_frame,
            f"AVG FPS: {avg_fps:0.1f}",
            pos=(0, 30),
            font_scale=1.0,
            text_color=(204, 85, 17),
            text_color_bg=(255, 255, 255),
            font_thickness=2,
        )
        cv2.imshow('RealSense Camera Detection', color_frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rs_camera.release()
    # webcam.release()
    cv2.destroyAllWindows()


def object_dictionary(text):
    if text in yolo.class_names:
        return {"name": text, "conf_threshold": 0.4}
    return None


def finger_counts_mapping_obj(object_code):
    if object_code == [1, 0]:
        return {"name": "bottle", "conf_threshold": 0.4}
    if object_code == [1, 1]:
        return {"name": "cup", "conf_threshold": 0.8}


def create_inferer(weights=osp.join(ROOT, 'yolov6s_mbla.pt'),
        yaml='data/coco.yaml',
        img_size=[640,640],
        conf_threshold=0.4,
        iou_threshold=0.45,
        max_det=1000,
        device='0',
        save_txt=False,
        not_save_img=True,
        save_dir=None,
        view_img=True,
        classes=None,
        agnostic_nms=False,
        project='runs/infereqnce',
        name='exp',
        hide_labels=False,
        hide_conf=False,
        half=False):
    weights = osp.join(os.getcwd(), weights)
    infer = Inferer(weights, device, yaml, img_size, half, conf_threshold, iou_threshold, agnostic_nms, max_det)
    return infer


if __name__ == "__main__":
    PATH_YOLOv6 = pathlib.Path(__file__).parent
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CLASS_NAMES = load_yaml(str(PATH_YOLOv6 / "data/coco.yaml"))['names']
    DANGEROUS_CLASS_NAMES = load_yaml(str(PATH_YOLOv6 / "data/dangerous_obj.yaml"))['names']

    # Load the YOLOv6 model (choose the appropriate function based on the model size you want to use)\
    screen_width, screen_height = [720, 1280]
    fc = FingersCount(screen_width, screen_height)
    yolo = create_inferer()
    custom_model = create_inferer(weights='dangerous_obj.pt', yaml='data/dangerous_obj.yaml')
    voice.speak("Please wait for system to start")
    run(fc, yolo, custom_model, voice)








