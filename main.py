import pathlib
import torch
import os
import time
import cv2
import sys
import os.path as osp
from ultralytics import YOLOWorld
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



def run(yolo, voice):
    rs_camera = RealsenseCamera(width=1280, height=720) # This is max allowed
    print("Starting RealSense camera. Press 'q' to quit.")
    mode = 'finding'  # For debug, change to disabled after that
    detection = None
    object_to_find = None
    while True:
        ret, color_frame, depth_frame = rs_camera.get_frame_stream()
        if not ret:
            print("Error: Could not read frame.")
            break
        t1 = time.time()

        # Only change gestures if the current mode is disabled or a mode exit gesture is detected
        if mode == 'disabled':
            pass

        # Implement the functionalities for each mode
        if mode == 'finding':
            if not object_to_find:
                previous_text = None
                voice.speak("What you want to find")
                mic = pyaudio.PyAudio()

                stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
                stream.start_stream()
                while not object_to_find:
                    data = stream.read(4096, exception_on_overflow=False)
                    if recognizer.AcceptWaveform(data):
                        text = recognizer.Result()
                        text = text[14:-3]
                        # print(text)
                        if text == "ok" or text == "k" or text == "okay":
                            object_to_find = previous_text
                            break
                        voice.speak("You want to find {} !Say ok to confirm".format(text))
                        previous_text = text



                stream.stop_stream()
                stream.close()
                mic.terminate()
                print(object_to_find)







        cv2.imshow('RealSense Camera Detection', color_frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rs_camera.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    screen_width, screen_height = [720, 1280]
    yolo = YOLOWorld('yolov8m-world.pt')
    voice.speak("Please wait for system to start")
    run(yolo, voice)








