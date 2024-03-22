from tools.voice_navigator import TextToSpeech
SPEAKER = TextToSpeech()
import argparse
import numpy as np
import cv2
import supervision as sv
import time
from collections import defaultdict
from ultralytics import YOLOWorld
from tools.realsense_camera import *


CENTER_ZONE_POLYGON = np.array([
    [0.3, 0],
    [0.7, 0],
    [0.7, 1],
    [0.3, 1]
])
RIGHT_ZONE_POLYGON = np.array([
    [0.0, 0],
    [0.3, 0],
    [0.3, 1],
    [0.0, 1]
])
LEFT_ZONE_POLYGON = np.array([
    [0.7, 0],
    [1.0, 0],
    [1.0, 1],
    [0.7, 1]
])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Camera args')
    parser.add_argument('--webcam-addr', type=int, default=0, help='the web camera address, local camera or rtsp address.')
    parser.add_argument("--webcam-res", default=[1280, 720], nargs=2, type=int, help='the web camera resolution')
    parser.add_argument("--classes", default=None, nargs='*', type=str, help='the list of classes (things) to be detected')
    parser.add_argument("--model", default='yolov8l-worldv2.pt', type=str)
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--iou", default=0.45, type=float)
    parser.add_argument("--visualize", default=False, type=bool)
    args = parser.parse_args()
    return args

class ObjectFindingProcessor:
    def __init__(self, webcam_addr, classes, model, webcam_res, conf=0.003, iou=0.1, visualize=True):
        """Initializes an ObjectFindingProcessor instance.

        Args:
            webcam_addr: The web camera address, either a local camera or an RTSP address.
            classes: The list of classes (things) to be detected.
            model: The YOLO model file.
            webcam_res: The resolution of the web camera.
            conf: The confidence threshold for object detection.
            iou: The intersection over union threshold for non-maximum supression.
            visualize: Flag indicating whether to visualize the processed frames (default: True).

        """
        self.model = YOLOWorld(model=model)
        if classes is None:
            self.classes = self.model.names
        else:
            self.classes = classes
            self.model.set_classes(self.classes)
        
        # self.tracker = sv.ByteTrack()
        # self.track_history = defaultdict(lambda: [])
        
        self.webcam_res = webcam_res
        width, height = self.webcam_res
        self.webcam_addr = webcam_addr
        if self.webcam_addr == 2:
            self.capture = RealsenseCamera(width=width, height=height)
        else:
            self.capture = cv2.VideoCapture(self.webcam_addr)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.conf = conf
        self.iou = iou
        
        self.visualize = visualize
        self.bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)
        
        self.zone = {
            'center': sv.PolygonZone(polygon=(CENTER_ZONE_POLYGON * np.array(self.webcam_res)).astype(int), frame_resolution_wh=tuple(self.webcam_res)),
            # 'left': sv.PolygonZone(polygon=(LEFT_ZONE_POLYGON * np.array(self.webcam_res)).astype(int), frame_resolution_wh=tuple(self.webcam_res)),
            # 'right': sv.PolygonZone(polygon=(RIGHT_ZONE_POLYGON * np.array(self.webcam_res)).astype(int), frame_resolution_wh=tuple(self.webcam_res))
        }
        
        self.zone_annotator = {
            'center': sv.PolygonZoneAnnotator(zone = self.zone['center'], color=sv.Color.RED, thickness=2, text_thickness=4, text_scale=2),
            # 'left': sv.PolygonZoneAnnotator(zone = self.zone['left'], color=sv.Color.RED, thickness=2, text_thickness=4, text_scale=2),
            # 'right': sv.PolygonZoneAnnotator(zone = self.zone['right'], color=sv.Color.RED, thickness=2, text_thickness=4, text_scale=2),
        }
        
        self.speaker = SPEAKER
        self.prev_speak_time = time.time()
        self.speak_time_interval = 5
    
    def process_camera(self):
        """Processes the camera frames and displays the annotated frames."""
        while True:
            prev_frame_time = time.time()
            
            if self.webcam_addr == 2:
                ret, frame, depth_frame = self.capture.get_frame_stream()
            else:
                ret, frame = self.capture.read()
            
            processed_frame = self.process_frame(frame)
            
            if self.visualize:
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time) 
                fps = int(fps) 
                fps = str(fps) 
                cv2.putText(processed_frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
                
                cv2.imshow("frame", processed_frame)
            
            # if cv2.waitKey(30) == 27:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        cv2.destroyAllWindows()
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Processes a single frame and returns the annotated frame.

        Args:
            frame: The input frame to be processed.

        Returns:
            np.ndarray: The annotated frame with added bounding boxes, labels, and annotations.

        """
        results = self.model.predict(frame, conf=self.conf, iou=self.iou, agnostic_nms=True) 
        # results = self.model.track(frame, persist=True, conf=self.conf_threshold, agnostic_nms=True)
        if self.visualize:
            annotated_frame = results[0].plot()
        boxes = results[0].boxes.xywh.cpu()
        confs = results[0].boxes.conf.cpu()
        # track_ids = results[0].boxes.id.int().cpu().tolist()

        def get_direction(boxes, confs):
            """Determines the direction based on the detected object's bounding box coordinates.

            Args:
                boxes (np.ndarray): A numpy array of shape `(N, 4)` where each
                row corresponds to a bounding box in the format (x, y, w, h).
                confs (np.ndarray): An array of confidence scores for the detected objects.

            Returns:
                str: The direction based on the object's position relative to the defined zone.
                    Possible values: 'Straight', 'Left', 'Right'.

            """
            idx = np.argmax(confs)
            box = boxes[idx]
            x, y, w, h = box    
            x1_zone = self.zone['center'].polygon[0, 0]
            x2_zone = self.zone['center'].polygon[3, 1]
            if (x > x1_zone) and (x < x2_zone):
                direction = 'Straight'
            elif x <= x1_zone:
                direction = 'Left'
            else:
                direction ='Right' 
            return direction
        
        if len(confs):
            direction = get_direction(boxes, confs)
            if self.visualize:
                cv2.putText(annotated_frame, direction, (7, int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) - 7)), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA) 
        
            if time.time() - self.prev_speak_time > self.speak_time_interval:
                self.speaker.speak(direction)
                self.prev_speak_time = time.time()
        
        return annotated_frame
    
        

if __name__ == "__main__":
    args = parse_arguments()

    processor = ObjectFindingProcessor(args.webcam_addr, args.classes, args.model, args.webcam_res)
    processor.process_camera()