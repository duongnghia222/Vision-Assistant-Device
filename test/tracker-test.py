import argparse
import numpy as np
import cv2
import supervision as sv
from collections import defaultdict
from ultralytics import YOLOWorld
from realsense_camera import *

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Camera args')
    # parser.add_argument('--webcam', action='store_true', help='whether to use webcam.')
    parser.add_argument('--webcam-addr', type=int, default=0, help='the web camera address, local camera or rtsp address.')
    parser.add_argument("--webcam-res", default=[1280, 720], nargs=2, type=int, help='the web camera resolution')
    # parser.add_argument("--source", type=str, default='', help='the source path of a single image / video')
    parser.add_argument("--classes", default=None, nargs='*', type=str, help='the list of classes (things) to be detected')
    parser.add_argument("--model", default='yolov8l-world.pt', type=str)
    args = parser.parse_args()
    return args

class CameraProcessor:
    def __init__(self, webcam_addr, classes, model, webcam_res, conf_threshold=0.25, iou_threshold=0.1):
        self.model = YOLOWorld(model=model)
        if classes is None:
            self.classes = self.model.names
        else:
            self.classes = classes
            self.model.set_classes(self.classes)
        
        # self.tracker = sv.ByteTrack()
        self.track_history = defaultdict(lambda: [])
        
        self.webcam_res = webcam_res
        width, height = self.webcam_res
        self.webcam_addr = webcam_addr
        if self.webcam_addr == 2:
            self.capture = RealsenseCamera(width=width, height=height)
        else:
            self.capture = cv2.VideoCapture(self.webcam_addr)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        self.bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)
        
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.interval = 5000
    
    def process_camera(self):
        while True:
            if self.webcam_addr == 2:
                ret, frame, depth_frame = self.capture.get_frame_stream()
            else:
                ret, frame = self.capture.read()
            
            
            processed_frame = self.process_frame2(frame)
            
            cv2.imshow("frame", processed_frame)
            # if cv2.waitKey(30) == 27:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        cv2.destroyAllWindows()
        
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model.predict(frame, conf=self.conf_threshold)
        detections = sv.Detections.from_ultralytics(results[0]).with_nms(threshold=self.iou_threshold)
        detections = self.tracker.update_with_detections(detections)
        
        annotated_frame = self.annotate_frame(frame, detections)
        return annotated_frame
    
    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections) ->np.ndarray:
        annotated_frame = frame.copy()
        annotated_frame = self.bounding_box_annotator.annotate(annotated_frame, detections=detections)
        
        labels = [
            f"#{tracker_id} {self.classes[class_id]}: {confidence:0.3f}"
            for tracker_id, class_id, confidence
            in zip(detections.tracker_id, detections.class_id, detections.confidence)
        ]
        annotated_frame = self.label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
        return annotated_frame    
    
    def process_frame2(self, frame: np.ndarray) -> np.ndarray:
        # results = self.model.predict(frame, conf=self.conf_threshold)
        results = self.model.track(frame, persist=True, conf=self.conf_threshold)
        annotated_frame = results[0].plot()
        
        # boxes = results[0].boxes.xywh.cpu()
        # track_ids = results[0].boxes.id.int().cpu().tolist()
        
        # for box, track_id in zip(boxes, track_ids):
        #     x, y, w, h = box
        #     track = self.track_history[track_id]
        #     track.append((float(x), float(y)))  # x, y center point
        #     if len(track) > 30:  # retain 90 tracks for 90 frames
        #         track.pop(0)

        #     # Draw the tracking lines
        #     points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        #     cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        return annotated_frame

        

if __name__ == "__main__":
    args = parse_arguments()

    processor = CameraProcessor(args.webcam_addr, args.classes, args.model, args.webcam_res)
    processor.process_camera()