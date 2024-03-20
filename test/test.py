import cv2
import supervision as sv
# import inference
# from inference.models import YOLOWorld
from ultralytics import YOLOWorld, YOLO
import numpy as np
import time
import argparse

IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
VID_FORMATS = ["mp4", "mov", "avi", "mkv"]
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])
VID_FORMATS.extend([f.upper() for f in VID_FORMATS])

def check_extension(path):
    if path.split('.')[-1].lower() in IMG_FORMATS:
        file_type = 'image'
    elif path.split('.')[-1].lower() in VID_FORMATS:
        file_type = 'video'
    else:
        file_type = ''
    return file_type

def get_output_name(path):
    file = path.split('/')[-1]
    return file.split('.')[0] + '-output.' + file.split('.')[-1]

def find_largest_conf_index(confs):
    max_conf = confs[0]
    max_index = 0
    
    for i in range(1, len(confs)):
        if confs[i] > max_conf:
            max_conf = confs[i]
            max_index = i
    
    return max_index


# If you want to directly modify the arguments for image and video inference rather than using command,
# uncomment the code in main() function. 

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
BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='YOLO-World')
    parser.add_argument('--webcam', action='store_true', help='whether to use webcam.')
    parser.add_argument('--webcam-addr', type=int, default=0, help='the web camera address, local camera or rtsp address.')
    parser.add_argument("--webcam-res", default=[1280, 720], nargs=2, type=int, help='the web camera resolution')
    parser.add_argument("--source", type=str, default='', help='the source path of a single image / video')
    parser.add_argument("--classes", default=None, nargs='*', type=str, help='the list of classes (things) to be detected')
    
    # The Inference package provides the YOLO-World model in three versions: S, M, and L. 
    # You can load them by defining model_id as yolo_world/s, yolo_world/m, and yolo_world/l, respectively.
    parser.add_argument("--model-id", default='yolov8m-world.pt', type=str)
    args = parser.parse_args()
    return args


def infer_image(source_path, classes, model_id, confidence=0.003, nms_threshold=0.1):
    # output_image = get_output_name(source_path)
    
    model = YOLOWorld(model=model_id)
    if classes:
        model.set_classes(classes)
    
    image = cv2.imread(source_path)
    
    results = model.predict(image, conf=confidence)
    detections = sv.Detections.from_ultralytics(results[0]).with_nms(threshold=nms_threshold)
    
    # def callback(image_slice: np.ndarray) -> sv.Detections:
    #     results = model.predict(image_slice, conf=confidence)
    #     return sv.Detections.from_ultralytics(results[0]).with_nms(threshold=nms_threshold)
    # slicer = sv.InferenceSlicer(callback = callback)
    # detections = slicer(image)
    
    annotated_image = image.copy()
    annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections=detections)
    
    if classes is None:
        classes = model.names
    labels = [
        f"{classes[class_id]} {confidence:0.3f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]
    annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections=detections, labels=labels)
    sv.plot_image(annotated_image, (10, 10))
    # with sv.ImageSink(target_dir_path='results/images', overwrite=True) as sink:
    #     sink.save_image(annotated_image, image_name=output_image)



def infer_video(source_path, classes, model_id, confidence=0.003, nms_threshold=0.1):
    target_path = get_output_name(source_path)
    
    frame_generator = sv.get_video_frames_generator(source_path)
    # frame = next(frame_generator)
    
    model = YOLOWorld(model=model_id)
    if classes:
        model.set_classes(classes)
    
    video_info = sv.VideoInfo.from_video_path(source_path)
    width, height = video_info.resolution_wh
    frame_area = width * height
    
    prev_frame_time = 0
    new_frame_time = 0
    
    with sv.VideoSink(target_path=target_path, video_info=video_info) as sink:
        for frame in frame_generator:
            results = model.predict(frame, conf=confidence)
            detections = sv.Detections.from_ultralytics(results[0]).with_nms(threshold=nms_threshold)
            
            # Filter detections by area (optional)
            detections = detections[(detections.area / frame_area) < 0.1]
            
            labels = [
                f"{classes[class_id]} {confidence:0.3f}"
                for class_id, confidence
                in zip(detections.class_id, detections.confidence)
            ]
            
            font = cv2.FONT_HERSHEY_SIMPLEX 
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) 
            fps = int(fps) 
            fps = str(fps) 
            cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA) 
            
            annotated_frame = frame.copy()
            annotated_frame = BOUNDING_BOX_ANNOTATOR.annotate(annotated_frame, detections=detections)
            annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, detections=detections, labels=labels)
            sink.write_frame(annotated_frame)
            
            # Just for showing video frames during code execution
            cv2.imshow("YOLO-World", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # prev_frame_time = new_frame_time 
            prev_frame_time = time.time()
                
def infer_webcam(webcam_addr, classes, model_id, webcam_res, confidence=0.003, nms_threshold=0.1): 
    width, height = webcam_res
    cap = cv2.VideoCapture(webcam_addr)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    model = YOLOWorld(model=model_id)
    if classes:
        model.set_classes(classes)
        
    center_zone = sv.PolygonZone(polygon=(CENTER_ZONE_POLYGON * np.array(webcam_res)).astype(int), frame_resolution_wh=tuple(webcam_res))
    center_zone_annotator = sv.PolygonZoneAnnotator(zone=center_zone, color=sv.Color.RED, thickness=2, text_thickness=4, text_scale=2)
    left_zone = sv.PolygonZone(polygon=(LEFT_ZONE_POLYGON * np.array(webcam_res)).astype(int), frame_resolution_wh=tuple(webcam_res))
    left_zone_annotator = sv.PolygonZoneAnnotator(zone=left_zone, color=sv.Color.RED, thickness=2, text_thickness=4, text_scale=2)
    right_zone = sv.PolygonZone(polygon=(RIGHT_ZONE_POLYGON * np.array(webcam_res)).astype(int), frame_resolution_wh=tuple(webcam_res))
    right_zone_annotator = sv.PolygonZoneAnnotator(zone=right_zone, color=sv.Color.RED, thickness=2, text_thickness=4, text_scale=2)
    
    interval = 5000
    prev_frame_time = 0
    new_frame_time = 0
    
    while True:
        ret, frame = cap.read()
        
        results = model.predict(frame, conf=confidence)
        detections = sv.Detections.from_ultralytics(results[0]).with_nms(threshold=nms_threshold)
        
        mask = center_zone.trigger(detections=detections)
        left_zone.trigger(detections=detections)
        right_zone.trigger(detections=detections)
        
        font = cv2.FONT_HERSHEY_SIMPLEX 
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) 
        fps = int(fps) 
        fps = str(fps) 
        cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA) 
        
        annotated_frame = frame.copy()
        annotated_frame = BOUNDING_BOX_ANNOTATOR.annotate(annotated_frame, detections=detections)
        
        if classes is None:
            classes = model.names
        labels = [
            f"{classes[class_id]} {confidence:0.3f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]
        annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, detections=detections, labels=labels)
        annotated_frame = center_zone_annotator.annotate(scene=annotated_frame)
        annotated_frame = left_zone_annotator.annotate(scene=annotated_frame)
        annotated_frame = right_zone_annotator.annotate(scene=annotated_frame)
        # Just for showing video frames during code execution
        cv2.imshow("YOLO-World", annotated_frame)
        # Wait 30 ms per frame. If Escape pressed, exit loop.
        if cv2.waitKey(30) == 27: 
            break
        
        # prev_frame_time = new_frame_time 
        prev_frame_time = time.time()


def avoid_obstacles(webcam_addr, classes, model_id, webcam_res, confidence=0.25, nms_threshold=0.1):
    width, height = webcam_res
    cap = cv2.VideoCapture(webcam_addr)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    model = YOLOWorld(model=model_id)
    if classes:
        model.set_classes(classes)
        
    center_zone = sv.PolygonZone(polygon=(CENTER_ZONE_POLYGON * np.array(webcam_res)).astype(int), frame_resolution_wh=tuple(webcam_res))
    center_zone_annotator = sv.PolygonZoneAnnotator(zone=center_zone, color=sv.Color.RED, thickness=2, text_thickness=4, text_scale=2)
    left_zone = sv.PolygonZone(polygon=(LEFT_ZONE_POLYGON * np.array(webcam_res)).astype(int), frame_resolution_wh=tuple(webcam_res))
    left_zone_annotator = sv.PolygonZoneAnnotator(zone=left_zone, color=sv.Color.RED, thickness=2, text_thickness=4, text_scale=2)
    right_zone = sv.PolygonZone(polygon=(RIGHT_ZONE_POLYGON * np.array(webcam_res)).astype(int), frame_resolution_wh=tuple(webcam_res))
    right_zone_annotator = sv.PolygonZoneAnnotator(zone=right_zone, color=sv.Color.RED, thickness=2, text_thickness=4, text_scale=2)
    
    interval = 5000
    prev_frame_time = 0
    new_frame_time = 0
    direction = "Straight"
    
    while True:
        ret, frame = cap.read()
        
        results = model.predict(frame, conf=confidence)
        detections = sv.Detections.from_ultralytics(results[0]).with_nms(threshold=nms_threshold)
        
        mask = center_zone.trigger(detections=detections)
        left_zone.trigger(detections=detections)
        right_zone.trigger(detections=detections)
        
        font = cv2.FONT_HERSHEY_SIMPLEX 
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) 
        fps = int(fps) 
        fps = str(fps) 
        cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA) 
        
        annotated_frame = frame.copy()
        annotated_frame = BOUNDING_BOX_ANNOTATOR.annotate(annotated_frame, detections=detections)
        
        if classes is None:
            classes = model.names
        labels = [
            f"{classes[class_id]} {confidence:0.3f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]
        annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, detections=detections, labels=labels)
        annotated_frame = center_zone_annotator.annotate(scene=annotated_frame)
        annotated_frame = left_zone_annotator.annotate(scene=annotated_frame)
        annotated_frame = right_zone_annotator.annotate(scene=annotated_frame)
        
        # Output directions
        center_count = center_zone_annotator.zone.current_count
        left_count = left_zone_annotator.zone.current_count
        right_count = right_zone_annotator.zone.current_count
        if (center_count <= left_count) and (center_count <= right_count):
            direction = "Straight"
        elif left_count <= right_count:
            direction = "Left"
        else:
            direction = "Right"
        # direction = "Straight"
        cv2.putText(annotated_frame, direction, (7, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - 7)), font, 3, (100, 255, 0), 3, cv2.LINE_AA) 
        
        # Show video frames during code execution
        cv2.imshow("YOLO-World", annotated_frame)
        # Wait 30 ms per frame. If Escape pressed, exit loop.
        if cv2.waitKey(30) == 27: 
            break
        
        # prev_frame_time = new_frame_time 
        prev_frame_time = time.time()
    
# def find_object(webcam_addr, classes, model_id, webcam_res, confidence=0.5, nms_threshold=0.1):
#     width, height = webcam_res
#     cap = cv2.VideoCapture(webcam_addr)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
#     model = YOLOWorld(model=model_id)
#     if classes:
#         model.set_classes(classes)
    
    
    

def main():
    args = parse_arguments()
    model_id = args.model_id
    classes = args.classes
    source_path = args.source
    webcam = args.webcam
    webcam_addr = args.webcam_addr
    webcam_res = args.webcam_res
    
    ## Example 1: Image inference
    model_id = 'yolov8l-world.pt'
    # classes = ["person", "backpack", "dog", "eye", "nose", "ear", "tongue"]
    # classes = ["scissors"]
    # source_path = "test/room.webp"
    # infer_image(source_path, classes, model_id)
    
    ## Example 2: Video inference
    # model_id = 'yolov8m-world.pt'
    # classes = ["yellow filling"]
    # source_path = "test/yellow-filling.mp4"
    # infer_video(source_path, classes, model_id)
    
    
    if webcam:
        avoid_obstacles(webcam_addr, classes, model_id, webcam_res)
        cv2.destroyAllWindows()
    # else:
    #     file_type = check_extension(source_path)
    #     if file_type == 'image':
    #         infer_image(source_path, classes, model_id)
    #     elif file_type == 'video':
    #         infer_video(source_path, classes, model_id)
    #     else:
    #         raise Exception("Source path must be a video / image")
    
    
    

if __name__ == "__main__":
    main()