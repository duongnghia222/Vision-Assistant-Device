from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np



def polygon_to_mask(polygon: np.ndarray, resolution_wh: Tuple[int, int]) -> np.ndarray:
    """Generate a mask from a polygon.

    Args:
        polygon (np.ndarray): The polygon for which the mask should be generated,
            given as a list of vertices.
        resolution_wh (Tuple[int, int]): The width and height of the desired resolution.

    Returns:
        np.ndarray: The generated 2D mask, where the polygon is marked with
            `1`'s and the rest is filled with `0`'s.
    """
    width, height = resolution_wh
    mask = np.zeros((height, width))

    cv2.fillPoly(mask, [polygon], color=1)
    return mask


def clip_boxes(xyxy: np.ndarray, resolution_wh: Tuple[int, int]) -> np.ndarray:
    """
    Clips bounding boxes coordinates to fit within the frame resolution.

    Args:
        xyxy (np.ndarray): A numpy array of shape `(N, 4)` where each
            row corresponds to a bounding box in
        the format `(x_min, y_min, x_max, y_max)`.
        resolution_wh (Tuple[int, int]): A tuple of the form `(width, height)`
            representing the resolution of the frame.

    Returns:
        np.ndarray: A numpy array of shape `(N, 4)` where each row
            corresponds to a bounding box with coordinates clipped to fit
            within the frame resolution.
    """
    result = np.copy(xyxy)
    width, height = resolution_wh
    result[:, [0, 2]] = result[:, [0, 2]].clip(0, width)
    result[:, [1, 3]] = result[:, [1, 3]].clip(0, height)
    return result


        
def in_zone(x1, x2, x) :
    if x > x1 and x < x2 :
        return 'center'
    elif (x <= x1):
        return 'left'
    else :
        return 'right'
        

# def process_frame2(self, frame: np.ndarray) -> np.ndarray:
#         results = self.model.predict(frame, conf=self.conf_threshold, iou=self.iou_threshold, agnostic_nms=True) 
#         # results = self.model.track(frame, persist=True, conf=self.conf_threshold, agnostic_nms=True)
#         annotated_frame = results[0].plot()
        
#         boxes = results[0].boxes.xywh.cpu()
#         track_ids = results[0].boxes.id.int().cpu().tolist()
        
#         for box, track_id in zip(boxes, track_ids):
#             x, y, w, h = box
#             track = self.track_history[track_id]
#             track.append((float(x), float(y)))  # x, y center point
#             if len(track) > 30:  # retain 90 tracks for 90 frames
#                 track.pop(0)

#             # Draw the tracking lines
#             points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
#             cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
        
#         return annotated_frame


# def process_frame(self, frame: np.ndarray) -> np.ndarray:
#     results = self.model.predict(frame, conf=self.conf_threshold, iou=self.iou_threshold, agnostic_nms=True)
#     detections = sv.Detections.from_ultralytics(results[0]).with_nms(threshold=self.iou_threshold)
#     # detections = self.tracker.update_with_detections(detections)
    
#     annotated_frame = self.annotate_frame(frame, detections)
#     return annotated_frame

# def annotate_frame(self, frame: np.ndarray, detections: sv.Detections) ->np.ndarray:
#     annotated_frame = frame.copy()
#     annotated_frame = self.bounding_box_annotator.annotate(annotated_frame, detections=detections)
    
#     labels = [
#         f"{self.classes[class_id]}: {confidence:0.3f}"
#         for class_id, confidence
#         in zip(detections.class_id, detections.confidence)
#     ]
    
#     # labels = [
#     #     f"#{tracker_id} {self.classes[class_id]}: {confidence:0.3f}"
#     #     for tracker_id, class_id, confidence
#     #     in zip(detections.tracker_id, detections.class_id, detections.confidence)
#     # ]
    
#     annotated_frame = self.label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
#     for direction in ['center', 'left', 'right']:
#         annotated_frame = self.zone_annotator[direction].annotate(scene=annotated_frame, label='')
#     return annotated_frame    