import cv2

class Tracker:
    def __init__(self, type="CSRT", frame=None, bbox=None):
        self.frame = frame
        self.bbox = bbox
        self.tracker = None
        if type == "CSRT":
            self.tracker = cv2.TrackerCSRT_create()
            self.tracker.init(frame, bbox)
        elif type == "KCF":
            self.tracker = cv2.TrackerKCF_create()
            self.tracker.init(frame, bbox)

    def track(self, frame, bbox):
        ok, bbox = self.tracker.update(frame)
        if ok:
            # Tracking success: Draw the bounding box on the frame
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
            self.frame = frame
            self.bbox = bbox
        else:
            # Tracking failure: Handle the failure, e.g., reinitialize the tracker
            print("lost track, using yolo")



