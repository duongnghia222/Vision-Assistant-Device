from ultralytics import YOLOWorld


class YoloWorld:
    def __init__(self, model_path):
        self.yolo_world = YOLOWorld(model_path)
        self.iou_threshold = 0.1

    def find_object(self, color_frame):
        results = self.yolo_world.predict(color_frame, verbose=False)[0]
        # Get the most confident detection
        if len(results.boxes.xyxy.cpu().tolist()) == 0:
            return None, 0
        bbox = results.boxes.xyxy.cpu().tolist()[0]
        # map to int
        bbox = [int(x) for x in bbox]
        confidence = results.boxes.conf.cpu()[0].item()
        return bbox, confidence

    def set_object_to_find(self, object_to_find: list):
        self.yolo_world.set_classes(object_to_find)


