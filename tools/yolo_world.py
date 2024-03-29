from ultralytics import YOLOWorld
import cv2

class YoloWorld:
    def __init__(self, model_path):
        self.yolo_world = YOLOWorld(model_path)
        self.iou_threshold = 0.1

    def find_object(self, color_frame, conf, iou, max_det, visualize=False):
        results = self.yolo_world.predict(color_frame, conf=conf, iou=iou, max_det=max_det, verbose=False)[0]
        # Get the most confident detection
        if len(results.boxes.xyxy.cpu().tolist()) == 0:
            return None, 0
        bbox = results.boxes.xyxy.cpu().tolist()[0]
        # map to int
        bbox = [int(x) for x in bbox]
        confidence = results.boxes.conf.cpu()[0].item()
        object_name = results.names[0]
        if visualize:
            # Annotate the frame with the bounding box
            label = f'{object_name} {confidence:.2f}'
            self.plot_box_and_label(color_frame, max(round(sum(color_frame.shape) / 2 * 0.003), 2), bbox, label)
        return bbox, confidence

    def set_object_to_find(self, object_to_find: list):
        self.yolo_world.set_classes(object_to_find)

    @staticmethod
    def draw_text(
            img,
            text,
            font=cv2.FONT_HERSHEY_SIMPLEX,
            pos=(0, 0),
            font_scale=1,
            font_thickness=2,
            text_color=(0, 255, 0),
            text_color_bg=(0, 0, 0),
    ):

        offset = (5, 5)
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        rec_start = tuple(x - y for x, y in zip(pos, offset))
        rec_end = tuple(x + y for x, y in zip((x + text_w, y + text_h), offset))
        cv2.rectangle(img, rec_start, rec_end, text_color_bg, -1)
        cv2.putText(
            img,
            text,
            (x, int(y + text_h + font_scale - 1)),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

        return text_size

    @staticmethod
    def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255),
                           font=cv2.FONT_HERSHEY_COMPLEX):
        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), font, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)


