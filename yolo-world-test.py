from ultralytics import YOLOWorld, YOLO
import cv2
import supervision as sv
import argparse
import time

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='YOLO-World live')
    parser.add_argument("--src", default=None, type=str)
    parser.add_argument("--obj-name", default=None, type=str)
    args = parser.parse_args()
    return args


def main():
    # args = parse_arguments()
    # src = args.src
    # obj_name = args.obj_name
    obj_name = ["fan"]
    src = 0
    model = YOLOWorld('yolov8m-world.pt')

    if obj_name == None:
        obj_name = ""
    model.set_classes(obj_name)

    bbox_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    cap = cv2.VideoCapture(src)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, img = cap.read()

        if not ret:
            break

        results = model.predict(img)

        print(results)
        detections = sv.Detections.from_ultralytics(results[0])
        labels = [
            f"{obj_name[class_id]} {confidence:0.3f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]
        annotated_frame = bbox_annotator.annotate(
            scene=img.copy(),
            detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

        # confidences = [detection.confidence for detection in detections]

        # Add confidence scores to the annotated frame

        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps_value = frame_count / elapsed_time

        # Overlay FPS information onto the frame
        cv2.putText(annotated_frame, f"FPS: {fps_value:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("test", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()