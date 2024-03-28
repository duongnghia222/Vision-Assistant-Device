from ultralytics import YOLOWorld, YOLO
import cv2
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
    obj_name = ["bottle"]
    src = 0
    model = YOLOWorld('yolov8m-world.pt')

    if obj_name == None:
        obj_name = ""
    model.set_classes(obj_name)


    cap = cv2.VideoCapture(src)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, img = cap.read()

        if not ret:
            break
        # predict the image without verbose
        results = model(img, verbose=False)
        print(results[0])
        print("=====")
        print(results[0].boxes)
        print("=====")
        if len (results[0].boxes.xyxy.cpu().tolist()) > 0:
            print(results[0].boxes.xyxy.cpu().tolist()[0])
            print("=====")
            # print confidence
            print(results[0].boxes.conf.cpu()[0])

        # Annotate the frame with the bounding box







        # confidences = [detection.confidence for detection in detections]

        # Add confidence scores to the annotated frame

        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps_value = frame_count / elapsed_time

        # Overlay FPS information onto the frame
        cv2.putText(img, f"FPS: {fps_value:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("test", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()