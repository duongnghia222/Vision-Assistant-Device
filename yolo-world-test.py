import cv2
import argparse
import torch
import supervision as sv
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmyolo.registry import RUNNERS

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World Demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='device used for inference.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def inference_detector(runner,
                       frame,
                       use_amp=False):

    data_info = dict(img_id=0, img_path='dummy_path', texts=[['']])
    data_info = runner.pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    with autocast(enabled=use_amp), torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances

    pred_instances = pred_instances.cpu().numpy()
    detections = sv.Detections(xyxy=pred_instances['bboxes'],
                               class_id=pred_instances['labels'],
                               confidence=pred_instances['scores'])

    labels = [
        f"{idx}: {class_id[0]} {confidence:.2f}" for idx, (class_id, confidence) in
        enumerate(zip(detections.class_id, detections.confidence))
    ]

    # label images
    annotated_frame = BOUNDING_BOX_ANNOTATOR.annotate(frame, detections)
    annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, detections, labels=labels)
    cv2.imshow('YOLO-World Demo', annotated_frame)
    cv2.waitKey(0)  # Wait for any key press to close


if __name__ == '__main__':
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = './work_dirs'
    cfg.load_from = args.checkpoint

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.call_hook('before_run')
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    runner.pipeline = Compose(pipeline)
    runner.model.eval()

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Couldn't open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame from webcam.")
            break

        inference_detector(runner,
                           frame,
                           use_amp=False)  # Adjust as needed

    cap.release()
    cv2.destroyAllWindows()
