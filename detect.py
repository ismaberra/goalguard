import argparse
import os
import platform
import sys
from pathlib import Path
import re 

import torch
import csv
import numpy as np
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) 

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode

ball_class_id = 32
tracker = None
tracking_started = False
ball_detected_last_frame = False
last_position = None
cooling_off_counter = 0
movement_threshold = -5  
# initial_window = (1700, 1425, 2125, 1700)
initial_window = (800, 600, 1200, 1000)

def is_within_initial_window(x_center, y_center, initial_window):
    x_min, y_min, x_max, y_max = initial_window
    print(f"Checking window: for point ({x_center}, {y_center})")
    return x_min <= x_center <= x_max and y_min <= y_center <= y_max

def initialize_tracker(frame, bbox):
    global tracker
    bbox = tuple(map(int, bbox))
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)
    print("Tracker initialized.")

def update_tracker(frame):
    global tracker
    success, bbox = tracker.update(frame)
    return success, bbox

def reset_tracking_state():
    global tracker, tracking_started, ball_detected_last_frame, last_position, cooling_off_counter
    print("Resetting tracking state.")
    tracker = None
    tracking_started = False
    ball_detected_last_frame = False
    last_position = None
    cooling_off_counter = 0

def process_detections(frame, det, detected_frames, frame_number, csv_writer, csv_file):
    global tracker, tracking_started, ball_detected_last_frame, last_position, cooling_off_counter, initial_window
    print(f"Processing detection for frame {frame_number}, detections: {len(det)}")
    if cooling_off_counter > 0:
        cooling_off_counter -= 1
        if cooling_off_counter == 0:
            reset_tracking_state()
            print("Cooling countdown finished.")

    if not tracking_started and len(det) > 0:
        for detection in det:
            x1, y1, x2, y2, conf, cls = detection
            if cls == ball_class_id and is_within_initial_window((x1+x2)/2, (y1+y2)/2, initial_window):
                bbox = (x1, y1, x2-x1, y2-y1)
                initialize_tracker(frame, bbox)
                tracking_started = True
                ball_detected_last_frame = True
                last_position = ((x1+x2)/2, (y1+y2)/2)
                break

    if tracking_started:
        success, bbox = update_tracker(frame)
        if success:
            x_center, y_center = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
            if last_position and y_center - last_position[1] < movement_threshold and cooling_off_counter == 0:
                detected_frames.append(frame_number)
                csv_writer.writerow([frame_number])
                csv_file.flush()
                cooling_off_counter = 300
                print(f"Movement detected of {y_center - last_position[1]} marking frame {frame_number}.")
                ball_detected_last_frame = True
            last_position = (x_center, y_center)
        else:
            if ball_detected_last_frame and cooling_off_counter == 0 and len(det) == 0:
                print(f"Ball lost, marking frame {frame_number}.")
                detected_frames.append(frame_number)
                csv_writer.writerow([frame_number])
                csv_file.flush()
                cooling_off_counter = 100
                ball_detected_last_frame = False
                last_position = None
                reset_tracking_state()
            elif len(det) > 0:
                ball_detected_last_frame = any(det[:, 5] == ball_class_id)

    return detected_frames, cooling_off_counter

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov9-e.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    global tracker, tracking_started, ball_detected_last_frame, last_position, cooling_off_counter, initial_window
    detected_frames = []

    source_path = Path(source)  # Ensure source is treated as a Path object
    save_img = not nosave and not source_path.suffix == '.txt'  # save inference images
    is_file = source_path.suffix[1:] in IMG_FORMATS + VID_FORMATS
    is_url = source_path.as_posix().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = str(source_path).isnumeric() or source_path.suffix == '.txt' or (is_url and not is_file)
    screenshot = source_path.as_posix().startswith('screen')
    if is_url and is_file:
        source_path = Path(check_file(source_path.as_posix()))  

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok) 
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True) 

    video_number = re.search(r'shoot-(\d+).mp4', source)
    video_number = video_number.group(1) if video_number else 'default'
    csv_filename = f"detected_frames_{video_number}.csv"
    csv_path = Path(project) / csv_filename  
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame'])

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    bs = 1 
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                det = det[det[:, 5] == ball_class_id]
                if len(det): 
                    detected_frames, cooling_off_counter = process_detections(frame=im0, det=det, detected_frames=detected_frames, frame_number=frame, csv_writer=csv_writer, csv_file=csv_file)

                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, " 

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img: 
                        c = int(cls) 
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  
                    if vid_path[i] != save_path: 
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  
                        if vid_cap: 
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4')) 
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    csv_file.close()
    t = tuple(x.t / seen * 1E3 for x in dt)  
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov9-e.pt', help='model path')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IOU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
