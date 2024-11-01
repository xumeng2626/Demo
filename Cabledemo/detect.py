# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s.xml                # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
from time import time

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

stable_frame_count = 0
last_detected_objects = [0,0,0,0,0,0]
last_logged_step = [0,0,0,0,0,0]

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
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
        process_definition=False,  # process defined in Steps.txt
):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download
    #process_definition = bool(process_definition)
        
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    names = ['Connection_R', 'O-ring_R', 'Coupling_R', 'Coupling_B', 'O-ring_B', 'Connection_B']
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
        #capture from camera at location 0
        cap = cv2.VideoCapture(0)
        #set the width and height, and UNSUCCESSFULLY set the exposure time
        cap.set(cv2.CAP_PROP_BRIGHTNESS,-24)
        cap.set(cv2.CAP_PROP_CONTRAST,24)
        cap.set(cv2.CAP_PROP_HUE,-8)
        cap.set(cv2.CAP_PROP_SATURATION,77)
        cap.set(cv2.CAP_PROP_GAMMA,109)
        cap.set(cv2.CAP_PROP_GAIN,12)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs



    # Define processes (We are writing into Steps.txt)
    if process_definition is True:
        print("process_definition is True")
        step = 0
        global last_detected_objects,stable_frame_count,last_logged_step
       
        


    # Load process definition(We are reading from Steps.txt)
    if process_definition is False:
        process_file_path = 'Steps.txt'  # Change this path accordingly
    
        print("process_definition is False")
        process_label = []
        
        process_steps = []
        current_step_index = 0  # Initialize the current step index
        

        with open(process_file_path, 'r') as file:
        #     for line in file:
        #         class_name, number = line.strip().split()
        #         goal.append(int(number))

        #         process_label += f"{number} {class_name}\n" if int(number) != 0 else ""

            for line in file:
                line = line.strip()
                if line.startswith("Step"):
                    current_step = line 
                    process_steps.append({current_step: []})
                    process_label.append({current_step: []})
                elif current_step:
                    class_name, number = line.split()
                    process_steps[-1][current_step].append((class_name, int(number)))
                    process_label[-1][current_step].append(f"{number} {class_name}" if int(number) != 0 else "")
            
        current_step = "Step 0"
        # expected_classes = process_steps[current_step_index][current_step]
        LOGGER.info(f"Process steps: {process_steps}")
        LOGGER.info(f"Process label: {process_label}")
    




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

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)


        
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
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()


                # Initialize coordinates of first class label ###################################################################################################
                org = (100,100)
                iii = 0
                process_1 = False 
                process_2 = False 
                process_3 = False 
                num_each_cls = [0,0,0,0,0,0]
                
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    # print(f"c is {c}")
                    # print(f"n is {n}")
                    int_n = int(n)
                    num_each_cls[int(c)] = int_n
                    
                
                    if process_definition is False: 
                        org_process = (int(im.shape[1]*0.8),100)
                        iii+=1
                        annotator.plot_counter((org_process[0], int(im.shape[2]*0.6)), "Soll:")

                        if current_step_index >= len(process_steps):
                            current_step_index = len(process_steps)-1
                        LOGGER.info(current_step_index)
                        LOGGER.info(current_step)

                        filtered_list = [item.strip() for item in process_label[current_step_index][current_step] if item.strip()]
                        
                        for ii, line in enumerate(filtered_list):
                            annotator.plot_counter((org[0], int(im.shape[2]*(ii*0.07+0.65))), line)

                        annotator.plot_counter((org_process[0], org_process[1] - int(im.shape[2]*0.05)), "Ist:")
                        if num_each_cls[int(c)] != 0:
                            total_det_per_class = '%g %s ' % (n, names[int(c)])  # add to string
                            
                        else: total_det_per_class = None
                        annotator.plot_counter((org[0], org[1] + int(im.shape[2]*(iii*0.07))), total_det_per_class)

                    else:
                        total_det_per_class = '%g %s ' % (n, names[int(c)])  # add to string
                        s += total_det_per_class + ', '
                        s += f"{n} {names[int(c)]}{'' * (n > 1)}, "  # add to string


                        annotator.plot_counter(org, total_det_per_class)
                        org = (org[0], org[1] + int(im.shape[2]*0.07))
                      
                        
        
                if process_definition is False:
                    if current_step:
                                expected_classes = process_steps[current_step_index][current_step]

                                truth_classes = [item[0] for item in expected_classes]
                                truth_numbers = [item[1] for item in expected_classes]

                                #differences = [(pred_class, pred_number, true_number) for pred_class, pred_number, true_number in zip(truth_classes, truth_numbers, num_each_cls) if pred_number != true_number]
                                differences = [(pred_class, pred_number, true_number) for pred_class, pred_number, true_number in zip(truth_classes, truth_numbers, num_each_cls) if pred_number == 1 and true_number == 0]
                                if differences:
                                    different_classes = ', '.join(pred_class for pred_class, _, _ in differences)


                                    annotator.stage_detector(label=f'{current_step} incomplete, different number of {different_classes}', color = (255,0,0))
                                    
                                else: 
                                    annotator.stage_detector(label=f'{current_step} complete')
                                    if current_step_index < len(process_steps)-1:
                                        current_step_index += 1  # Move to the next step
                                
                                        if current_step_index < len(process_steps):
                                            current_step = list(process_steps[current_step_index].keys())[0]

                                

                # Check if process_definition is True and if detected objects have changed
                if process_definition is True:
                    LOGGER.info(f"curr detected obj {num_each_cls}\n last detected obj {last_detected_objects}")

                    # current frame contains same non-trivial result as last frame -> current frame is a stable frame
                    if num_each_cls == last_detected_objects and num_each_cls != last_logged_step and any(num_each_cls):
                        stable_frame_count += 1
                        LOGGER.info(f"stable_frame_count  {stable_frame_count}\n")
                    else :
                        stable_frame_count = 0
                        unstable_frame_count = num_each_cls

                    if stable_frame_count == 20:
                        stable_frame_count = 0
                        
                        # Checking changes from 0 to 1 in num_each_cls
                        changes = [i for i in range(len(num_each_cls)) if num_each_cls[i] == 1 and last_logged_step[i] == 0]

                        if changes:
                            with open('Steps.txt', 'a') as f:
                                f.write(f"Step {step}\n")
                                step += 1
                                LOGGER.info(f"STABLE FRAME detected obj {num_each_cls}, logging STEP {step}, class names {names}")
                                diff = [num_each_cls[i] - last_logged_step[i] for i in range(len(num_each_cls))]
                                for index,obj in enumerate(diff):
                                    f.write(f"{names[index]} {obj}\n")
                                    LOGGER.info(f"{names[index]} {obj}\n")
                        
                        last_logged_step = num_each_cls

                    last_detected_objects = num_each_cls
                        

                """
                
                if process_definition==False:
                    missing_parts = []
                    redundant_part = False
                    LOGGER.info(num_each_cls)
                    if (num_each_cls[2] != 0) and (num_each_cls[3] != 0) : 
                        process_1 = True 
                    if process_1 == True and (num_each_cls[1] != 0) and (num_each_cls[4] != 0) :
                        process_2 = True 
                    if process_2 == True and (num_each_cls[0] != 0) and (num_each_cls[5] != 0) :
                        process_3 = True 
                    if(max(num_each_cls)>1):
                        redundant_part = True

                    if ((num_each_cls[2] == 0) or (num_each_cls[3] == 0)) and ((num_each_cls[1] != 0) or (num_each_cls[4] != 0) or (num_each_cls[0] != 0) or (num_each_cls[5] != 0)):
                        missing_parts.append('Coupling_nut')

                    elif ((num_each_cls[1] == 0) or (num_each_cls[4] == 0)) and ((num_each_cls[0] != 0) or (num_each_cls[5] != 0)):
                        missing_parts.append('O-ring')
                    

                    while 1 :
                        if redundant_part:
                            annotator.stage_detector(label=(f'Redundant part exists!'), color = (0,0,255))
                            break
                        if len(missing_parts) != 0: 
                            annotator.stage_detector(label=(f'Missing {missing_parts}!'), color = (0,0,255))
                            break
                        if process_3 :
                            annotator.stage_detector(label='Process 3 done!')
                        elif process_2:
                            annotator.stage_detector(label='Process 2 done!')
                        elif process_1:
                            annotator.stage_detector(label='Process 1 done!')
                        break
                    
                    """
    
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))             

                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
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
    parser.add_argument('--process_definition', action='store_true', help='process defined in Process.txt')
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
