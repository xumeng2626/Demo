import os
import sys
from pathlib import Path
from time import time
import tkinter as tk
from PIL import Image, ImageTk

import threading
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import (LOGGER, Profile, check_img_size, check_imshow, cv2,
                           increment_path, non_max_suppression, scale_boxes, strip_optimizer)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode

stable_frame_count = 0
last_detected_objects = [0,0,0,0,0,0]
last_logged_step = [0,0,0,0,0,0]

@smart_inference_mode()
def run(
        left_frame,upper_right_frame,lower_right_frame,mid_right_frame,
        weights=ROOT / 'runs/train/v5m6/best.pt',  # model path or triton URL
        source=0,  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(1280,1280),  # inference size (height, width)
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
    global process_definition
    last_process_definition = False

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt')
    
  
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

    global soll_label,ist_label,comment_label,image_label,complete_label,step
    global current_step_index, current_step, process_steps, process_label,last_step_index
    step = 0
    soll_label = tk.Label(upper_right_frame, text="", font=("Helvetica", 20), bg="white", anchor="n", wraplength=300)
    ist_label = tk.Label(upper_right_frame, text="", font=("Helvetica", 20), bg="white", anchor="center", wraplength=300)
    comment_label = tk.Label(lower_right_frame, text="Step 1 incomplete", font=("Helvetica", 20), bg="white", anchor="center", wraplength=350)
    complete_label = tk.Label(mid_right_frame, text="", font=("Helvetica", 20), bg="white",fg="green", anchor="center", wraplength=350)
    image_label = tk.Label(left_frame, image=None)

    # Load process definition(We are reading from Steps.txt)
    if process_definition is False:
        global all_complete
        process_file_path = 'Steps.txt'  # Change this path accordingly
    
        process_label = []
        
        process_steps = []
        current_step_index = 0  # Initialize the current step index
        last_target_dict = {}

        curr_target_dict = {}
        with open(process_file_path, 'r') as file:
                first_char = file.read(1)
                if not first_char:
                    # File is empty, write default content
                    with open(process_file_path, 'w') as new_file:
                        new_file.write("Step 0\nConnection_R 0\nO-ring_R 0\nCoupling_R 0\nCoupling_B 1\nO-ring_B 0\nConnection_B 0\n")


        with open(process_file_path, 'r') as file:

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
        all_complete = False    
        last_step_index = -1
        current_step = "Step 0"
        # expected_classes = process_steps[current_step_index][current_step]
        absolute_path = os.path.abspath(process_file_path)
        LOGGER.info(f"Absolute Ubuntu Path: {absolute_path}")   
        LOGGER.info(f"Process steps: {process_steps}")
        LOGGER.info(f"Process label: {process_label}")
        global stable_frame_detect
        stable_frame_count = 0


    curr_target_dict = {}
    def on_process_definition_change():
        global last_detected_objects,stable_frame_count,last_logged_step,step,complete_label,comment_label,instruction_label,process_steps,process_label,all_complete,current_step_index,current_step,last_target_dict,curr_target_dict
        global soll_label,ist_label,comment_label,image_label
        # Define processes (We are writing into Steps.txt)
        if process_definition is True:
            curr_target_dict = {}
            last_target_dict = {}
            step = 0
            last_logged_step = [0,0,0,0,0,0]
            soll_label.config(text="")
            ist_label.config(text="")
            complete_label.config(text="")
            comment_label.config(text="")
            soll_label.pack_forget()
            ist_label.pack_forget()
            complete_label.pack()
            comment_label.pack()

            instruction_label = tk.Label(upper_right_frame, text="Defining Steps\nPlease place a part", font=("Helvetica", 20,"bold"), bg="white",fg = "green", wraplength=350)
            instruction_label.place(relx=0.5, rely=0.1, anchor="n")
            instruction_label.pack(fill="none", expand=False)


            open('Steps.txt', 'w').close()  # Clear the file

        # Load process definition(We are reading from Steps.txt)
        if process_definition is False:
            process_file_path = 'Steps.txt'  # Change this path accordingly
            curr_target_dict = {}
            process_label = []
            comment_label.config(text="")
            comment_label.pack()
            process_steps = []
            current_step_index = 0  # Initialize the current step index
            last_target_dict = {}

            with open(process_file_path, 'r') as file:
                first_char = file.read(1)
                if not first_char:
                    # File is empty, write default content
                    with open(process_file_path, 'w') as new_file:
                        new_file.write("Step 0\nConnection_R 0\nO-ring_R 0\nCoupling_R 0\nCoupling_B 1\nO-ring_B 0\nConnection_B 0\n")

            with open(process_file_path, 'r') as file:
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
            all_complete = False    
            
            current_step = "Step 0"
            # expected_classes = process_steps[current_step_index][current_step]
            LOGGER.info(f"Process steps: {process_steps}")
            LOGGER.info(f"Process label: {process_label}")
            instruction_label.config(text="")
            instruction_label.pack_forget()

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        
        if process_definition != last_process_definition:
            last_target_dict={}
            curr_target_dict = {}
            on_process_definition_change()
            last_step_index = -1
        
        last_process_definition = process_definition
        

        LOGGER.info(f"PROCESS DEFINITION: {process_definition}")
        if stop_thread:
            break
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        if process_definition is False: 
                        
                        if current_step_index >= len(process_steps):
                            current_step_index = len(process_steps)-1
                        
                        filtered_list = [item.strip() for item in process_label[current_step_index][current_step] if item.strip()]
                        # LOGGER.info(f"FILTERED LIST {filtered_list}")
                        current_target = filtered_list
                        
                        soll = "Target:\n"
                        
                        for ii, line in enumerate(filtered_list):
                            
                            
                            soll = soll + line + '\n'
                        
                        soll_label.config(text="")
                        soll_label.config(text=soll)
                        soll_label.pack(fill="none", expand=False)


                        if last_step_index < current_step_index:
                            
                            a=1
                            curr_target_dict = {}
                            for item in filtered_list:
                                prefix, key = item.split()
                                if key in last_target_dict:
                                    last_target_dict[key] += int(prefix)
                                else:
                                    last_target_dict[key] = int(prefix)

                                # if key in curr_target_dict:
                                #     curr_target_dict[key] += int(prefix)
                                # else:
                                #     curr_target_dict[key] = int(prefix)

                            LOGGER.info(f"last targer:{last_target_dict}")
                            last_step_index = current_step_index

                        
                        total_det_per_class = ""
                        
                        
        
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                
                num_each_cls = [0,0,0,0,0,0]
                            
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    int_n = int(n)
                    num_each_cls[int(c)] = int_n
                    
                
                    if process_definition is False: 
                        if num_each_cls[int(c)] != 0:
                           total_det_per_class = total_det_per_class + f"{n} {names[int(c)]},\n"                  
#####ADD IST LABEL
                        else: total_det_per_class = None

                    else:
                        total_det_per_class = '%g %s ' % (n,names[int(c)])  # add to string
                        s += total_det_per_class + ', '
                        s += f"{n} {names[int(c)]}{'' * (n > 1)}, "  # add to string

                if process_definition is False:
                    # if current_step_index == 0: 
                        
                    #     ist_label.config(text="")
                    #     ist_label.config(text=f"Misplaced parts:\n{total_det_per_class}\n")
                    # else:
                    det_per_class_list = total_det_per_class.split(',')
                    det_per_class_list.pop()
                    LOGGER.info(F"DET_PER_CLASS: {det_per_class_list}")

                    # Initialize a dictionary to store the difference
                    difference_dict = {}
                    LOGGER.info(f"det_per_class_list:{det_per_class_list}")
                    # Iterate through each item in det_per_class_list
                    for item in det_per_class_list:
                        
                        prefix, key = item.strip().split()
                        detected_value = int(prefix)

                        # Subtract both current step targets (last_target_dict) and cumulative targets (curr_target_dict)
                        curr_target_value = curr_target_dict.get(key, 0)
                        last_target_value = last_target_dict.get(key, 0)
                        # expected_total_value = curr_target_value + last_target_value
                        difference = detected_value - (last_target_value)

                        if difference != 0:
                        #     difference_dict[key] = difference
                        # else:
                            # Add the item to the difference_dict if it doesn't exist in last_target_dict
                            difference_dict[key] = (int(difference))

                    # Construct the result string from the difference_dict
                    result_str = '\n'.join([f"{value} {key}" for key, value in difference_dict.items() if value > 0])

                    LOGGER.info(f"Difference_str:{result_str}")
                    ist_label.config(text="",bg="white")
                    if result_str.strip():
                        ist_label.config(text=f"Misplaced parts:\n{result_str}\n",fg="red")

                        ist_label.pack(fill="none", expand=False)      
                        
                        
        
                if process_definition is False:
                    if current_step:
                                expected_classes = process_steps[current_step_index][current_step]

                                truth_classes = [item[0] for item in expected_classes]
                                truth_numbers = [item[1] for item in expected_classes]

                                #differences = [(pred_class, pred_number, true_number) for pred_class, pred_number, true_number in zip(truth_classes, truth_numbers, num_each_cls) if pred_number != true_number]
                                differences = [(pred_class, pred_number, true_number) for pred_class, pred_number, true_number in zip(truth_classes, truth_numbers, num_each_cls) if pred_number == 1 and true_number == 0]
                                if all_complete:
                                    comment_label.config(text=f"")
                                    
                                else:
                                    if differences and current_step_index < len(process_steps):
                                        # different_classes = ', '.join(pred_class for pred_class, _, _ in differences)
                                        if not all_complete:
                                            comment_label.config(text=f"{current_step} incomplete")
                                            LOGGER.info(f"{current_step} incomplete")
                                        
                                        comment_label.pack(fill="none", expand=False)
                            
                                    elif len(difference_dict) == 0: 
                                        stable_frame_count+=1
                                        if stable_frame_count==50:
                                            stable_frame_count = 0
                                            comment_label.config(text=f"")
                                            comment_label.pack(fill="none", expand=False)
                                            complete_label.config(text=f"{current_step} complete")
                                            complete_label.pack(fill="none", expand=False)
                                            LOGGER.info(f"current_step_index{current_step_index}, len(process_steps){len(process_steps)}")
                                            if current_step_index == len(process_steps)-1:
                                                all_complete = True
                                                complete_label.config(text=f"All steps complete",font=("Helvetica", 26))
                                                complete_label.pack(fill="none", expand=False)
                                                ist_label.config(text='')
                                                ist_label.pack(fill="none", expand=False)
                                                soll_label.config(text='')
                                                soll_label.pack(fill="none", expand=False) 
                                                soll_label.update_idletasks()

                                            if current_step_index < len(process_steps)-1:
                                                current_step_index += 1  # Move to the next step
                                                last_target = current_target
                                                if current_step_index < len(process_steps):
                                                    current_step = list(process_steps[current_step_index].keys())[0]

                                

                # Check if process_definition is True and if detected objects have changed
                if process_definition is True:
                    global last_detected_objects,last_logged_step
                    LOGGER.info(f"curr detected obj {num_each_cls}\n last detected obj {last_detected_objects}")

                    # current frame contains same non-trivial result as last frame -> current frame is a stable frame
                    if num_each_cls == last_detected_objects and num_each_cls != last_logged_step and any(num_each_cls):
                        stable_frame_count += 1
                        LOGGER.info(f"stable_frame_count  {stable_frame_count}\n")
                    else :
                        stable_frame_count = 0
                        unstable_frame_count = num_each_cls

                    if stable_frame_count == 40:
                        stable_frame_count = 0
                        
                        # Checking changes from 0 to 1 in num_each_cls
                        changes = [i for i in range(len(num_each_cls)) if (num_each_cls[i] == 1 and last_logged_step[i] == 0) or (num_each_cls[i] == 2 and last_logged_step[i] == 1)]

                        if changes:
                            with open('Steps.txt', 'a') as f:
                                f.write(f"Step {step}\n")
                                instruction_text = f"'\n\nStep {step}: \n"
                                LOGGER.info(f"STABLE FRAME detected obj {num_each_cls}, logging STEP {step}, class names {names}")
                                diff = [num_each_cls[i] - last_logged_step[i] if (num_each_cls[i] - last_logged_step[i]) > 0 else 0 for i in range(len(num_each_cls))]

                                
                                for index,obj in enumerate(diff):
                                    f.write(f"{names[index]} {obj}\n")
                                    LOGGER.info(f"{names[index]} {obj}\n")
                                    instruction_text += f"{names[index]} {obj}\n"

                                instruction_label.config(text=f"\nStep {step} registered\nPlease place the next part")
                                instruction_label.pack(fill="none", expand=False)
                                comment_label.config(text=instruction_text)
                                comment_label.pack(fill="none", expand=False)
                                step += 1
                        
                        last_logged_step = num_each_cls

                    last_detected_objects = num_each_cls
                        
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))             


            # Stream results
            im0 = annotator.result()
            
            if view_img:
                pil_image = Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
                photo = ImageTk.PhotoImage(pil_image)

                
                
                image_label.config(image=photo)
                image_label.image = photo  # Keep a reference
                image_label.pack(fill=tk.BOTH, expand=True)


        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # if stop_thread:
    #     return
    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


stop_thread = False

def create_gui(width, height):
    global detection_thread, stop_thread

    def quit_program():
        global stop_thread
        stop_thread = True
        root.quit()

    def define_steps():
        LOGGER.info(f"define_steps called")        
        global process_definition
        process_definition = True
        

    def run_detection():
        LOGGER.info(f"run_detection called")
        global process_definition
        process_definition = False

    # Create the main window
    root = tk.Tk()
    root.title("Assembly Detection")
    root.geometry(f"{width}x{height}")

    # Create a frame for the left section (image)
    left_frame = tk.Frame(root, bg="white", width=width*0.75, height=height)  # 75% of window width
    left_frame.pack_propagate(False)  # Prevent the frame from resizing to fit its contents
    left_frame.pack(side=tk.LEFT)

    # Create a frame for the right section (text)
    right_frame = tk.Frame(root, bg="white", width=width*0.25, height=height)  # 25% of window width
    right_frame.pack_propagate(False)
    right_frame.pack(side=tk.LEFT)

    # Create a frame for the upper part of the right section (instruction)
    upper_right_frame = tk.Frame(right_frame, bg="white", width=width*0.25, height= height*0.4)  # 75% of right section height
    upper_right_frame.place(relx=0.5, rely=0.15, anchor="center")
    upper_right_frame.pack_propagate(False)
    upper_right_frame.pack(fill=tk.BOTH, expand=True)

    # Add a label for the instruction subtitle
    instruction_label = tk.Label(upper_right_frame, text="Instruction", font=("Helvetica", 20, "bold", "underline"),anchor="n")
    instruction_label.pack(fill=tk.BOTH, expand=True)

    # Create a frame for the lower part of the right section (comment)
    lower_right_frame = tk.Frame(right_frame, bg="white", width=width*0.25, height=height*0.3)  # 25% of right section height
    lower_right_frame.pack_propagate(False)
    lower_right_frame.pack(fill=tk.BOTH, expand=True)

    # Create a frame for the lower part of the right section (comment)
    mid_right_frame = tk.Frame(right_frame, bg="white", width=width*0.25, height=height*0.2)  # 25% of right section height
    mid_right_frame.pack_propagate(False)
    mid_right_frame.pack(fill=tk.BOTH, expand=True)

    # complete_label = tk.Label(mid_right_frame, text="Complete", font=("Helvetica", 12), bg="white",fg="green", anchor="center", wraplength=350)
    # complete_label.pack(fill=tk.BOTH, expand=True)

    # Add a label for the comment subtitle
    comment_label = tk.Label(lower_right_frame, text="Comment", font=("Helvetica", 20, "bold", "underline"),anchor="n")
    comment_label.pack(fill=tk.BOTH, expand=True)

    # Create a frame for the "Quit" button
    quit_frame = tk.Frame(right_frame, bg="orange", width=width*0.25, height=height*0.1)  # Set the size of the frame for the button
    quit_frame.pack_propagate(False)  # Prevent the frame from resizing to fit its contents
    quit_frame.pack(fill=tk.BOTH, expand=True)  # Position the frame in the lower-right corner
    
    # Define Steps button
    define_steps_button = tk.Button(quit_frame, text="Define Steps", command=define_steps)
    define_steps_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Run Detection button
    run_detection_button = tk.Button(quit_frame, text="Run Detection", command=run_detection)
    run_detection_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)


    # Add a button to quit the program
    quit_button = tk.Button(quit_frame, text="Quit", command=quit_program)
    quit_button.pack(fill=tk.BOTH,expand=True)

    return root,left_frame,upper_right_frame,lower_right_frame,mid_right_frame


def run_gui(width, height):
    global detection_thread
    root, left_frame, upper_right_frame, lower_right_frame, mid_right_frame = create_gui(width, height)
    detection_thread = threading.Thread(target=continuous_detection, args=(root, left_frame, upper_right_frame, lower_right_frame, mid_right_frame, False))
    detection_thread.start()
    root.mainloop()


def continuous_detection(root,left_frame,upper_right_frame,lower_right_frame,mid_right_frame,process_definitio):
    global stop_thread, process_definition
    process_definition=False
    stop_thread = False
    while not stop_thread:
        try:
            run(left_frame=left_frame, upper_right_frame=upper_right_frame, lower_right_frame=lower_right_frame, mid_right_frame=mid_right_frame)
        except Exception as e:
            print("An error occurred:", e)
        root.after(10)  # Update GUI every 10 milliseconds



def main():
    # check_requirements(exclude=('tensorboard', 'thop'))

    width = 1600
    height = 800

    run_gui(width, height)


if __name__ == "__main__":
    
    main()
