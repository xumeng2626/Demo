This folder contains two seperated demos: Legodemo and Cabledemo. 

# README for Cabledemo

[`Cabledemo`](Cabledemo) is a Python script designed to run a GUI for object detection using the YOLOv5 model. The script leverages the Tkinter library for the GUI and integrates YOLOv5 for real-time object detection.

### Installation

1. **Clone the repository:**
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Create and activate a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. **Run the GUI:**
    ```sh
    python detect_GUI.py
    ```

2. **GUI Controls:**
    - **Define Steps:** Define the steps for the detection process. The defined steps are stored in Steps.txt
    - **Run Detection:** Start the object detection process. The detected results are compareds to the content of Steps.txt.
    - **Quit:** Exit the application.

### Script Details

#### Key Functions

- **`run`**: The main function to run the object detection process.
    - **Parameters:**
        - `left_frame`, `upper_right_frame`, `lower_right_frame`, `mid_right_frame`: GUI frames for displaying results.
        - `weights`: Path to the model weights.
        - `source`: Source for the input (0 for webcam).
        - `data`: Path to the dataset configuration file.
        - `imgsz`: Inference size.
        - `conf_thres`: Confidence threshold.
        - `iou_thres`: IoU threshold for non-max suppression.
        - `max_det`: Maximum detections per image.
        - `device`: Device to run the inference on (e.g., CPU, GPU).
        - `view_img`: Flag to display results.
        - `save_txt`: Flag to save results to text files.
        - `save_conf`: Flag to save confidences in text labels.
        - `save_crop`: Flag to save cropped prediction boxes.
        - `nosave`: Flag to not save images/videos.
        - `classes`: Filter by class.
        - `agnostic_nms`: Class-agnostic NMS.
        - `augment`: Augmented inference.
        - `visualize`: Visualize features.
        - `update`: Update all models.
        - `project`: Directory to save results.
        - `name`: Name of the experiment.
        - `exist_ok`: Flag to allow existing project/name.
        - `line_thickness`: Thickness of bounding box lines.
        - `hide_labels`: Flag to hide labels.
        - `hide_conf`: Flag to hide confidences.
        - `half`: Use FP16 half-precision inference.
        - `dnn`: Use OpenCV DNN for ONNX inference.
        - `vid_stride`: Video frame-rate stride.

#### Global Variables

- `stable_frame_count`: Counter for stable frames. Larger number reduces false positives but leads to longer detection time.
- `last_detected_objects`: List to store the last detected objects.
- `last_logged_step`: List to store the last logged steps.

### Notes

- Ensure that the required model weights are correctly specified.
- Ensure that the required file Steps.txt is present.
- The script is designed to work with YOLOv5. Ensure that the YOLOv5 repository is correctly set up and integrated.
- For training details, consult [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) and yolov5_ultralytics.ipynb. 

### Acknowledgements

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for the object detection model.
- [Tkinter](https://docs.python.org/3/library/tkinter.html) for the GUI framework.
- [PyTorch](https://pytorch.org/) for the deep learning framework.
- [OpenCV](https://opencv.org/) for computer vision functions.
- [Pillow](https://python-pillow.org/) for image processing.


# README for Legodemo

[`Legodemo`](Legodemo) is a Python script designed to run a graphical user interface (GUI) for assembly detection using the YOLO model. The script leverages the Tkinter library for the GUI and integrates YOLO for real-time object detection.

## Installation

1. **Clone the repository:**
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Create and activate a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the GUI:**
    ```sh
    python gui.py
    ```

2. **GUI Controls:**
    - **Define Steps:** Define the steps for the detection process.
    - **Run Detection:** Start the object detection process.
    - **Quit:** Exit the application.

## Script Details

### Key Classes

- **`AssemblyDetectionApp`**: The main class to run the assembly detection GUI.
    - **Attributes:**
        - `root`: The root Tkinter window.
        - `cap`: Video capture object.
        - `model`: YOLO model for detection.
        - `running`: Boolean flag to indicate if detection is running.
        - `current_line`: Current line in the process file.
        - `filtered_annotations`: List of filtered annotations.
        - `frame_width`: Width of the video frame.
        - `frame_height`: Height of the video frame.
        - `annotations`: List of annotations.
        - `process_file_path`: Path to the process file.
        - `model_path`: Path to the YOLO model weights.
        - `assets_folder`: Path to the assets folder.
        - `available_classes`: List of available classes for detection.
    - **Methods:**
        - `__init__`: Initializes the GUI and sets up the default mode.
        - `setup_gui`: Sets up the GUI layout and components.
        - `define_steps`: Starts the GUI in Define Steps mode.

### GUI Layout

- **Left Frame**: Displays the webcam feed.
- **Right Frame**: Contains instructions and comments.
    - **Upper Right Frame**: Displays instructions.
    - **Middle Right Frame**: Displays comments.

## Notes

- Ensure that the required model weights and process file are correctly specified.
- The script is designed to work with the YOLO model. Ensure that the YOLO repository is correctly set up and integrated.

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the object detection model.
- [Tkinter](https://docs.python.org/3/library/tkinter.html) for the GUI framework.
- [OpenCV](https://opencv.org/) for computer vision functions.
- [Pillow](https://python-pillow.org/) for image processing.

