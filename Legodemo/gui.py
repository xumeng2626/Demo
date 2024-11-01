import os
import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

class AssemblyDetectionApp:
    def __init__(self, root):
        self.root = root
        self.cap = None
        self.model = None
        self.running = False
        self.current_line = 0
        self.filtered_annotations = []
        self.frame_width = 640
        self.frame_height = 480
        self.annotations = []
        self.process_file_path = '/home/zotac/Desktop/Demo/Legodemo/Process.txt'
        self.model_path = '/home/zotac/Desktop/Demo/Legodemo/weights/best.pt'
        self.assets_folder = '/home/zotac/Desktop/Demo/Legodemo/asset/'
    
        self.available_classes = []
        self.setup_gui()

        # Start in Define Steps mode by default
        self.define_steps()

    def setup_gui(self):
        width = 1080
        height = 720
        self.root.title("Assembly Detection")
        self.root.geometry(f"{width}x{height}")
        print(self.process_file_path)
        # Left Frame for Webcam Feed
        self.left_frame = tk.Frame(self.root, bg="white", width=width*0.7, height=height)
        self.left_frame.pack_propagate(False)
        self.left_frame.pack(side=tk.LEFT)

        # Label to display the webcam feed
        self.video_label = tk.Label(self.left_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Right Frame for Instructions and Comments
        self.right_frame = tk.Frame(self.root, bg="white", width=width*0.3, height=height)
        self.right_frame.pack_propagate(False)
        self.right_frame.pack(side=tk.LEFT)

        # Upper Right Frame for Instructions
        self.upper_right_frame = tk.Frame(self.right_frame, bg="gray", width=width*0.3, height= height*0.4)
        self.upper_right_frame.pack_propagate(False)
        self.upper_right_frame.pack(fill=tk.BOTH, expand=True)

        # Instruction Label
        self.instruction_label = tk.Label(self.upper_right_frame, text="", font=("Helvetica", 20, "bold", "underline"), anchor="n")
        self.instruction_label.pack(fill=tk.BOTH, expand=True)

        # Class Image Label (Not used in Define Steps mode)
        self.class_image_label = tk.Label(self.upper_right_frame)
        self.class_image_label.pack(fill=tk.BOTH, expand=True)

        # Middle Right Frame for Comments
        self.mid_right_frame = tk.Frame(self.right_frame, bg="white", width=width*0.25, height=height*0.4)
        self.mid_right_frame.pack_propagate(False)
        self.mid_right_frame.pack(fill=tk.BOTH, expand=True)

        # Comment Label
        self.comment_label = tk.Label(self.mid_right_frame, text="", font=("Helvetica", 20), bg="white", fg="black", anchor="center", wraplength=300)
        self.comment_label.pack(fill=tk.BOTH, expand=True)

        # Lower Right Frame for Buttons
        self.quit_frame = tk.Frame(self.right_frame, bg="orange", width=width*0.25, height=height*0.03)
        self.quit_frame.pack_propagate(False)
        self.quit_frame.pack(fill=tk.BOTH, expand=True)

        # Define Steps Button
        self.define_steps_button = tk.Button(self.quit_frame, text="Define Steps", command=self.define_steps)
        self.define_steps_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Run Detection Button
        self.run_detection_button = tk.Button(self.quit_frame, text="Run Detection", command=self.run_detection)
        self.run_detection_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Quit Button
        self.quit_button = tk.Button(self.quit_frame, text="Quit", command=self.quit_program)
        self.quit_button.pack(fill=tk.BOTH, expand=True)

    def define_steps(self):
        # Open Process.txt and delete its content
        with open(self.process_file_path, 'w') as f:
            pass

        # Initialize variables
        self.detecting_background = True
        self.background_detected = False
        self.background_detection_count = 0
        self.brick_detection_counts = {}
        self.logged_bricks = []
        self.comment_label.config(text="Detecting background plate", fg="blue")
        self.instruction_label.config(text="")
        self.class_image_label.config(image='')
        excluded_classes = [3, 5,7, 13,16, 17, 18, 21, 22, 26, 27, 28, 29, 30, 32, 37, 38, 39, 40,51]

        # Start camera feed
        self.model = YOLO(self.model_path)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS,-20)
        self.cap.set(cv2.CAP_PROP_CONTRAST,25)
        self.cap.set(cv2.CAP_PROP_HUE,-5)
        self.cap.set(cv2.CAP_PROP_SATURATION,60)
        self.cap.set(cv2.CAP_PROP_GAMMA,109)
        self.cap.set(cv2.CAP_PROP_GAIN,12)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.running = True
        self.mode = 'define_steps'

        # Start the video loop
        self.video_loop()

    def run_detection(self):
        # Read and parse Process.txt
        try:
            with open(self.process_file_path, 'r') as file:
                lines = file.readlines()
        except FileNotFoundError:
            messagebox.showerror("Error", f"File {self.process_file_path} not found.")
            return

        self.annotations = []
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 5:
                    annotation = {
                        'class_name': int(parts[0]),
                        'offset_x': float(parts[1]),
                        'offset_y': float(parts[2]),
                        'width': float(parts[3]),
                        'height': float(parts[4])
                    }
                    self.annotations.append(annotation)
                else:
                    messagebox.showwarning("Warning", f"Invalid line in Process.txt: {line}")

        self.filtered_annotations = self.annotations

        if not self.filtered_annotations:
            messagebox.showwarning("Warning", "No steps defined or no matching class images found.")
            return

        # Initialize variables for detection mode
        self.current_line = 0
        self.mode = 'detection'
        self.comment_label.config(text="Detecting background plate", fg="blue")
        self.instruction_label.config(text="")
        self.background_plate_position = None
        self.background_detection_count = 0


    def video_loop(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        # Rotate the frame by 180 degrees
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        if not ret:
            self.cap.release()
            self.running = False
            return
        
        excluded_classes = [3,5,12, 13,14, 17, 18, 21, 22, 25,26, 27, 28, 29, 30, 32,33, 37, 38, 39, 40,51]

        if self.mode == 'define_steps':
            results = self.model(frame)
            detections = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            if self.detecting_background:
                # Detect class 50
                detected = False
                for i, bbox in enumerate(detections):
                    detected_class = int(classes[i])
                    if detected_class == 50:
                        # Draw bounding box for class 50
                        x1, y1, x2, y2 = bbox.astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Class {detected_class}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        self.background_detection_count += 1
                        detected = True
                        # Compute the center
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        self.background_plate_position = (int(x_center), int(y_center))
                        break  # Only process the first detected background plate
                if not detected:
                    self.background_detection_count = 0

                if self.background_detection_count >= 100:
                    # Background plate detected
                    self.detecting_background = False
                    self.background_detected = True
                    self.comment_label.config(text=f"Background plate detected at: {self.background_plate_position}", fg="green")
                    self.instruction_label.config(text="Place one brick")
                    self.background_detection_count = None
                    self.brick_detection_counts = {}
            else:
                # Now detect other classes
                detected_classes_positions = []
                for i, bbox in enumerate(detections):
                    detected_class = int(classes[i])
                    if detected_class != 50 and detected_class not in excluded_classes:
                        # Get the center of the detected brick
                        x1, y1, x2, y2 = bbox.astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Class {detected_class}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        position = (int(x_center), int(y_center))
                        detected_classes_positions.append((detected_class, position))

                        # Check if this brick has already been logged at a similar position
                        already_logged = False
                        for logged_brick in self.logged_bricks:
                            if logged_brick['class'] == detected_class and self.is_same_position(logged_brick['position'], position):
                                already_logged = True
                                break
                        if already_logged:
                            continue  # Skip this brick

                        # Update detection count for this brick at this position
                        key = (detected_class, position)
                        if key in self.brick_detection_counts:
                            self.brick_detection_counts[key] += 1
                        else:
                            self.brick_detection_counts[key] = 1

                        # If detected for 15 frames, log it
                        if self.brick_detection_counts[key] >= 10:
                            # Calculate relative position to background plate
                            offset_x = x_center - self.background_plate_position[0]
                            offset_y = y_center - self.background_plate_position[1]

                            # Normalize positions to [0,1]
                            width = (x2 - x1) / self.frame_width
                            height = (y2 - y1) / self.frame_height

                            # Normalize offsets
                            offset_x_rel = offset_x / self.frame_width
                            offset_y_rel = offset_y / self.frame_height

                            # Store in Process.txt
                            with open(self.process_file_path, 'a') as f:
                                f.write(f"{detected_class} {offset_x_rel} {offset_y_rel} {width} {height}\n")

                            # Log the brick
                            self.logged_bricks.append({
                                'class': detected_class,
                                'position': position
                            })
                            # Display the corresponding class image
                            class_image_path = os.path.join(self.assets_folder, f"{detected_class}.jpg")
                            class_image = cv2.imread(class_image_path)
                            if class_image is not None:
                                class_image = cv2.cvtColor(class_image, cv2.COLOR_BGR2RGB)
                                class_image = cv2.rotate(class_image, cv2.ROTATE_90_CLOCKWISE)
                                # Dynamically resize the image to fit within a maximum size while maintaining aspect ratio
                                max_dimension = 280
                                height, width = class_image.shape[:2]
                                if max(height, width) > max_dimension:
                                    scaling_factor = max_dimension / float(max(height, width))
                                    new_size = (int(width * scaling_factor), int(height * scaling_factor))
                                    class_image = cv2.resize(class_image, new_size, interpolation=cv2.INTER_AREA)
                                im = Image.fromarray(class_image)
                                imgtk = ImageTk.PhotoImage(image=im)
                                self.class_image_label.imgtk = imgtk
                                self.class_image_label.configure(image=imgtk)
                            else:
                                self.class_image_label.config(image='')

                            self.comment_label.config(text=f"Class {detected_class} detected at: {x_center}, {y_center}", fg="green")

                            # Remove the detection count for this brick
                            del self.brick_detection_counts[key]
                    else:
                        continue  # Skip background plate detection in this phase

                # Reset detection counts for bricks not detected in current frame
                keys_to_delete = []
                for key in self.brick_detection_counts:
                    detected = False
                    for detected_class, position in detected_classes_positions:
                        if key == (detected_class, position):
                            detected = True
                            break
                    if not detected:
                        keys_to_delete.append(key)
                for key in keys_to_delete:
                    del self.brick_detection_counts[key]

        elif self.mode == 'detection':
            results = self.model(frame)
            detections = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            # First, detect background plate if not detected yet
            if self.background_plate_position is None:
                detected = False
                for i, bbox in enumerate(detections):
                    detected_class = int(classes[i])
                    if detected_class == 50:
                        self.background_detection_count += 1
                        detected = True
                        # Optionally, draw the bounding box
                        x1, y1, x2, y2 = bbox.astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Compute the center
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        self.background_plate_position = (x_center, y_center)
                        break
                if not detected:
                    self.background_detection_count = 0
                if self.background_detection_count >= 100:
                    # Background plate detected
                    self.comment_label.config(text="Background plate detected", fg="green")
                    self.instruction_label.config(text="")
                    self.background_detection_count = None
            else:
                # Proceed with detecting bricks
                if self.current_line >= len(self.filtered_annotations):
                    self.comment_label.config(text="All steps completed!", fg="green")
                    # Clear instruction and class image labels
                    self.instruction_label.config(text="")
                    self.class_image_label.config(image='')
                    # Do not stop the video loop

                else:
                    anno = self.filtered_annotations[self.current_line]
                    class_name = anno['class_name']

                    # Compute expected position
                    offset_x = anno['offset_x'] * self.frame_width
                    offset_y = anno['offset_y'] * self.frame_height

                    expected_x_center = self.background_plate_position[0] + offset_x
                    expected_y_center = self.background_plate_position[1] + offset_y

                    # Compute bounding box coordinates
                    width = anno['width'] * self.frame_width
                    height = anno['height'] * self.frame_height

                    x1 = int(expected_x_center - width / 2)
                    y1 = int(expected_x_center + width / 2)
                    x2 = int(expected_y_center - height / 2)
                    y2 = int(expected_y_center + height / 2)

                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, x2), (y1, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Place Class {class_name}", (x1, x2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Update instruction label
                    self.instruction_label.config(text=f"Place brick {class_name}")
                    # Display the corresponding class image
                    class_image_path = os.path.join(self.assets_folder, f"{class_name}.jpg")
                    class_image = cv2.imread(class_image_path)
                    if class_image is not None:
                        class_image = cv2.cvtColor(class_image, cv2.COLOR_BGR2RGB)
                        class_image = cv2.rotate(class_image, cv2.ROTATE_90_CLOCKWISE)
                        # Dynamically resize the image to fit within a maximum size while maintaining aspect ratio
                        max_dimension = 280
                        height, width = class_image.shape[:2]
                        if max(height, width) > max_dimension:
                            scaling_factor = max_dimension / float(max(height, width))
                            new_size = (int(width * scaling_factor), int(height * scaling_factor))
                            class_image = cv2.resize(class_image, new_size, interpolation=cv2.INTER_AREA)
                        im = Image.fromarray(class_image)
                        imgtk = ImageTk.PhotoImage(image=im)
                        self.class_image_label.imgtk = imgtk
                        self.class_image_label.configure(image=imgtk)
                    else:
                        self.class_image_label.config(image='')

                    # Initialize correct detection counter if not already done
                    if not hasattr(self, 'correct_detection_count'):
                        self.correct_detection_count = 0

                    # Run detection
                    detected_at_position = False
                    for i, bbox in enumerate(detections):
                        detected_class = int(classes[i])
                        if detected_class == class_name:
                            x1_det, y1_det, x2_det, y2_det = bbox.astype(int)
                            x_center_det = (x1_det + x2_det) / 2
                            y_center_det = (y1_det + y2_det) / 2

                            # Check if detected brick is at the expected position
                            if self.is_same_position((x_center_det, y_center_det), (expected_x_center, expected_y_center)):
                                detected_at_position = True
                                break

                    if detected_at_position:
                        self.correct_detection_count += 1
                        if self.correct_detection_count >= 10:
                            correct_detection = True
                            self.comment_label.config(text=f"Correct brick {class_name} detected at the correct position.", fg="green")
                            self.current_line += 1
                            self.correct_detection_count = 0  # Reset counter for next brick
                        else:
                            self.comment_label.config(text=f"Verifying brick {class_name}... ({self.correct_detection_count}/15)", fg="orange")
                    else:
                        self.correct_detection_count = 0
                        self.comment_label.config(text=f"Waiting for brick {class_name} to be placed correctly.", fg="red")

        # Convert frame to RGB and display in Tkinter
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=im)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Update the GUI and call this function again
        self.root.after(10, self.video_loop)

    def is_same_position(self, pos1, pos2, threshold=20):
        return abs(pos1[0] - pos2[0]) < threshold and abs(pos1[1] - pos2[1]) < threshold

    def quit_program(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AssemblyDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.quit_program)  # Handle window close button
    root.mainloop()
