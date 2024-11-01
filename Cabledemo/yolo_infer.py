import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('./runs/train/v5m6/best.pt')

# Open the video file
# video_path = "path/to/your/video/file.mp4"
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1)==ord('q'):
            break
cv2.destroyAllWindows()
cap.release()