import cv2
from ultralytics import YOLO
# import torch
import matplotlib.pyplot as plt

# Check if CUDA is available and set the device accordingly
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# Loading the v8 model
model = YOLO("yolov8n.pt")

# Opening the video file
video_path = "E:\\Major_demo\\Data\\2.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
    
# Capture for single frame to select roi
ret, frame = cap.read()
if not ret:
    print("Could not read the frame")
    cap.release()
    exit()
    
# Display the roi in the frame
roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select ROI")

# Get the ROI coordinates (x, y, width, height)
roi_x, roi_y, roi_width, roi_height = map(int, roi)
    
# Output height, width
width_output = 800
height_output = 700

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object
output_path = '2_output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width_output, height_output))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize the frame
    frame = cv2.resize(frame, (width_output, height_output))
    
    # ROI frame
    roi_frame = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
    
    # Perform detection with YOLOv8
    results = model(roi_frame)

    # Visualize the results on the frame
    annotated_roi = results[0].plot()
    
    # Place the annotated ROI back into the original frame (optional)
    frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width] = annotated_roi

    # Display the frame with annotations (optional)
    cv2.imshow('YOLOv8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Write the annotated frame to the output video
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
