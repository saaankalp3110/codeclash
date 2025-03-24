import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import math
from scipy.spatial import distance
from sort import Sort

# Load YOLOv8 model
model = YOLO('yolov8m.pt')  # This will auto-download if it's not in your directory

# Open video capture
cap = cv2.VideoCapture("codeclash/03_collision_risk/test/test_drive.mp4")  #change path

# Class names for YOLO
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Parameters
ttc_threshold = 0.7  # Time-to-collision threshold in seconds
fps = cap.get(cv2.CAP_PROP_FPS)

# Conversion factor from pixels to meters
PIXEL_TO_METER = 0.001


# Function to calculate TTC
def calculate_ttc(distance, relative_velocity):
    if relative_velocity <= 0:  # Vehicle is moving away or stationary
        return float('inf')
    return distance / relative_velocity  # TTC in seconds


# Function to calculate intersection over union (IoU)
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate the area of the two boxes
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Calculate the area of overlap
    x_left = max(x1, x1_2)
    y_top = max(y1, y1_2)
    x_right = min(x2, x2_2)
    y_bottom = min(y2, y2_2)

    overlap_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Calculate the IoU
    iou = overlap_area / float(area1 + area2 - overlap_area)
    return iou


prev_frame_time = None
prev_positions = {}

# To track trajectories
trajectory_history = {}

while True:
    success, frame = cap.read()
    if not success:
        break

    current_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Get current time in seconds

    results = model(frame, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Only display bounding boxes for objects with confidence over 60%
            if currentClass in ["person", "car", "truck", "bus", "motorbike", "cow"] and conf > 0.6:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    current_positions = {}

    for result in resultsTracker:
        x1, y1, x2, y2, obj_id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        # Update positions
        current_positions[obj_id] = (cx, cy, x1, y1, x2, y2)

        # Store trajectory history
        if obj_id not in trajectory_history:
            trajectory_history[obj_id] = []

        # Append current center to trajectory
        trajectory_history[obj_id].append((cx, cy))

        # Default bounding box color is green for safe distance
        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(0, 255, 0))
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Draw trajectory line for each object
        if len(trajectory_history[obj_id]) > 1:
            for i in range(1, len(trajectory_history[obj_id])):
                start_point = trajectory_history[obj_id][i - 1]
                end_point = trajectory_history[obj_id][i]
                cv2.line(frame, start_point, end_point, (255, 255, 0), 4)

    # Handle bounding box overlap
    for obj_id, (prev_cx, prev_cy, prev_x1, prev_y1, prev_x2, prev_y2) in prev_positions.items():
        if obj_id not in current_positions:
            continue

        curr_cx, curr_cy, curr_x1, curr_y1, curr_x2, curr_y2 = current_positions[obj_id]

        # Check overlap with other bounding boxes
        for other_obj_id, (other_cx, other_cy, other_x1, other_y1, other_x2, other_y2) in current_positions.items():
            if obj_id != other_obj_id:
                iou = calculate_iou([curr_x1, curr_y1, curr_x2, curr_y2], [other_x1, other_y1, other_x2, other_y2])

                # Create a copy of the original frame to apply transparency
                frame_copy = frame.copy()

                # If IoU is greater than 70%, show a bigger warning
                if iou > 0.7:
                    # Create a semi-transparent red bounding box
                    overlay = frame_copy.copy()
                    cv2.rectangle(overlay, (curr_x1, curr_y1), (curr_x2, curr_y2), (0, 0, 255), -1)  # Filled red box
                    cv2.addWeighted(overlay, 0.4, frame, 1 - 0.4, 0, frame)  # Blend with 40% transparency

                    cv2.putText(frame, "Severe Collision Risk", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)  # White text on red background
                    cv2.rectangle(frame, (curr_x1, curr_y1), (curr_x2, curr_y2), (0, 0, 255), 3)  # Outline with thicker border

                # If IoU is greater than 50%, show a moderate warning
                elif iou > 0.5:
                    # Create a semi-transparent yellow bounding box
                    overlay = frame_copy.copy()
                    cv2.rectangle(overlay, (curr_x1, curr_y1), (curr_x2, curr_y2), (0, 255, 255), -1)  # Filled yellow box
                    cv2.addWeighted(overlay, 0.4, frame, 1 - 0.4, 0, frame)  # Blend with 40% transparency

                    cv2.putText(frame, "Collision Risk", (75, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)  # Red text on yellow background
                    cv2.rectangle(frame, (curr_x1, curr_y1), (curr_x2, curr_y2), (0, 255, 255), 3)  # Outline with thicker border

              

    resized_frame = cv2.resize(frame, (1260, 720))  # Smaller display size
    cv2.imshow("Frame", resized_frame)

    if cv2.waitKey(4) & 0xFF == ord('q'):
        break

    prev_positions = current_positions
    prev_frame_time = current_frame_time

cap.release()
cv2.destroyAllWindows()
