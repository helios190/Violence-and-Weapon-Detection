import cv2
from ultralytics import YOLO
from config import YOLO_MODEL_PATH, YOLO_CLASSES, CONFIDENCE_THRESHOLD

# Load YOLO model
model = YOLO(YOLO_MODEL_PATH)

def detect_objects(frame):
    detections = []
    results = model(frame)
    for detection in results[0].boxes:
        cls_id = int(detection.cls[0])
        label = YOLO_CLASSES[cls_id]
        conf = float(detection.conf[0])
        if conf > CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            detections.append({
                "label": label,
                "confidence": conf,
                "box": (x1, y1, x2, y2)
            })
    return detections