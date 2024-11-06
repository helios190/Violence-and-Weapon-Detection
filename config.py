import os

# Paths to model files
TFLITE_MODEL_PATH = './models/model.tflite'
YOLO_MODEL_PATH = './models/best-10.pt'

# Check if model files exist
assert os.path.exists(TFLITE_MODEL_PATH), "TFLITE model path is invalid."
assert os.path.exists(YOLO_MODEL_PATH), "YOLO model path is invalid."

# Classes
YOLO_CLASSES = ['pistol', 'knife', 'celurit', 'person']
TFLITE_CLASSES = ['Fight', 'No_Fight']

# YOLO Detection Configuration
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# YOLO Processing Interval
YOLO_INTERVAL = 5
