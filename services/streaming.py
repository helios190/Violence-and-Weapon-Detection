import cv2
import json
import base64
import asyncio
import gc
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from config import YOLO_INTERVAL  # Assuming you still want to import it if needed elsewhere
from detectors.object_detector import detect_objects
from detectors.violence_detector import detect_violence, init_states
from detectors.utils import non_max_suppression

# Initialize thread pools for concurrent processing
yolo_executor = ThreadPoolExecutor(max_workers=2)
violence_executor = ThreadPoolExecutor(max_workers=2)

# Frame skipping parameter
SKIP_FRAMES = 2  # Adjust to control detection frequency

# Helper function to determine status based on detection results
def determine_status(group, weapon_availability, weapon_range, anomaly):
    if group == 1 and weapon_availability == 1 and weapon_range == 2 and anomaly == 1:
        return 2  # Danger
    elif group == 1 and weapon_availability == 1 and weapon_range == 2 and anomaly == 0:
        return 1  # Warning
    elif group == 1 and weapon_availability == 1 and weapon_range == 1 and anomaly == 1:
        return 2  # Danger
    elif group == 1 and weapon_availability == 1 and weapon_range == 1 and anomaly == 0:
        return 1  # Warning
    elif group == 1 and weapon_availability == 0 and anomaly == 1:
        return 1  # Warning
    elif group == 1 and weapon_availability == 0 and anomaly == 0:
        return 0  # Normal
    elif group == 0 and weapon_availability == 1 and weapon_range == 2 and anomaly == 1:
        return 2  # Danger
    elif group == 0 and weapon_availability == 1 and weapon_range == 2 and anomaly == 0:
        return 1  # Warning
    elif group == 0 and weapon_availability == 1 and weapon_range == 1 and anomaly == 1:
        return 1  # Warning
    elif group == 0 and weapon_availability == 1 and weapon_range == 1 and anomaly == 0:
        return 1  # Warning
    elif group == 0 and weapon_availability == 0 and anomaly == 1:
        return 1  # Warning
    elif group == 0 and weapon_availability == 0 and anomaly == 0:
        return 0  # Normal
    else:
        return 0  # Normal

# Main detection stream generator function
async def detection_stream():
    
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():
        yield f"data: {json.dumps({'error': 'Could not open video source.'})}\n\n"
        return

    states = init_states
    frame_counter = 0

    # Recording settings
    is_recording = False
    record_start_time = None
    record_frames = []
    RECORD_DURATION = 20  # seconds
    FPS = 10  # Frame rate for saved video
    FRAME_INTERVAL = 1 / FPS

    # Scale factors for drawing bounding boxes from small_frame to original frame
    scale_x = 320 / 320  # width scaling factor
    scale_y = 240 / 240  # height scaling factor

    while True:
        ret, frame = cap.read()
        if not ret:
            yield f"data: {json.dumps({'error': 'Failed to grab frame.'})}\n\n"
            break

        # Resize frame for faster processing (reduced resolution)
        small_frame = cv2.resize(frame, (320, 240))

        # Run detection only on every SKIP_FRAMES frame to save processing power
        if frame_counter % SKIP_FRAMES == 0:
            # Run YOLO detection in a separate thread
            detections = await asyncio.get_running_loop().run_in_executor(yolo_executor, detect_objects, small_frame)

            # Violence detection and probabilities (also in a separate thread)
            resized_for_violence = cv2.resize(frame, (172, 172))
            anomaly, states, fight_prob = await asyncio.get_running_loop().run_in_executor(violence_executor, detect_violence, resized_for_violence, states)
            
            # Initialize counts and flags for detected objects
            person_count, celurit_count, pisau_count, pistol_count, weapon_count, group = 0, 0, 0, 0, 0, 0
            detections_info = []
            raw_person_boxes = []

            for detection in detections:
                label = detection["label"]
                confidence = detection["confidence"]
                x, y, w, h = detection["box"]

                # Scale bounding box coordinates to match the original frame size
                x = int(x * scale_x)
                y = int(y * scale_y)
                w = int(w * scale_x)
                h = int(h * scale_y)

                if label == 'person':
                    person_count += 1
                    raw_person_boxes.append(detection)
                elif label == 'pistol':
                    pistol_count += 1
                    weapon_count += 1
                elif label == 'celurit':
                    celurit_count += 1
                    weapon_count += 1
                elif label == 'knife':
                    pisau_count += 1
                    weapon_count += 1

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                detections_info.append({"label": label, "confidence": confidence})

            # Non-max suppression to reduce overlapping detections
            grouped_boxes = non_max_suppression(raw_person_boxes, iou_threshold=0.3)
            group = 1 if len(grouped_boxes) > 1 else 0

            # Display fight probabilities on frame
            fight_prob_0, fight_prob_1 = float(fight_prob[0]), float(fight_prob[1])
            cv2.putText(frame, f"Fight Probability: {fight_prob_0:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Non-Fight Probability: {fight_prob_1:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Determine weapon range type
            weapon_availability = 1 if weapon_count > 0 else 0
            weapon_range = 2 if pistol_count > 0 else (1 if celurit_count > 0 or pisau_count > 0 else 0)

            # Determine status
            status = determine_status(group, weapon_availability, weapon_range, anomaly)

            # Start recording if status is Warning or Danger
            if status > 0 and not is_recording:
                is_recording = True
                record_start_time = datetime.now()
                record_frames = []

            # Only yield JSON if status is Warning (1) or Danger (2)
            if status > 0:
                _, jpeg_frame = cv2.imencode('.jpg', frame)
                jpeg_base64 = base64.b64encode(jpeg_frame).decode('utf-8')
                timestamp = datetime.utcnow().isoformat()

                # Yield JSON output with detection data
                yield json.dumps({
                    "timestamp": timestamp,
                    "frame": frame_counter,
                    "status": status,
                    "group": group,
                    "persons": person_count,
                    "celurit": celurit_count,
                    "pisau": pisau_count,
                    "pistol": pistol_count,
                    "weapons": weapon_count,
                    "anomaly": anomaly,
                    "detections": detections_info
                })

        # Append frame to record_frames if currently recording
        if is_recording:
            record_frames.append(frame)

            # If recording has reached the duration limit, save video
            if (datetime.now() - record_start_time).total_seconds() >= RECORD_DURATION:
                video_filename = f"record_{record_start_time.strftime('%Y%m%d_%H%M%S')}.mp4"
                video_path = os.path.join("recordings", video_filename)
                os.makedirs("recordings", exist_ok=True)

                # Define the video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, FPS, (frame.shape[1], frame.shape[0]))

                for recorded_frame in record_frames:
                    out.write(recorded_frame)

                out.release()  # Close the video writer

                # Reset recording flags and buffer
                is_recording = False
                record_frames = []

        frame_counter += 1
        if frame_counter % 100 == 0:
            states = init_states
            gc.collect()

        # Display the frame with annotations
        cv2.imshow("Detection Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        await asyncio.sleep(FRAME_INTERVAL)  # Control FPS for display and processing

    cap.release()
    cv2.destroyAllWindows()
