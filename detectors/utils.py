def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    intersection_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area != 0 else 0

def non_max_suppression(detections, iou_threshold):
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    suppressed_boxes = []
    while detections:
        chosen_box = detections.pop(0)
        suppressed_boxes.append(chosen_box)
        detections = [box for box in detections if calculate_iou(chosen_box["box"], box["box"]) < iou_threshold]
    return suppressed_boxes
