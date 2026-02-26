from ultralytics import YOLO
import supervision as sv
import cv2

# Pick ONE:
MODEL_PATH = r"weights/y11m_t0_960/best.ptt"

model = YOLO(MODEL_PATH)

# Annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("🎥 Webcam started. Press 'q' to quit.")

# Optional: if your model has class names inside
names = model.names  # dict: {id: "class_name"}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv11 can take BGR directly, but we'll stay consistent and use RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Predict
    results = model.predict(
        rgb_frame,
        imgsz=960,     # match your training/eval
        conf=0.25,     # start at your eval conf; adjust live if needed
        iou=0.5,
        verbose=False
    )

    det = sv.Detections.from_ultralytics(results[0])

    # Build labels like: "pilot_ladder 0.82"
    labels = []
    for class_id, conf in zip(det.class_id, det.confidence):
        class_name = names.get(int(class_id), str(class_id)) if class_id is not None else "obj"
        labels.append(f"{class_name} {conf:.2f}")

    annotated = box_annotator.annotate(scene=rgb_frame.copy(), detections=det)
    annotated = label_annotator.annotate(scene=annotated, detections=det, labels=labels)

    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imshow("YOLOv11 Pilot Ladder Detection", annotated_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()