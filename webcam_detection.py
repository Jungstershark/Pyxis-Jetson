from ultralytics import YOLO
import supervision as sv
import cv2
from constants import NAMES_PROMPT


model = YOLO("yoloe-11l-seg.pt")

# Custom Prompt Classes
names = NAMES_PROMPT
model.set_classes(names, model.get_text_pe(names))

# Annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("🎥 Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOE expects RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run Prediction
    results = model.predict(rgb_frame, imgsz=1024, conf=0.15, iou=0.5, verbose=False)

    # Convert to Supervision format
    detections = sv.Detections.from_ultralytics(results[0])

    # Annotate
    annotated_frame = box_annotator.annotate(scene=rgb_frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    # Convert back to BGR for OpenCV display
    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Display
    cv2.imshow("YOLOE Pilot Ladder Detection", annotated_frame_bgr)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
