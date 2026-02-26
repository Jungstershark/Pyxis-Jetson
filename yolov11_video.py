import supervision as sv
from ultralytics import YOLO
import cv2

# 1) Load your fine-tuned YOLOv11 checkpoint
# e.g. "model_finetuning/w"
MODEL_PATH = "weights/y11m_t0_960/best.pt"
model = YOLO(MODEL_PATH)

SOURCE_VIDEO_PATH = "data/input_video.mp4"
TARGET_VIDEO_PATH = "data/annotated_y11_video.mp4"

# 2) Video reader/writer
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()  # uses detections.class_id to map into model.names

with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    for frame in frame_generator:
        # 3) Inference
        results = model.predict(
            frame,
            conf=0.25,     # tune
            iou=0.7,       # tune
            verbose=False
        )

        # 4) Convert to supervision detections
        detections = sv.Detections.from_ultralytics(results[0])

        # 5) Annotate
        annotated = frame.copy()
        annotated = box_annotator.annotate(annotated, detections)
        annotated = label_annotator.annotate(annotated, detections)

        sink.write_frame(annotated)

print("Done! YOLOv11 annotated video saved to:", TARGET_VIDEO_PATH)