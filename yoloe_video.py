# Inspiration: https://blog.roboflow.com/yoloe-zero-shot-object-detection-segmentation/
import supervision as sv
from ultralytics import YOLO
import cv2
from constants import NAMES_PROMPT


model = YOLO("yoloe-11l-seg.pt")

# Custom prompt classes
names = NAMES_PROMPT
model.set_classes(names, model.get_text_pe(names))

SOURCE_VIDEO_PATH = "data/input_video.mp4"
TARGET_VIDEO_PATH = "data/annotated_video.mp4"

# 3. Prepare Video Reader/Writer
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    for frame in frame_generator:

        # YOLOE inference
        results = model.predict(
            frame
        )

        # Convert predictions to Supervision detections
        detections = sv.Detections.from_ultralytics(results[0])

        # Annotation
        annotated = frame.copy()
        annotated = box_annotator.annotate(annotated, detections)
        annotated = label_annotator.annotate(annotated, detections)

        # Write to output video
        sink.write_frame(annotated)

print("Done! Annotated video saved to:", TARGET_VIDEO_PATH)
