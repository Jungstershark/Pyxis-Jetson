# Inspiration from: https://blog.roboflow.com/yoloe-zero-shot-object-detection-segmentation/
from ultralytics import YOLO
import os
from PIL import Image
import supervision as sv
import matplotlib.pyplot as plt
from constants import NAMES_PROMPT


# Initialize a YOLOE model
model = YOLO("yoloe-11l-seg.pt")

# Custom labels through prompt
names = NAMES_PROMPT
model.set_classes(names, model.get_text_pe(names))

RAW_DIR = "data/raw"
OUT_DIR = "data/annotated"
os.makedirs(OUT_DIR, exist_ok=True)

for filename in os.listdir(RAW_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    
    input_path = os.path.join(RAW_DIR, filename)
    output_path = os.path.join(OUT_DIR, filename)

    print(f"Processing: {filename}")

    # Execute prediction on an image
    image = Image.open(input_path)

    results = model.predict(input_path)

    detections = sv.Detections.from_ultralytics(results[0])

    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator().annotate(scene=annotated_image, detections=detections)
    annotated_image = sv.LabelAnnotator().annotate(scene=annotated_image, detections=detections)

    annotated_image.save(output_path)

print("Done! All images annotated")

# plt.figure(figsize=(10,10))
# plt.imshow(annotated_image)
# plt.axis("off")
# plt.show()