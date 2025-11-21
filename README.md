Pyxis Jetson

Install pyenv Python version 3.10

To run virtual environment
```bash
pyenv activate myvenv
```

Default to run object detection interference on Webcam using jetson and yolov8n.pt

```bash
yolo detect predict model=yolov8n.pt source=0 show=True
```

pip install supervision pillow