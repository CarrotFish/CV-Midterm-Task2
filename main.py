from ultralytics import YOLO
import os

model = YOLO(os.path.abspath('./yolov8x.pt')) 

results = model.train(
    data = 'VisDrone.yaml',
    epochs=20,
    imgsz=640,
    batch=16,
    device=0,
    project='visdrone_project',
    name='yolov8x_finetune',
    workers=0,
    resume=True,
    freeze=22,
    lr0=0.001
)