from ultralytics import YOLO
import os

model = YOLO(os.path.abspath('./best.pt')) 

results = model.track(
    source=os.path.abspath('./tracksource.mp4'),
    conf=0.3,
    iou=0.5,
    device='cuda',
    save=True,
    save_txt=True,
    tracker='bytetrack.yaml'
)