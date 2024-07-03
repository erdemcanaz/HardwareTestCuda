from ultralytics import YOLO
from pathlib import Path

script_path = Path(__file__).absolute()
script_folder = script_path.parent.absolute()

model_path = script_folder / "yolo_models" / "net_google_mask_28_06_2024.pt"
model = YOLO(model_path)  # pretrained YOLOv8n model
results = model(source=0, vid_stride=1, show=True, save=False, project=None, conf=0.5)  # use webcam as source
