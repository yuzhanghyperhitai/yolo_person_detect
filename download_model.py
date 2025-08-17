import torch 

print(f"Pytorch version: {torch.version.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
   print(f"CUDA version: {torch.version.cuda}")
   print(f"GPU name: {torch.cuda.get_device_name()}")

from ultralytics import YOLO

model = YOLO('yolov8x.pt')