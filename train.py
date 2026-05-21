import sys
import os
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore", message=".*adaptive_max_pool2d_backward_cuda does not have a deterministic implementation.*")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # to fix macOS issue
if __name__ == "__main__":

    model = YOLO(model='ultralytics/cfg/models/11/yolo11test.yaml', task='detect')

    for i, layer in enumerate(model.model.model):
        print(f"Layer {i}: {layer.type.split('.')[-1]}, f={layer.f}")
    model.train(data="ultralytics/cfg/datasets/VisDrone.yaml",
                epochs=300, batch=16, imgsz=640, name="yolo11test",deterministic=False, )  # Pass the rest of the command-line arguments to the train metho