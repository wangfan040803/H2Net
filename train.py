import os
import warnings

warnings.filterwarnings("ignore")
from ultralytics import RTDETR

if __name__ == "__main__":
    model = RTDETR("my_cfg/H2Net.yaml")

    model.train(
        data="dataset/A_drowning_person.yaml",
        cache=True,
        imgsz=640,
        epochs=100,
        batch=4,
        workers=4,
        device="0",
        optimizer="AdamW",
        project="runs/train",
        name="H2Net",
    )
