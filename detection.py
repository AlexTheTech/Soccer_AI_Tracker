from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    cls: str
    conf: float
    bbox: np.ndarray


class Detector:
    def __init__(self, weights: str, device: str = "cuda", conf: float = 0.35) -> None:
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(frame, conf=self.conf, device=self.device, verbose=False)
        detections: List[Detection] = []
        if not results:
            return detections
        result = results[0]
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls.item())
            cls_name = names.get(cls_id, str(cls_id))
            conf = float(box.conf.item())
            bbox = box.xyxy.cpu().numpy().astype(float).reshape(-1)
            detections.append(Detection(cls=cls_name, conf=conf, bbox=bbox))
        return detections
