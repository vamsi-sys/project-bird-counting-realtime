"""
detector.py
-----------
Memory-optimised YOLOv8 detector.

Render free tier = 512 MB RAM.
Key optimisations:
  - resize frames to 320px (not 640) during inference → ~4x less memory per frame
  - half=False kept (float16 needs CUDA; CPU must use float32)
  - model loaded once as a singleton; never reloaded
  - numpy arrays released immediately after use
  - no frame copies held in memory between calls
"""

from pathlib import Path

import cv2
import gc
import numpy as np
from ultralytics import YOLO

from .weight import WeightEstimator

MODEL_PATH = Path(__file__).parent.parent / "models" / "yolov8n.pt"

# Colour palette per track ID (BGR)
_COLOURS = [
    (0, 255, 0),   (0, 200, 255), (255, 100, 0), (180, 0, 255),
    (0, 140, 255), (255, 0, 130), (0, 255, 180), (255, 200, 0),
]


def _colour(tid: int):
    return _COLOURS[int(tid) % len(_COLOURS)]


def _safe_bbox(box) -> list:
    """Convert numpy bbox → plain Python float list for JSON serialisation."""
    return [round(float(v), 2) for v in box]


class BirdDetector:
    def __init__(self, conf: float = 0.35, infer_width: int = 320):
        """
        conf        – detection confidence threshold (lower = more detections)
        infer_width – resize width for inference (320 uses ~4x less RAM than 640)
        """
        self.conf        = conf
        self.infer_width = infer_width
        self.model       = None
        self.weight_est  = WeightEstimator()

    def load(self):
        if self.model is None:
            print(f"📦 Loading YOLO model from {MODEL_PATH} …")
            self.model = YOLO(str(MODEL_PATH))
            # Warm-up pass on a tiny blank frame so first real frame isn't slow
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False)
            del dummy
            gc.collect()
            print("✅ YOLO model loaded and warmed up")
        return self

    def process_frame(
        self,
        frame: np.ndarray,
        unique_ids: set,
        bird_boxes: dict,
    ):
        """
        Run YOLOv8 tracking on one frame.
        Mutates unique_ids and bird_boxes in-place.
        Returns (annotated_frame, detections_list).
        All values in detections_list are plain Python types (JSON-safe).
        """
        orig_h, orig_w = frame.shape[:2]
        scale = self.infer_width / orig_w
        infer_h = int(orig_h * scale)

        # Resize for inference (key memory saving)
        small = cv2.resize(frame, (self.infer_width, infer_h))

        results = self.model.track(
            small,
            persist=True,
            conf=self.conf,
            verbose=False,
            imgsz=self.infer_width,
        )

        # Free the small frame immediately
        del small

        detections = []

        if results and results[0].boxes.id is not None:
            # Pull tensors to CPU numpy once, then release GPU memory
            boxes_np = results[0].boxes.xyxy.cpu().numpy()
            ids_np   = results[0].boxes.id.cpu().numpy()

            for box, tid in zip(boxes_np, ids_np):
                tid = int(tid)
                # Scale box back to original resolution
                x1 = int(box[0] / scale)
                y1 = int(box[1] / scale)
                x2 = int(box[2] / scale)
                y2 = int(box[3] / scale)

                orig_box = np.array([x1, y1, x2, y2], dtype=np.float32)
                bird_boxes[tid] = orig_box
                unique_ids.add(tid)

                col = _colour(tid)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(
                    frame, f"#{tid}",
                    (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2,
                )

                detections.append({
                    "id":       tid,
                    "bbox":     _safe_bbox(orig_box),
                    "weight_g": self.weight_est.estimate(orig_box),
                })

            del boxes_np, ids_np

        # Counter overlay (black background for readability)
        cv2.rectangle(frame, (8, 6), (230, 46), (0, 0, 0), -1)
        cv2.putText(
            frame, f"Unique birds: {len(unique_ids)}",
            (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 80), 2,
        )

        return frame, detections


# ── module-level singleton (one model for the whole process) ───────────────────
_instance: "BirdDetector | None" = None


def get_detector() -> BirdDetector:
    global _instance
    if _instance is None:
        _instance = BirdDetector()
        _instance.load()
    return _instance
