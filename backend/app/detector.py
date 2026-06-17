from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from .weight import WeightEstimator

MODEL_PATH = Path(__file__).parent.parent / "models" / "yolov8n.pt"

_COLOURS = [
    (0, 255, 0), (0, 200, 255), (255, 100, 0), (180, 0, 255),
    (0, 140, 255), (255, 0, 130), (0, 255, 180), (255, 200, 0),
]


def _colour(tid: int):
    return _COLOURS[int(tid) % len(_COLOURS)]


def _safe_bbox(box) -> list:
    """Convert any numpy array / scalar bbox to a plain Python float list."""
    return [round(float(v), 2) for v in box]


class BirdDetector:
    def __init__(self, conf: float = 0.40, resize_width: int = 640):
        self.conf         = conf
        self.resize_width = resize_width
        self.model        = None
        self.weight_est   = WeightEstimator()

    def load(self):
        if self.model is None:
            self.model = YOLO(str(MODEL_PATH))
        return self

    def process_frame(
        self,
        frame: np.ndarray,
        unique_ids: set,
        bird_boxes: dict,
    ):
        h, w  = frame.shape[:2]
        scale   = self.resize_width / w
        resized = cv2.resize(frame, (self.resize_width, int(h * scale)))

        results = self.model.track(resized, persist=True, conf=self.conf, verbose=False)

        detections = []

        if results and results[0].boxes.id is not None:
            boxes     = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()

            for box, tid in zip(boxes, track_ids):
                # ── force plain Python int / float ────────────────────
                tid  = int(tid)
                box  = box / scale          # still numpy array; convert below

                bird_boxes[tid] = box       # store numpy array internally
                unique_ids.add(tid)

                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                col = _colour(tid)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(
                    frame, f"Bird {tid}", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2,
                )

                weight = self.weight_est.estimate(box)
                detections.append({
                    "id":       tid,                    # plain int
                    "bbox":     _safe_bbox(box),        # plain float list
                    "weight_g": weight,                 # plain float
                })

        # live counter overlay
        cv2.rectangle(frame, (10, 8), (240, 50), (0, 0, 0), -1)
        cv2.putText(
            frame, f"Unique birds: {len(unique_ids)}",
            (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 80), 2,
        )

        return frame, detections


# ── module-level singleton ─────────────────────────────────────────────────────
_instance: "BirdDetector | None" = None


def get_detector() -> BirdDetector:
    global _instance
    if _instance is None:
        _instance = BirdDetector()
        _instance.load()
    return _instance
