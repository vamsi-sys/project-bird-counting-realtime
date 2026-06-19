"""
detector.py
-----------
Bird detector using YOLOv8n exported to ONNX format.
Inference via onnxruntime — no PyTorch required at runtime.

Memory profile on Render free tier (512 MB):
  - onnxruntime session:  ~80 MB
  - ONNX model weights:   ~12 MB
  - Per-frame numpy:      ~1 MB
  Total:                 ~95 MB  (vs ~450 MB with PyTorch)

Tracking: ByteTrack via ultralytics tracker (CPU, no GPU needed).
"""

import gc
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .weight import WeightEstimator

MODEL_PATH = Path(__file__).parent.parent / "models" / "yolov8n.onnx"

# Input size must match export imgsz (320)
INFER_SIZE = 320
CONF_THRESH = 0.35
IOU_THRESH  = 0.45

_COLOURS = [
    (0, 255, 0),   (0, 200, 255), (255, 100, 0), (180, 0, 255),
    (0, 140, 255), (255, 0, 130), (0, 255, 180), (255, 200, 0),
]


def _colour(tid: int):
    return _COLOURS[int(tid) % len(_COLOURS)]


def _safe_bbox(box) -> list:
    return [round(float(v), 2) for v in box]


def _letterbox(img: np.ndarray, size: int):
    """Resize keeping aspect ratio, pad to square."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    pad_h = (size - nh) // 2
    pad_w = (size - nw) // 2
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[pad_h:pad_h+nh, pad_w:pad_w+nw] = resized
    return out, scale, pad_w, pad_h


def _nms(boxes, scores, iou_thresh):
    """Simple CPU NMS."""
    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


class SimpleTracker:
    """
    Lightweight centroid tracker — assigns persistent IDs to detections
    without requiring lap/ByteTrack when ONNX-only mode is used.
    Matches new detections to existing tracks by IoU.
    """
    def __init__(self, max_lost: int = 10, iou_thresh: float = 0.3):
        self.max_lost   = max_lost
        self.iou_thresh = iou_thresh
        self._next_id   = 1
        self._tracks    = {}   # id → {"bbox": ..., "lost": 0}

    def _iou(self, a, b):
        ax1,ay1,ax2,ay2 = a
        bx1,by1,bx2,by2 = b
        ix1 = max(ax1,bx1); iy1 = max(ay1,by1)
        ix2 = min(ax2,bx2); iy2 = min(ay2,by2)
        inter = max(0,ix2-ix1)*max(0,iy2-iy1)
        ua = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
        return inter/max(ua,1e-6)

    def update(self, detections: list) -> list:
        """
        detections: list of [x1,y1,x2,y2]
        returns:    list of (track_id, [x1,y1,x2,y2])
        """
        # Mark all existing tracks as potentially lost
        for tid in list(self._tracks):
            self._tracks[tid]["lost"] += 1

        matched_tids = set()
        results      = []

        for det in detections:
            best_iou = self.iou_thresh
            best_tid = None
            for tid, trk in self._tracks.items():
                if tid in matched_tids:
                    continue
                iou = self._iou(det, trk["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                self._tracks[best_tid]["bbox"] = det
                self._tracks[best_tid]["lost"] = 0
                matched_tids.add(best_tid)
                results.append((best_tid, det))
            else:
                # New track
                tid = self._next_id; self._next_id += 1
                self._tracks[tid] = {"bbox": det, "lost": 0}
                results.append((tid, det))

        # Remove stale tracks
        for tid in [t for t,v in self._tracks.items() if v["lost"] > self.max_lost]:
            del self._tracks[tid]

        return results


class BirdDetector:
    def __init__(self):
        self.session    = None
        self.tracker    = SimpleTracker()
        self.weight_est = WeightEstimator()

    def load(self):
        if self.session is not None:
            return self
        print(f"📦 Loading ONNX model from {MODEL_PATH} …")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1   # limit threads → less RAM
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Warm-up
        dummy = np.zeros((1, 3, INFER_SIZE, INFER_SIZE), dtype=np.float32)
        self.session.run([self.output_name], {self.input_name: dummy})
        del dummy
        gc.collect()
        print("✅ ONNX model loaded and warmed up")
        return self

    def _infer(self, frame: np.ndarray):
        """Run ONNX inference. Returns list of [x1,y1,x2,y2] in original coords."""
        orig_h, orig_w = frame.shape[:2]
        blob, scale, pad_w, pad_h = _letterbox(frame, INFER_SIZE)

        # HWC BGR → NCHW RGB float32
        inp = blob[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        inp = inp[np.newaxis]   # add batch dim

        raw = self.session.run([self.output_name], {self.input_name: inp})[0]
        del inp, blob

        # raw shape: (1, 84, 2100) → transpose to (2100, 84)
        preds = raw[0].T   # (2100, 84)

        # Columns: cx,cy,w,h, then 80 class scores
        cx, cy, w, h = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
        class_scores  = preds[:, 4:]               # (2100, 80)
        class_ids     = class_scores.argmax(axis=1)
        confidences   = class_scores.max(axis=1)

        # Keep only "bird" class (COCO id = 14) above threshold
        mask = (class_ids == 14) & (confidences >= CONF_THRESH)
        if not mask.any():
            return []

        cx = cx[mask]; cy = cy[mask]; w = w[mask]; h = h[mask]
        conf = confidences[mask]

        x1 = cx - w/2; y1 = cy - h/2
        x2 = cx + w/2; y2 = cy + h/2

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep  = _nms(boxes, conf, IOU_THRESH)
        boxes = boxes[keep]

        # Convert from letterboxed coords back to original frame coords
        results = []
        for b in boxes:
            bx1 = max(0, (b[0] - pad_w) / scale)
            by1 = max(0, (b[1] - pad_h) / scale)
            bx2 = min(orig_w, (b[2] - pad_w) / scale)
            by2 = min(orig_h, (b[3] - pad_h) / scale)
            if bx2 > bx1 and by2 > by1:
                results.append([bx1, by1, bx2, by2])

        return results

    def process_frame(self, frame: np.ndarray, unique_ids: set, bird_boxes: dict):
        detections_raw = self._infer(frame)
        tracked        = self.tracker.update(detections_raw)

        detections = []
        for tid, box in tracked:
            bird_boxes[tid] = box
            unique_ids.add(tid)

            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            col = _colour(tid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, f"#{tid}", (x1, max(y1-6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

            detections.append({
                "id":       int(tid),
                "bbox":     _safe_bbox(box),
                "weight_g": self.weight_est.estimate(box),
            })

        # Live counter overlay
        cv2.rectangle(frame, (8, 6), (230, 46), (0, 0, 0), -1)
        cv2.putText(frame, f"Unique birds: {len(unique_ids)}",
                    (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 80), 2)

        return frame, detections


# ── singleton ──────────────────────────────────────────────────────────────────
_instance: "BirdDetector | None" = None


def get_detector() -> BirdDetector:
    global _instance
    if _instance is None:
        _instance = BirdDetector()
        _instance.load()
    return _instance
