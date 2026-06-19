"""
detector.py
-----------
Pure onnxruntime inference. Zero PyTorch. Zero ultralytics.
Dependencies: onnxruntime, opencv-python-headless, numpy, scipy

RAM usage on Render free tier:
  onnxruntime session : ~80 MB
  ONNX model weights  : ~12 MB
  Per-frame numpy     :  ~1 MB
  Total               : ~95 MB  (free tier limit = 512 MB)
"""

import gc
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .weight import WeightEstimator

MODEL_PATH  = Path(__file__).parent.parent / "models" / "yolov8n.onnx"
INFER_SIZE  = 320
CONF_THRESH = 0.35
IOU_THRESH  = 0.45
BIRD_CLS    = 14     # COCO class index for "bird"

_COLOURS = [
    (0, 255, 0),   (0, 200, 255), (255, 100, 0), (180, 0, 255),
    (0, 140, 255), (255, 0, 130), (0, 255, 180), (255, 200, 0),
]

def _colour(tid): return _COLOURS[int(tid) % len(_COLOURS)]
def _safe(box):   return [round(float(v), 2) for v in box]


# ── preprocessing ──────────────────────────────────────────────────────────────

def _letterbox(img, size):
    h, w   = img.shape[:2]
    scale  = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    resized = cv2.resize(img, (nw, nh))
    ph, pw = (size - nh) // 2, (size - nw) // 2
    canvas[ph:ph+nh, pw:pw+nw] = resized
    return canvas, scale, pw, ph


# ── NMS ────────────────────────────────────────────────────────────────────────

def _nms(boxes, scores, thresh):
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas  = (x2-x1) * (y2-y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= thresh)[0] + 1]
    return keep


# ── simple IoU tracker ─────────────────────────────────────────────────────────

class _Tracker:
    def __init__(self, max_lost=8, iou_thresh=0.3):
        self.max_lost   = max_lost
        self.iou_thresh = iou_thresh
        self._nxt       = 1
        self._tracks    = {}   # id → {bbox, lost}

    def _iou(self, a, b):
        ix1 = max(a[0],b[0]); iy1 = max(a[1],b[1])
        ix2 = min(a[2],b[2]); iy2 = min(a[3],b[3])
        inter = max(0,ix2-ix1)*max(0,iy2-iy1)
        ua = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
        return inter/max(ua,1e-6)

    def update(self, dets):
        for t in self._tracks.values(): t["lost"] += 1
        used, out = set(), []
        for d in dets:
            best_iou, best_id = self.iou_thresh, None
            for tid, trk in self._tracks.items():
                if tid in used: continue
                iou = self._iou(d, trk["bbox"])
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            if best_id is not None:
                self._tracks[best_id].update({"bbox": d, "lost": 0})
                used.add(best_id); out.append((best_id, d))
            else:
                tid = self._nxt; self._nxt += 1
                self._tracks[tid] = {"bbox": d, "lost": 0}
                out.append((tid, d))
        stale = [t for t,v in self._tracks.items() if v["lost"] > self.max_lost]
        for t in stale: del self._tracks[t]
        return out


# ── detector ───────────────────────────────────────────────────────────────────

class BirdDetector:
    def __init__(self):
        self.session    = None
        self.inp_name   = None
        self.out_name   = None
        self.tracker    = _Tracker()
        self.weight_est = WeightEstimator()

    def load(self):
        if self.session:
            return self
        print(f"📦 Loading ONNX model …")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads        = 2
        opts.inter_op_num_threads        = 1
        opts.graph_optimization_level    = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode              = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session  = ort.InferenceSession(
            str(MODEL_PATH), sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.inp_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name

        # warm-up
        dummy = np.zeros((1, 3, INFER_SIZE, INFER_SIZE), dtype=np.float32)
        self.session.run([self.out_name], {self.inp_name: dummy})
        del dummy; gc.collect()
        print("✅ Model ready")
        return self

    def _detect(self, frame):
        oh, ow = frame.shape[:2]
        blob, scale, pw, ph = _letterbox(frame, INFER_SIZE)
        inp = blob[:,:,::-1].transpose(2,0,1)[np.newaxis].astype(np.float32) / 255.0
        raw = self.session.run([self.out_name], {self.inp_name: inp})[0]
        del inp, blob

        preds = raw[0].T                          # (8400, 84)
        scores = preds[:,4:].max(axis=1)
        cls_id = preds[:,4:].argmax(axis=1)
        mask   = (cls_id == BIRD_CLS) & (scores >= CONF_THRESH)
        if not mask.any():
            return []

        cx,cy,w,h = preds[mask,0], preds[mask,1], preds[mask,2], preds[mask,3]
        boxes  = np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], axis=1)
        keep   = _nms(boxes, scores[mask], IOU_THRESH)
        boxes  = boxes[keep]

        results = []
        for b in boxes:
            x1 = max(0, (b[0]-pw)/scale)
            y1 = max(0, (b[1]-ph)/scale)
            x2 = min(ow, (b[2]-pw)/scale)
            y2 = min(oh, (b[3]-ph)/scale)
            if x2>x1 and y2>y1:
                results.append([x1, y1, x2, y2])
        return results

    def process_frame(self, frame, unique_ids, bird_boxes):
        dets    = self._detect(frame)
        tracked = self.tracker.update(dets)

        detections = []
        for tid, box in tracked:
            bird_boxes[tid] = box
            unique_ids.add(tid)
            x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
            col = _colour(tid)
            cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
            cv2.putText(frame, f"#{tid}", (x1, max(y1-6,12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            detections.append({
                "id":       int(tid),
                "bbox":     _safe(box),
                "weight_g": self.weight_est.estimate(box),
            })

        cv2.rectangle(frame, (8,6), (230,46), (0,0,0), -1)
        cv2.putText(frame, f"Unique birds: {len(unique_ids)}",
                    (14,36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,80), 2)
        return frame, detections


# ── singleton ──────────────────────────────────────────────────────────────────
_inst = None

def get_detector():
    global _inst
    if _inst is None:
        _inst = BirdDetector()
        _inst.load()
    return _inst
