import asyncio
import base64
import json
import threading
import time
from collections import deque

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .detector import get_detector


# ── helpers ────────────────────────────────────────────────────────────────────

def _encode(frame: np.ndarray, quality: int = 75) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


def _decode(b64: str):
    data = base64.b64decode(b64)
    arr  = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _build_stats(unique_ids: set, detections: list, elapsed: float,
                 bird_boxes: dict, counts_ts: deque, det) -> dict:
    """Build a JSON-safe stats dict (all plain Python types)."""
    weights = [det.weight_est.estimate(b) for b in bird_boxes.values()]
    w_stats = det.weight_est.aggregate(weights)
    return {
        "unique_birds":       int(len(unique_ids)),
        "current_detections": int(len(detections)),
        "elapsed_sec":        round(float(elapsed), 1),
        "weight_estimation":  w_stats,
        "counts_over_time":   list(counts_ts)[-20:],
    }


# ── RTSP streamer (background thread) ─────────────────────────────────────────

class RTSPStreamer:
    def __init__(self, source, frame_skip: int = 2):
        self.source     = source
        self.frame_skip = frame_skip
        self._stop      = threading.Event()
        self._thread    = None
        self._queue     = None
        self._loop      = None

    def start(self, loop: asyncio.AbstractEventLoop):
        self._loop   = loop
        self._queue  = asyncio.Queue(maxsize=4)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _put(self, msg: str):
        try:
            asyncio.run_coroutine_threadsafe(
                self._queue.put_nowait(msg), self._loop
            )
        except asyncio.QueueFull:
            pass

    def _run(self):
        det        = get_detector()
        unique_ids: set  = set()
        bird_boxes: dict = {}
        counts_ts        = deque(maxlen=200)
        frame_idx        = 0
        t0               = time.time()

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self._put(json.dumps({
                "type":    "error",
                "message": f"Cannot open source: {self.source}",
            }))
            return

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % self.frame_skip != 0:
                continue

            annotated, detections = det.process_frame(frame, unique_ids, bird_boxes)
            elapsed = round(float(time.time() - t0), 1)
            counts_ts.append({"time_sec": elapsed, "count": int(len(unique_ids))})

            stats = _build_stats(unique_ids, detections, elapsed, bird_boxes, counts_ts, det)
            self._put(json.dumps({"type": "frame", "data": _encode(annotated), "stats": stats}))

        cap.release()
        self._put(json.dumps({"type": "stopped"}))

    async def get(self) -> str:
        return await self._queue.get()


# ── WebSocket handler ──────────────────────────────────────────────────────────

async def handle_websocket(ws: WebSocket):
    await ws.accept()

    streamer: RTSPStreamer | None = None
    unique_ids: set  = set()
    bird_boxes: dict = {}
    counts_ts        = deque(maxlen=200)
    t0               = time.time()

    try:
        while True:
            raw    = await ws.receive_text()
            msg    = json.loads(raw)
            action = msg.get("action", "")

            # ── start RTSP or server-side webcam ──────────────────────
            if action in ("start_rtsp", "start_webcam"):
                if streamer:
                    streamer.stop()
                unique_ids.clear(); bird_boxes.clear(); counts_ts.clear()
                t0 = time.time()

                source   = msg.get("url", 0) if action == "start_rtsp" else 0
                streamer = RTSPStreamer(source)
                streamer.start(asyncio.get_event_loop())
                asyncio.create_task(_forward(ws, streamer))

            # ── browser sends webcam frames ────────────────────────────
            elif action == "frame":
                b64 = msg.get("data", "")
                if not b64:
                    continue
                frame = _decode(b64)
                if frame is None:
                    continue

                det = get_detector()
                annotated, detections = det.process_frame(frame, unique_ids, bird_boxes)

                elapsed = round(float(time.time() - t0), 1)
                counts_ts.append({"time_sec": elapsed, "count": int(len(unique_ids))})

                stats = _build_stats(unique_ids, detections, elapsed, bird_boxes, counts_ts, det)
                await ws.send_text(json.dumps({
                    "type":  "frame",
                    "data":  _encode(annotated),
                    "stats": stats,
                }))

            # ── stop ───────────────────────────────────────────────────
            elif action == "stop":
                if streamer:
                    streamer.stop()
                    streamer = None
                unique_ids.clear(); bird_boxes.clear(); counts_ts.clear()
                await ws.send_text(json.dumps({"type": "stopped"}))

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
    finally:
        if streamer:
            streamer.stop()


async def _forward(ws: WebSocket, streamer: RTSPStreamer):
    try:
        while True:
            msg  = await streamer.get()
            await ws.send_text(msg)
            if json.loads(msg).get("type") in ("stopped", "error"):
                break
    except Exception:
        pass
