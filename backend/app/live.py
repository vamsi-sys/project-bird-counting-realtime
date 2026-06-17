"""
live.py
-------
WebSocket handler for:
  1. Browser webcam  — browser sends binary JPEG frames, server returns annotated JPEG
  2. RTSP/IP camera  — server opens RTSP stream with OpenCV, pushes annotated frames

Binary frames (not Base64) are used to reduce payload by ~33% and remove
CPU overhead of base64 encoding/decoding.

Message protocol
----------------
Browser → Server  (TEXT):
  {"action": "start_webcam"}
  {"action": "start_rtsp", "url": "rtsp://..."}
  {"action": "stop"}

Browser → Server  (BINARY):
  Raw JPEG bytes  (webcam frame)

Server → Browser  (TEXT):
  {"type": "frame", "stats": {...}}        ← stats-only message
  {"type": "error",   "message": "..."}
  {"type": "stopped"}

Server → Browser  (BINARY):
  Raw annotated JPEG bytes                 ← the video frame
"""

import asyncio
import gc
import json
import threading
import time
from collections import deque

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from .detector import get_detector


# ── frame encode/decode ────────────────────────────────────────────────────────

def _encode_binary(frame: np.ndarray, quality: int = 70) -> bytes:
    """Encode BGR frame → JPEG bytes (binary, not base64)."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _decode_binary(data: bytes) -> "np.ndarray | None":
    """Decode raw JPEG bytes → BGR frame."""
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _stats_payload(unique_ids: set, detections: list, elapsed: float,
                   bird_boxes: dict, counts_ts: deque, det) -> str:
    """Build JSON-safe stats string."""
    weights = [det.weight_est.estimate(b) for b in bird_boxes.values()]
    w_stats = det.weight_est.aggregate(weights)
    return json.dumps({
        "type": "stats",
        "stats": {
            "unique_birds":       int(len(unique_ids)),
            "current_detections": int(len(detections)),
            "elapsed_sec":        round(float(elapsed), 1),
            "weight_estimation":  w_stats,
            "counts_over_time":   list(counts_ts)[-20:],
        },
    })


# ── RTSP background streamer ───────────────────────────────────────────────────

class RTSPStreamer:
    """
    Opens a camera source (RTSP URL or device index) in a daemon thread.
    Puts (jpeg_bytes, stats_json) tuples into an asyncio queue
    that the WebSocket coroutine drains.
    """

    def __init__(self, source, frame_skip: int = 3):
        self.source     = source
        self.frame_skip = frame_skip
        self._stop      = threading.Event()
        self._thread    = None
        self._queue: "asyncio.Queue | None" = None
        self._loop: "asyncio.AbstractEventLoop | None" = None

    def start(self, loop: asyncio.AbstractEventLoop):
        self._loop   = loop
        self._queue  = asyncio.Queue(maxsize=3)   # drop old frames if queue full
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=6)

    def _put(self, item):
        """Non-blocking put into the asyncio queue from the background thread."""
        try:
            asyncio.run_coroutine_threadsafe(
                self._queue.put_nowait(item), self._loop
            )
        except asyncio.QueueFull:
            pass   # drop frame — client is too slow; keeps latency low

    def _run(self):
        det        = get_detector()
        unique_ids: set  = set()
        bird_boxes: dict = {}
        counts_ts        = deque(maxlen=200)
        frame_idx        = 0
        t0               = time.time()

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self._put(("error", json.dumps({
                "type": "error",
                "message": f"Cannot open stream: {self.source}",
            })))
            return

        try:
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % self.frame_skip != 0:
                    del frame
                    continue

                annotated, detections = det.process_frame(frame, unique_ids, bird_boxes)
                del frame

                elapsed = round(float(time.time() - t0), 1)
                counts_ts.append({"time_sec": elapsed, "count": int(len(unique_ids))})

                jpeg_bytes = _encode_binary(annotated)
                del annotated

                stats_json = _stats_payload(
                    unique_ids, detections, elapsed, bird_boxes, counts_ts, det
                )
                self._put(("frame", jpeg_bytes, stats_json))
                gc.collect()

        finally:
            cap.release()
            self._put(("stopped", json.dumps({"type": "stopped"})))

    async def get(self):
        return await self._queue.get()


# ── WebSocket entry point ──────────────────────────────────────────────────────

async def handle_websocket(ws: WebSocket):
    await ws.accept()

    streamer: "RTSPStreamer | None" = None
    unique_ids: set  = set()
    bird_boxes: dict = {}
    counts_ts        = deque(maxlen=200)
    t0               = time.time()
    pending_frame    = False   # simple backpressure — don't stack frames

    try:
        while True:
            msg = await ws.receive()

            # ── binary frame from browser webcam ──────────────────────
            if msg.get("type") == "websocket.receive" and msg.get("bytes"):
                if pending_frame:
                    continue   # skip if still processing previous frame

                jpeg_bytes = msg["bytes"]
                frame = _decode_binary(jpeg_bytes)
                if frame is None:
                    continue

                pending_frame = True
                det = get_detector()
                annotated, detections = det.process_frame(frame, unique_ids, bird_boxes)
                del frame

                elapsed = round(float(time.time() - t0), 1)
                counts_ts.append({"time_sec": elapsed, "count": int(len(unique_ids))})

                out_jpeg = _encode_binary(annotated)
                del annotated

                stats_json = _stats_payload(
                    unique_ids, detections, elapsed, bird_boxes, counts_ts, det
                )

                # Send frame binary first, then stats JSON
                await ws.send_bytes(out_jpeg)
                await ws.send_text(stats_json)
                pending_frame = False
                gc.collect()

            # ── text control messages ──────────────────────────────────
            elif msg.get("type") == "websocket.receive" and msg.get("text"):
                data   = json.loads(msg["text"])
                action = data.get("action", "")

                if action in ("start_rtsp", "start_webcam"):
                    if streamer:
                        streamer.stop()
                    unique_ids.clear(); bird_boxes.clear(); counts_ts.clear()
                    t0 = time.time()

                    source = data.get("url", 0) if action == "start_rtsp" else 0
                    streamer = RTSPStreamer(source)
                    streamer.start(asyncio.get_event_loop())
                    asyncio.create_task(_forward_rtsp(ws, streamer))

                elif action == "stop":
                    if streamer:
                        streamer.stop()
                        streamer = None
                    unique_ids.clear(); bird_boxes.clear(); counts_ts.clear()
                    await ws.send_text(json.dumps({"type": "stopped"}))

            elif msg.get("type") == "websocket.disconnect":
                break

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
        gc.collect()


async def _forward_rtsp(ws: WebSocket, streamer: RTSPStreamer):
    """Drain the RTSP streamer queue and forward to the WebSocket client."""
    try:
        while True:
            item = await streamer.get()
            kind = item[0]
            if kind == "frame":
                _, jpeg_bytes, stats_json = item
                await ws.send_bytes(jpeg_bytes)
                await ws.send_text(stats_json)
            elif kind in ("stopped", "error"):
                _, payload = item
                await ws.send_text(payload)
                break
    except Exception:
        pass
