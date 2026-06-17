import subprocess
import time
from pathlib import Path

import cv2

from .detector import BirdDetector


def _safe_bbox(box) -> list:
    return [round(float(v), 2) for v in box]


def _reencode_h264(src: Path, dst: Path) -> bool:
    """
    Re-encode src (mp4v) → dst (H.264/AAC mp4) using ffmpeg so browsers
    can play it inline. Returns True on success, False if ffmpeg missing.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(src),
                "-vcodec", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",   # required for browser compat
                "-movflags", "+faststart",  # allow streaming before full download
                "-an",                    # no audio track
                str(dst),
            ],
            capture_output=True,
            timeout=300,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class VideoAnalyzer:
    def __init__(self, frame_skip: int = 2):
        self.frame_skip = frame_skip
        self.detector   = BirdDetector()

    def load(self):
        self.detector.load()

    def analyze(self, video_path: str, output_dir: str) -> dict:
        self.load()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # write raw frames to a temp file first
        tmp_path   = out_dir / "annotated_tmp.mp4"
        final_path = out_dir / "annotated_video.mp4"

        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        writer = cv2.VideoWriter(
            str(tmp_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (w, h),
        )

        frame_idx        = 0
        processed        = 0
        unique_ids: set  = set()
        bird_boxes: dict = {}
        counts_over_time = []
        last_snap_sec    = 0
        t0               = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            cur_sec = int(frame_idx / fps)
            if cur_sec - last_snap_sec >= 5:
                counts_over_time.append({
                    "time_sec": int(cur_sec),
                    "count":    int(len(unique_ids)),
                })
                last_snap_sec = cur_sec

            if frame_idx % self.frame_skip != 0:
                writer.write(frame)
                continue

            processed += 1
            annotated, _ = self.detector.process_frame(frame, unique_ids, bird_boxes)
            writer.write(annotated)

        cap.release()
        writer.release()

        # ── re-encode to H.264 for browser playback ──────────────────
        encoded = _reencode_h264(tmp_path, final_path)
        if encoded:
            tmp_path.unlink(missing_ok=True)   # remove temp file
        else:
            # ffmpeg not available — just rename temp as final (may not play inline)
            tmp_path.rename(final_path)

        # ── weights ───────────────────────────────────────────────────
        weights = [self.detector.weight_est.estimate(b) for b in bird_boxes.values()]
        w_stats = self.detector.weight_est.aggregate(weights)

        tracks_sample = [
            {"id": int(tid), "bbox": _safe_bbox(bbox)}
            for tid, bbox in list(bird_boxes.items())[:10]
        ]

        elapsed = time.time() - t0

        return {
            "frames_processed":    int(processed),
            "unique_birds":        int(len(unique_ids)),
            "counts_over_time":    counts_over_time,
            "tracks_sample":       tracks_sample,
            "weight_estimation":   w_stats,
            "processing_time_sec": round(float(elapsed), 2),
            "fps":                 round(float(processed / max(elapsed, 0.001)), 2),
        }
