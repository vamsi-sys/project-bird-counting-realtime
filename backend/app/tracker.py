"""
tracker.py
----------
Offline video file analysis. Memory-conscious:
  - processes frames one at a time (no batching)
  - explicitly calls gc.collect() after writing output
  - skips every other frame (frame_skip=2)
  - re-encodes output to H.264 via ffmpeg for browser playback
"""
import gc
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import cv2

from .detector import BirdDetector


def _safe_bbox(box) -> list:
    return [round(float(v), 2) for v in box]


def _reencode_h264(src: Path, dst: Path) -> bool:
    """
    Re-encode mp4v → H.264/yuv420p so every browser can play inline.
    ffmpeg is installed in the Docker image; also works locally with ffmpeg in PATH.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        print("⚠️  ffmpeg not found — video may not play in browser")
        return False

    cmd = [
        ffmpeg_bin, "-y",
        "-i",        str(src),
        "-vcodec",   "libx264",
        "-preset",   "ultrafast",   # faster than 'fast', lower CPU/memory spike
        "-crf",      "28",          # slightly lower quality = smaller file = less RAM
        "-pix_fmt",  "yuv420p",     # mandatory for browser compat
        "-movflags", "+faststart",  # allows streaming before full download
        "-an",                      # no audio track
        str(dst),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode != 0:
            print(f"❌ ffmpeg stderr:\n{result.stderr.decode(errors='replace')}")
            return False
        print(f"✅ ffmpeg H.264 encode → {dst.name}")
        return True
    except subprocess.TimeoutExpired:
        print("❌ ffmpeg timed out after 600s")
        return False
    except Exception as exc:
        print(f"❌ ffmpeg exception: {exc}")
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

        out_dir    = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_path   = out_dir / "annotated_tmp.mp4"
        final_path = out_dir / "annotated_video.mp4"

        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        print(f"📹 Analyzing: {w}x{h} @ {fps:.1f}fps  file={Path(video_path).name}")

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

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                cur_sec = int(frame_idx / fps)

                # Record bird count snapshot every 5 seconds
                if cur_sec - last_snap_sec >= 5:
                    counts_over_time.append({
                        "time_sec": int(cur_sec),
                        "count":    int(len(unique_ids)),
                    })
                    last_snap_sec = cur_sec

                # Skip frames to reduce CPU/memory load
                if frame_idx % self.frame_skip != 0:
                    writer.write(frame)
                    del frame
                    continue

                processed += 1
                annotated, _ = self.detector.process_frame(frame, unique_ids, bird_boxes)
                writer.write(annotated)
                del frame, annotated

        except Exception as exc:
            print(f"❌ Frame error at frame {frame_idx}: {exc}\n{traceback.format_exc()}")
            raise RuntimeError(f"Video processing failed at frame {frame_idx}: {exc}")
        finally:
            cap.release()
            writer.release()
            gc.collect()   # free OpenCV/numpy memory before ffmpeg runs

        print(f"✅ Done: {processed} frames processed, {len(unique_ids)} unique birds")

        # Re-encode to H.264 for browser playback
        ok = _reencode_h264(tmp_path, final_path)
        if ok:
            tmp_path.unlink(missing_ok=True)
        elif tmp_path.exists():
            tmp_path.rename(final_path)

        gc.collect()

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
