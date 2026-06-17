"""
main.py — FastAPI application entry point
"""
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .detector import get_detector
from .live import handle_websocket
from .tracker import VideoAnalyzer

# ── directories ────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent.parent
STATIC = BASE / "static"
UPLOAD = BASE / "uploads"
OUTPUT = BASE / "outputs"

for d in (STATIC, UPLOAD, OUTPUT):
    d.mkdir(parents=True, exist_ok=True)

# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Bird Counting & Weight Estimation",
    description="YOLOv8 · WebSocket · RTSP · Webcam · Offline video",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = VideoAnalyzer()


# ── startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def warmup():
    print("⏳  Loading YOLOv8 model …")
    get_detector()
    print("✅  Model ready.")


# ── static frontend files ──────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def serve_index():
    f = STATIC / "index.html"
    if not f.exists():
        raise HTTPException(404, "index.html not found in backend/static/")
    return FileResponse(str(f))


@app.get("/app.js", include_in_schema=False)
def serve_js():
    f = STATIC / "app.js"
    if not f.exists():
        raise HTTPException(404, "app.js not found")
    return FileResponse(str(f), media_type="application/javascript")


@app.get("/styles.css", include_in_schema=False)
def serve_css():
    f = STATIC / "styles.css"
    if not f.exists():
        raise HTTPException(404, "styles.css not found")
    return FileResponse(str(f), media_type="text/css")


# ── video streaming endpoint (supports range requests for browser playback) ────
@app.get("/outputs/{filename}", include_in_schema=False)
def serve_video(filename: str, request_range: str = None):
    """
    Serve video files with proper headers for browser inline playback.
    Supports HTTP Range requests so the browser can seek.
    """
    from fastapi import Request
    file_path = OUTPUT / filename
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {filename}")

    return FileResponse(
        str(file_path),
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        },
    )


# ── health ─────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "version": "2.0.0"}


# ── upload & analyze ───────────────────────────────────────────────────────────
@app.post("/analyze-video", tags=["offline"])
def analyze_video(file: UploadFile = File(...)):
    allowed = (".mp4", ".avi", ".mov", ".mkv")
    if not file.filename.lower().endswith(allowed):
        raise HTTPException(400, f"Unsupported format. Allowed: {allowed}")

    dest = UPLOAD / Path(file.filename).name
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = analyzer.analyze(str(dest), str(OUTPUT))
    except Exception as exc:
        raise HTTPException(500, str(exc))

    return {**result, "annotated_video": "/outputs/annotated_video.mp4"}


# ── WebSocket live stream ──────────────────────────────────────────────────────
@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await handle_websocket(ws)
