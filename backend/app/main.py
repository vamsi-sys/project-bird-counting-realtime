"""
main.py — FastAPI application
"""
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve annotated videos with range-request support (required for browser seek)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT)), name="outputs")

analyzer = VideoAnalyzer()


# ── startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def warmup():
    print("⏳  Loading YOLOv8 model …")
    get_detector()
    print("✅  Model ready.")


# ── frontend static files ──────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def serve_index():
    f = STATIC / "index.html"
    if not f.exists():
        raise HTTPException(404, "index.html missing from backend/static/")
    return FileResponse(str(f))


@app.get("/app.js", include_in_schema=False)
def serve_js():
    f = STATIC / "app.js"
    if not f.exists():
        raise HTTPException(404, "app.js missing")
    return FileResponse(str(f), media_type="application/javascript")


@app.get("/styles.css", include_in_schema=False)
def serve_css():
    f = STATIC / "styles.css"
    if not f.exists():
        raise HTTPException(404, "styles.css missing")
    return FileResponse(str(f), media_type="text/css")


@app.get("/tailwind.css", include_in_schema=False)
def serve_tailwind():
    f = STATIC / "tailwind.css"
    if not f.exists():
        raise HTTPException(404, "tailwind.css missing")
    return FileResponse(str(f), media_type="text/css")


# ── health ─────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "version": "2.0.0"}


# ── video upload & analysis ────────────────────────────────────────────────────
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
    except RuntimeError as exc:
        raise HTTPException(500, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Unexpected error: {exc}")

    return {**result, "annotated_video": "/outputs/annotated_video.mp4"}


# ── WebSocket live stream ──────────────────────────────────────────────────────
@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await handle_websocket(ws)
