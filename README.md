# 🐔 Bird Counting & Weight Estimation System

> **Live Demo:** [https://project-bird-counting-realtime.onrender.com](https://project-bird-counting-realtime.onrender.com)

A real-time computer vision web application that automatically **counts birds** and **estimates their weight** from live camera feeds or recorded video footage. Built for poultry farm management using state-of-the-art YOLOv8 object detection and tracking.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.13 |
| **Backend Framework** | FastAPI |
| **ML Model** | YOLOv8 Nano (Ultralytics) |
| **Computer Vision** | OpenCV |
| **Real-Time Streaming** | WebSocket |
| **Video Re-encoding** | FFmpeg (H.264) |
| **Frontend** | HTML5 · Tailwind CSS · Chart.js · Vanilla JS |
| **Deployment** | Render.com (Docker) |

---

## 📁 Project Structure

```
bcr-final/
├── Dockerfile                   ← Docker build config
├── docker-compose.yml           ← One-command local Docker run
├── .gitignore
├── README.md
└── backend/
    ├── app/
    │   ├── __init__.py
    │   ├── main.py              ← FastAPI app: all routes (REST + WebSocket)
    │   ├── detector.py          ← YOLOv8 singleton detector + frame annotation
    │   ├── live.py              ← WebSocket handler for webcam and RTSP streams
    │   ├── tracker.py           ← Offline video file analysis pipeline
    │   └── weight.py            ← Heuristic bird weight estimator
    ├── models/
    │   └── yolov8n.pt           ← Pre-trained YOLOv8 Nano model (included)
    ├── static/                  ← Frontend files (served directly by FastAPI)
    │   ├── index.html           ← 3-tab web UI
    │   ├── app.js               ← Frontend logic: WebSocket client, upload, charts
    │   └── styles.css           ← Animations, spinner, tab styles
    ├── uploads/                 ← Temporary storage for uploaded videos
    ├── outputs/                 ← Annotated output videos
    └── requirements.txt         ← Python dependencies
```

---

## 🧠 How the System Works

### Core Detection Pipeline

Every mode (webcam, RTSP, upload) uses the same underlying pipeline:

```
Video Frame
     ↓
Resize to 640px width  (faster inference)
     ↓
YOLOv8 .track()        (detect + assign unique Track IDs)
     ↓
Scale boxes back       (original resolution)
     ↓
Draw bounding boxes    (colour-coded per bird ID)
     ↓
Estimate weight        (bounding box area → grams)
     ↓
Annotated Frame + Stats
```

### Bird Tracking

YOLOv8's built-in `.track()` method assigns a **persistent unique ID** to each bird across frames. This means:
- Even if a bird leaves and re-enters the frame, it gets the same ID
- The system never double-counts the same bird
- The "Unique Birds" counter only goes up, never down

### Weight Estimation

Weight is estimated using a **heuristic formula** based on bounding box pixel area:

```
weight (grams) = (bounding_box_area / base_area) × base_weight
```

Where:
- `base_area = 5000 pixels²` — typical bounding box area for a broiler at 640px width
- `base_weight = 1200 grams` — typical broiler chicken weight

> ⚠️ This is an approximation. Accuracy improves significantly with camera calibration data and a fine-tuned model trained on your specific farm setup.

---

## 🎥 Features Explained

---

### Tab 1 — 🎥 Live Webcam

**What it does:**
Uses your computer's built-in camera or USB webcam to detect and count birds in real time.

**How it works step by step:**

1. You click **Start Camera**
2. The browser asks for camera permission — click Allow
3. Your browser captures live video frames using the `getUserMedia` API
4. Every 3rd frame is taken (to reduce bandwidth), resized to 640px, and converted to a Base64-encoded JPEG
5. The Base64 frame is sent to the backend over a **WebSocket connection** (`/ws/live`)
6. The backend receives the frame, runs YOLOv8 detection on it, draws bounding boxes and bird IDs, overlays the live count
7. The annotated frame is encoded back to Base64 JPEG and sent back to the browser over the same WebSocket
8. The browser displays the returned annotated frame on a `<canvas>` element
9. The stats panel (unique birds, in-frame count, weight, elapsed time) updates live
10. The "Birds Over Time" chart plots new data points every few seconds

**Buttons:**
- **Start Camera** — requests camera access, opens WebSocket, starts streaming
- **Stop** — closes the webcam stream and WebSocket connection, clears the canvas

**Best for:** Real-time monitoring when your computer is physically near the birds.

---

### Tab 2 — 📡 RTSP / IP Camera

**What it does:**
Connects to a network CCTV camera (the kind used in poultry farms) using an RTSP stream URL and performs real-time bird detection.

**What is RTSP?**
RTSP (Real Time Streaming Protocol) is the standard protocol used by IP cameras, DVRs, and NVRs to stream video over a network. Every farm CCTV camera supports it.

**How it works step by step:**

1. You enter your camera's RTSP URL in the input box
   - Format: `rtsp://username:password@camera_ip:554/stream`
   - Example: `rtsp://admin:pass123@192.168.1.100:554/stream1`
2. You click **Connect**
3. The browser sends a WebSocket message to the backend with the RTSP URL
4. The **backend** (not the browser) opens the RTSP stream using OpenCV's `VideoCapture`
5. The backend reads frames from the camera, runs YOLOv8 on every 2nd frame
6. Annotated frames are encoded as JPEG and pushed to the browser over WebSocket
7. The browser displays the incoming annotated frames as a live `<img>` feed
8. Stats update in real time — unique birds, in-frame count, estimated weights

**Buttons:**
- **Connect** — sends RTSP URL to backend, starts the stream pipeline
- **Disconnect** — stops the backend stream and closes the WebSocket

> ⚠️ **Important:** The RTSP camera must be reachable from the machine running the backend — not from your browser. When running locally, your camera must be on the same network as your PC. When deployed on Render, RTSP will not work because Render's servers cannot access your local network camera.

**Best for:** Production farm monitoring where fixed IP cameras are installed.

---

### Tab 3 — 📁 Upload Video

**What it does:**
Upload a pre-recorded video file (.mp4, .avi, .mov, .mkv) for full offline analysis. The system processes the entire video and returns detailed results including an annotated video, count chart, and weight statistics.

**How it works step by step:**

1. You drag and drop a video file onto the upload zone, or click to browse
2. You click **Analyze Video**
3. The frontend sends the file to the backend via HTTP POST to `/analyze-video`
4. A progress bar animates while you wait
5. The backend saves the video to `backend/uploads/`
6. OpenCV reads the video frame by frame
7. Every 2nd frame is processed through YOLOv8 detection + tracking
8. Each processed frame is annotated and written to a temporary output file
9. After all frames are processed, FFmpeg re-encodes the output from `mp4v` codec to **H.264** — the format all browsers can play inline
10. The annotated video is saved to `backend/outputs/annotated_video.mp4`
11. The backend returns a JSON response with all stats
12. The frontend displays:
    - **4 stat cards** — unique birds, average weight, frames processed, processing time
    - **Annotated video player** — watch the video with bounding boxes and IDs
    - **Download button** — save the annotated video to your computer
    - **Birds Over Time chart** — line graph of bird count at every 5-second interval
    - **Top-10 Bird Tracks table** — track ID and bounding box coordinates for each detected bird
    - **Weight range** — min, average, and max estimated weight

**Buttons:**
- **Analyze Video** — uploads the file and triggers full analysis
- **Download annotated video** — downloads the H.264 MP4 with all bounding boxes

**Best for:** Analysing existing footage, testing the system, or reviewing recordings from a DVR.

---

## 📊 Stats Dashboard Explained

| Stat | Description |
|------|-------------|
| **Unique Birds** | Total number of distinct birds detected across all frames (never counts the same bird twice) |
| **In Frame** | Number of birds visible in the current frame right now |
| **Avg Weight** | Average estimated weight across all tracked birds (grams) |
| **Elapsed** | How long the stream or video has been running |
| **Min Weight** | Lightest bird detected |
| **Max Weight** | Heaviest bird detected |
| **Birds Over Time** | Chart showing how the unique bird count grew over time |

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the web UI |
| `GET` | `/health` | Health check → `{"status":"ok","version":"2.0.0"}` |
| `GET` | `/app.js` | Serves frontend JavaScript |
| `GET` | `/styles.css` | Serves frontend CSS |
| `GET` | `/outputs/{filename}` | Streams output video with range request support |
| `POST` | `/analyze-video` | Upload video for offline analysis |
| `WS` | `/ws/live` | WebSocket endpoint for live streaming |

### WebSocket Message Protocol

**Browser → Server:**
```json
{ "action": "start_webcam" }
{ "action": "start_rtsp", "url": "rtsp://admin:pass@192.168.1.100:554/stream1" }
{ "action": "frame", "data": "<base64 JPEG>" }
{ "action": "stop" }
```

**Server → Browser:**
```json
{
  "type": "frame",
  "data": "<base64 JPEG>",
  "stats": {
    "unique_birds": 42,
    "current_detections": 8,
    "elapsed_sec": 12.5,
    "weight_estimation": {
      "average_grams": 1450.0,
      "min_grams": 900.0,
      "max_grams": 2100.0
    },
    "counts_over_time": [
      { "time_sec": 5.0, "count": 20 },
      { "time_sec": 10.0, "count": 35 }
    ]
  }
}
```
```json
{ "type": "error",   "message": "Cannot open RTSP stream" }
{ "type": "stopped" }
```

---

## 🚀 Run Locally

### Prerequisites
- **Python 3.13** — https://python.org/downloads
- **FFmpeg** — required for browser-playable video output

**Install FFmpeg on Windows:**
```cmd
winget install ffmpeg
```

**Install FFmpeg on Mac:**
```bash
brew install ffmpeg
```

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/vamsi-sys/bird-counting-realtime.git
cd bird-counting-realtime

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Start the server (from inside backend/)
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**

---

## 🐳 Run with Docker

```bash
docker compose up --build
```

Open **http://localhost:8000**

---

## ☁️ Deployment

**Live deployment:** [https://project-bird-counting-realtime.onrender.com](https://project-bird-counting-realtime.onrender.com)

Deployed on **Render.com** using Docker.

### Deploy your own instance

1. Fork this repository
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your forked repository
4. Set **Runtime** to **Docker**
5. Click **Create Web Service**

Render auto-deploys on every push to `main`.

> **Note:** The free tier on Render sleeps after 15 minutes of inactivity. The first request after sleep takes ~30 seconds to wake up. RTSP camera streams only work when running locally — not on Render's cloud servers.

---

## ⚙️ Configuration

To tune the weight estimator for your specific camera and bird size, edit `backend/app/weight.py`:

```python
class WeightEstimator:
    def __init__(self, base_area: float = 5000.0, base_weight: float = 1200.0):
        self.base_area   = base_area    # adjust based on your camera height/zoom
        self.base_weight = base_weight  # adjust based on your bird breed/age
```

To change detection confidence threshold, edit `backend/app/detector.py`:

```python
class BirdDetector:
    def __init__(self, conf: float = 0.40, resize_width: int = 640):
        self.conf = 0.40   # increase for fewer false positives, decrease for more sensitivity
```

---

## 👤 Author

**Vamsikrishna Sirimalla**
GitHub: [github.com/vamsi-sys](https://github.com/vamsi-sys)

## 📄 License

MIT — free to use, modify, and deploy.
