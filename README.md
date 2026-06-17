# 🐔 Bird Counting & Weight Estimation — v2.0.0

> YOLOv8 · Real-Time WebSocket · RTSP/IP Camera · Browser Webcam · Offline Video Upload

---

## 📁 Project Structure

```
bcr-final/
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── README.md
└── backend/
    ├── app/
    │   ├── __init__.py
    │   ├── main.py        ← FastAPI routes (REST + WebSocket)
    │   ├── detector.py    ← YOLOv8 singleton detector
    │   ├── live.py        ← WebSocket handler (webcam + RTSP)
    │   ├── tracker.py     ← Offline video analyzer
    │   └── weight.py      ← Weight estimator
    ├── models/
    │   └── yolov8n.pt     ← Pre-trained model (included)
    ├── static/            ← Frontend files (already here — no copying needed)
    │   ├── index.html
    │   ├── app.js
    │   └── styles.css
    ├── uploads/           ← Auto-used for uploaded videos
    ├── outputs/           ← Auto-used for annotated output
    └── requirements.txt
```

---

## 🚀 Run Locally

### Step 1 — Create virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r backend/requirements.txt
```

### Step 3 — Start the server

```bash
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Step 4 — Open the app

```
http://127.0.0.1:8000
```

No copying of files needed. Everything is already in place.

---

## 🐳 Run with Docker

```bash
docker compose up --build
```

Open **http://localhost:8000**

---

## 🎥 Features

| Tab | Description |
|-----|-------------|
| **Live Webcam** | Browser captures frames → sends via WebSocket → YOLO detects → annotated frame returned |
| **RTSP / IP Camera** | Enter RTSP URL → backend opens stream → annotated frames sent to browser via WebSocket |
| **Upload Video** | Upload .mp4/.avi/.mov/.mkv → full offline analysis → annotated video + chart + stats |

---

## ☁️ Deploy

### Render.com
1. Push to GitHub
2. New Web Service → Docker runtime → connect repo → Deploy

### Railway.app
1. Push to GitHub
2. New project → Deploy from GitHub → auto-detects Dockerfile

---

## 👤 Author
Vamsikrishna Sirimalla — [github.com/vamsi-sys](https://github.com/vamsi-sys)
