/**
 * Bird Counting & Weight Estimation — v2.0.0
 * Binary WebSocket (no Base64 → 33% less bandwidth, no CPU encoding overhead)
 * Modes: Browser Webcam | RTSP/IP Camera | Offline Video Upload
 */

const API_BASE = window.location.origin;
const WS_BASE  = API_BASE.replace(/^http/, "ws");

// ── shared Chart.js instance ───────────────────────────────────────────────────
let _chart = null;

function makeChart(canvasId) {
  const el = document.getElementById(canvasId);
  if (!el) return;
  if (_chart) { _chart.destroy(); _chart = null; }
  _chart = new Chart(el.getContext("2d"), {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: "Birds",
        data: [],
        borderColor: "#2563eb",
        backgroundColor: "rgba(37,99,235,.12)",
        borderWidth: 2,
        pointRadius: 3,
        fill: true,
        tension: 0.35,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, ticks: { precision: 0 },
             title: { display: true, text: "Bird count" } },
        x: { title: { display: true, text: "Elapsed (s)" } },
      },
    },
  });
}

function pushChart(countsOverTime) {
  if (!_chart || !countsOverTime || !countsOverTime.length) return;
  const pt = countsOverTime[countsOverTime.length - 1];
  _chart.data.labels.push(`${pt.time_sec}s`);
  _chart.data.datasets[0].data.push(pt.count);
  if (_chart.data.labels.length > 60) {
    _chart.data.labels.shift();
    _chart.data.datasets[0].data.shift();
  }
  _chart.update("none");  // "none" skips animation = lower CPU
}

function setChartStatic(countsOverTime) {
  if (!_chart) return;
  _chart.data.labels              = countsOverTime.map(d => `${d.time_sec}s`);
  _chart.data.datasets[0].data   = countsOverTime.map(d => d.count);
  _chart.update();
}

// ── helpers ────────────────────────────────────────────────────────────────────
function setText(id, v) {
  const e = document.getElementById(id);
  if (e) e.textContent = v;
}

function applyStats(stats, prefix) {
  const p = prefix || "";
  setText(p + "statUnique",  stats.unique_birds       ?? "—");
  setText(p + "statCurrent", stats.current_detections ?? "—");
  setText(p + "statElapsed", stats.elapsed_sec != null ? stats.elapsed_sec + "s" : "—");
  const w = stats.weight_estimation;
  setText(p + "statAvg",    w ? w.average_grams + " g" : "—");
  setText(p + "statMin",    w ? w.min_grams     + " g" : "—");
  setText(p + "statMax",    w ? w.max_grams     + " g" : "—");
  pushChart(stats.counts_over_time);
}

// ── Tab switching ──────────────────────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.add("hidden"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.remove("hidden");
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// TAB 1 — BROWSER WEBCAM  (binary WebSocket — no Base64)
// ══════════════════════════════════════════════════════════════════════════════
{
  let ws         = null;
  let stream     = null;
  let capturing  = false;
  const rawVideo = document.getElementById("webcamVideo");
  const canvas   = document.getElementById("liveCanvas");
  const ctx2d    = canvas.getContext("2d");
  const capture  = document.createElement("canvas");
  const captCtx  = capture.getContext("2d");

  document.getElementById("btnStartWebcam").addEventListener("click", async () => {
    setWcStatus("Requesting camera…", false);
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      rawVideo.srcObject = stream;
      await rawVideo.play();
    } catch (e) {
      setWcStatus("Camera error: " + e.message, false);
      return;
    }

    ws = new WebSocket(WS_BASE + "/ws/live");
    ws.binaryType = "blob";

    ws.onopen = () => {
      setWcStatus("Connected — streaming…", true);
      makeChart("webcamChart");
      capturing = true;
      sendLoop();
    };

    ws.onerror   = () => setWcStatus("WebSocket error", false);
    ws.onclose   = () => { capturing = false; setWcStatus("Disconnected", false); };

    ws.onmessage = e => {
      if (e.data instanceof Blob) {
        // Binary frame — draw on canvas
        const url = URL.createObjectURL(e.data);
        const img = new Image();
        img.onload = () => {
          canvas.width  = img.width;
          canvas.height = img.height;
          ctx2d.drawImage(img, 0, 0);
          URL.revokeObjectURL(url);   // free memory immediately
        };
        img.src = url;
      } else {
        // Text stats JSON
        const msg = JSON.parse(e.data);
        if (msg.type === "stats")       applyStats(msg.stats, "");
        else if (msg.type === "error")  setWcStatus("Error: " + msg.message, false);
      }
    };

    document.getElementById("btnStartWebcam").classList.add("hidden");
    document.getElementById("btnStopWebcam").classList.remove("hidden");
  });

  document.getElementById("btnStopWebcam").addEventListener("click", () => {
    capturing = false;
    if (ws) { ws.send(JSON.stringify({ action: "stop" })); ws.close(); ws = null; }
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    ctx2d.clearRect(0, 0, canvas.width, canvas.height);
    setWcStatus("Stopped", false);
    document.getElementById("btnStartWebcam").classList.remove("hidden");
    document.getElementById("btnStopWebcam").classList.add("hidden");
  });

  /**
   * sendLoop — captures frames and sends as binary JPEG blobs.
   * Uses a self-scheduling approach: sends next frame only after
   * toBlob() completes, implementing natural backpressure.
   */
  function sendLoop() {
    if (!capturing || !ws || ws.readyState !== WebSocket.OPEN) return;

    const vw = rawVideo.videoWidth;
    const vh = rawVideo.videoHeight;
    if (!vw || !vh) { setTimeout(sendLoop, 100); return; }

    // Resize to 480px wide to keep bandwidth low
    capture.width  = 480;
    capture.height = Math.round(vh * 480 / vw);
    captCtx.drawImage(rawVideo, 0, 0, capture.width, capture.height);

    capture.toBlob(blob => {
      if (ws && ws.readyState === WebSocket.OPEN && blob) {
        ws.send(blob);   // send raw binary JPEG — no Base64 needed
      }
      // Schedule next frame only after this one is sent
      if (capturing) requestAnimationFrame(sendLoop);
    }, "image/jpeg", 0.65);
  }

  function setWcStatus(msg, live) {
    const el = document.getElementById("webcamStatus");
    el.innerHTML = live ? `<span class="pulse-dot mr-2"></span>${msg}` : msg;
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// TAB 2 — RTSP / IP CAMERA
// ══════════════════════════════════════════════════════════════════════════════
{
  let ws = null;

  document.getElementById("btnConnectRTSP").addEventListener("click", () => {
    const url = document.getElementById("rtspUrl").value.trim();
    if (!url) { setRtspStatus("Please enter an RTSP URL.", false); return; }

    setRtspStatus("Connecting…", false);
    ws = new WebSocket(WS_BASE + "/ws/live");
    ws.binaryType = "blob";

    ws.onopen = () => {
      ws.send(JSON.stringify({ action: "start_rtsp", url }));
      setRtspStatus("Connected — receiving frames…", true);
      makeChart("rtspChart");
      document.getElementById("btnConnectRTSP").classList.add("hidden");
      document.getElementById("btnDisconnectRTSP").classList.remove("hidden");
    };

    ws.onmessage = e => {
      if (e.data instanceof Blob) {
        const url = URL.createObjectURL(e.data);
        const img = document.getElementById("rtspImg");
        img.onload = () => URL.revokeObjectURL(url);
        img.src    = url;
      } else {
        const msg = JSON.parse(e.data);
        if (msg.type === "stats")       applyStats(msg.stats, "r");
        else if (msg.type === "error")  setRtspStatus("Error: " + msg.message, false);
        else if (msg.type === "stopped") setRtspStatus("Stream ended", false);
      }
    };

    ws.onerror = () => setRtspStatus("WebSocket error — check RTSP URL", false);
    ws.onclose = () => setRtspStatus("Disconnected", false);
  });

  document.getElementById("btnDisconnectRTSP").addEventListener("click", () => {
    if (ws) { ws.send(JSON.stringify({ action: "stop" })); ws.close(); ws = null; }
    setRtspStatus("Disconnected", false);
    document.getElementById("btnConnectRTSP").classList.remove("hidden");
    document.getElementById("btnDisconnectRTSP").classList.add("hidden");
  });

  function setRtspStatus(msg, live) {
    const el = document.getElementById("rtspStatus");
    el.innerHTML = live ? `<span class="pulse-dot mr-2"></span>${msg}` : msg;
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// TAB 3 — UPLOAD VIDEO
// ══════════════════════════════════════════════════════════════════════════════
{
  const fileInput = document.getElementById("videoFile");
  const dropZone  = document.getElementById("dropZone");
  const fileLabel = document.getElementById("fileNameDisplay");
  let   progTimer = null;

  // File selection display
  fileInput.addEventListener("change", () => {
    fileLabel.textContent = fileInput.files[0]?.name ?? "Click or drag a video here";
    dropZone.classList.toggle("drag-over", !!fileInput.files.length);
  });

  // Drag and drop
  ["dragenter","dragover","dragleave","drop"].forEach(ev =>
    dropZone.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); })
  );
  ["dragenter","dragover"].forEach(ev =>
    dropZone.addEventListener(ev, () => dropZone.classList.add("drag-over"))
  );
  ["dragleave","drop"].forEach(ev =>
    dropZone.addEventListener(ev, () => {
      if (!fileInput.files.length) dropZone.classList.remove("drag-over");
    })
  );
  dropZone.addEventListener("drop", e => {
    const f = e.dataTransfer.files;
    if (f.length) { fileInput.files = f; fileInput.dispatchEvent(new Event("change")); }
  });

  // Form submit
  document.getElementById("uploadForm").addEventListener("submit", async e => {
    e.preventDefault();
    if (!fileInput.files.length) { showUploadError("Please select a video file first."); return; }

    const btn = document.getElementById("analyzeBtn");
    btn.disabled = true;
    document.getElementById("uploadError").classList.add("hidden");
    document.getElementById("uploadResults").classList.add("hidden");
    document.getElementById("uploadLoading").classList.remove("hidden");
    startProgress();

    try {
      const fd  = new FormData();
      fd.append("file", fileInput.files[0]);

      const res = await fetch(API_BASE + "/analyze-video", {
        method: "POST",
        body: fd,
      });

      if (!res.ok) {
        let detail = "Unknown server error";
        try { detail = (await res.json()).detail || detail; } catch(_) {}
        throw new Error(detail);
      }

      const data = await res.json();
      renderUploadResults(data);
    } catch (err) {
      showUploadError(err.message);
    } finally {
      btn.disabled = false;
      document.getElementById("uploadLoading").classList.add("hidden");
      clearInterval(progTimer);
    }
  });

  function startProgress() {
    let pct = 0;
    clearInterval(progTimer);
    progTimer = setInterval(() => {
      pct = Math.min(pct + Math.random() * 2.5, 90);
      document.getElementById("uploadProgressBar").style.width = pct + "%";
      document.getElementById("uploadProgressText").textContent =
        "Processing… " + Math.round(pct) + "%";
    }, 600);
  }

  function renderUploadResults(data) {
    clearInterval(progTimer);
    document.getElementById("uploadProgressBar").style.width = "100%";
    document.getElementById("uploadProgressText").textContent = "Done!";

    setText("upUnique", data.unique_birds);
    setText("upFrames", data.frames_processed);
    setText("upTime",   data.processing_time_sec + " s");
    setText("upFps",    data.fps + " fps");

    const w = data.weight_estimation;
    setText("upAvg", w ? w.average_grams + " g" : "N/A");
    setText("upMin", w ? w.min_grams     + " g" : "N/A");
    setText("upMax", w ? w.max_grams     + " g" : "N/A");

    // Video player — force reload for new result
    const videoUrl = API_BASE + data.annotated_video + "?t=" + Date.now();
    const vid = document.getElementById("resultVideo");
    vid.src = videoUrl;
    vid.load();

    const dlBtn = document.getElementById("dlBtn");
    dlBtn.href = videoUrl;
    dlBtn.setAttribute("download", "annotated_video.mp4");

    makeChart("uploadChart");
    setChartStatic(data.counts_over_time || []);

    const tbody = document.getElementById("tracksBody");
    tbody.innerHTML = (data.tracks_sample || []).map(t =>
      `<tr class="border-b hover:bg-gray-50">
         <td class="px-4 py-2 font-mono">${t.id}</td>
         <td class="px-4 py-2 font-mono text-xs">[${t.bbox.join(", ")}]</td>
       </tr>`
    ).join("");

    document.getElementById("uploadResults").classList.remove("hidden");
    document.getElementById("uploadResults").scrollIntoView({ behavior: "smooth" });
  }

  function showUploadError(msg) {
    document.getElementById("uploadErrMsg").textContent = msg;
    document.getElementById("uploadError").classList.remove("hidden");
  }
}
