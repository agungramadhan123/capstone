import sys
import os
import cv2
import threading
import json
import time
from fastapi import FastAPI, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Tambahkan root path ke sys.path supaya bisa import src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.traffic_monitor import TrafficMonitorApp
from src.config import CLASS_NAMES

app = FastAPI(title="Smart Traffic API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
latest_frame = None
is_analyzing = False
traffic_app = None
vehicle_stats = {"mobil": 0, "motor": 0, "bis": 0, "truk": 0}

class DummyArgs:
    def __init__(self):
        self.source = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4" # Dummy fallback
        self.model = "yolov8n.pt"
        self.csv_output = "traffic_logs_buahbatu.csv"
        self.save_video = False
        self.show = False

def update_frame(annotated_frame):
    global latest_frame, traffic_app, vehicle_stats
    # Encode frame ke JPEG
    ret, buffer = cv2.imencode('.jpg', annotated_frame)
    if ret:
        latest_frame = buffer.tobytes()
        
    # Update stats
    if traffic_app and hasattr(traffic_app, 'counter'):
        counts = traffic_app.counter.total_counts
        vehicle_stats["mobil"] = counts.get("Mobil", counts.get("car", 0))
        vehicle_stats["motor"] = counts.get("Motor", counts.get("motorcycle", 0))
        vehicle_stats["bis"] = counts.get("Bis", counts.get("bus", 0))
        vehicle_stats["truk"] = counts.get("Truk", counts.get("truck", 0))

from pydantic import BaseModel

class StartRequest(BaseModel):
    url: str

def run_traffic_monitor(custom_url: str = ""):
    global is_analyzing, traffic_app
    args = DummyArgs()
    if custom_url:
        args.source = custom_url
        
    traffic_app = TrafficMonitorApp(args, frame_callback=update_frame)
    traffic_app.run()
    is_analyzing = False

@app.post("/api/start")
def start_analysis(req: StartRequest, background_tasks: BackgroundTasks):
    global is_analyzing, traffic_app
    
    # Jika sudah jalan tapi beda URL, kita stop yang lama (kasar, butuh metode stop() di TrafficMonitorApp)
    if is_analyzing:
        return {"status": "already_running", "message": "Analisis sedang berjalan. Refresh halaman atau tunggu selesai."}
        
    is_analyzing = True
    background_tasks.add_task(run_traffic_monitor, req.url)
    return {"status": "started", "url": req.url}

def frame_generator():
    """Generator untuk multipart/x-mixed-replace (MJPEG stream)"""
    while True:
        if latest_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
        time.sleep(0.05) # 20 fps target web stream

@app.get("/api/video_feed")
def video_feed():
    """Endpoint untuk stream video"""
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/traffic/status")
def get_traffic_status():
    """Endpoint untuk mendapatkan data live count"""
    total = sum(vehicle_stats.values())
    
    # Simple logic for condition
    if total < 50:
        cond = "Lancar"
    elif total < 150:
        cond = "Sedang"
    elif total < 300:
        cond = "Padat"
    else:
        cond = "Sangat Padat"
        
    return {
        "vehicles": vehicle_stats,
        "condition": cond,
        "is_running": is_analyzing
    }

@app.get("/api/insight")
def get_insight():
    # Ini adalah endpoint dummy untuk AI insight, bisa diganti dengan LLM
    return {
        "insight": "Integrasi backend berhasil! Anda siap menghubungkan dengan Groq/Gemini."
    }
