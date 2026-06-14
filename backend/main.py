import sys
import os
import cv2
import threading
import json
import time
import shutil
import csv
from fastapi import FastAPI, BackgroundTasks, Response, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Tambahkan root path ke sys.path supaya bisa import src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.traffic_monitor import TrafficMonitorApp
from src.config import CLASS_NAMES, DEFAULT_CSV

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Smart Traffic API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Smart Traffic API is running!"}


# Global State
latest_frame = None
is_analyzing = False
traffic_app = None
vehicle_stats = {"mobil": 0, "motor": 0, "bis": 0, "truk": 0}

class DummyArgs:
    def __init__(self):
        from src.config import DEFAULT_MODEL, DEFAULT_CSV, DEFAULT_VIDEO
        self.source = DEFAULT_VIDEO
        self.model = DEFAULT_MODEL
        self.csv_output = DEFAULT_CSV
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
    
    # Jika sudah jalan tapi beda URL, kita stop yang lama
    if is_analyzing:
        return {"status": "already_running", "message": "Analisis sedang berjalan. Refresh halaman atau tunggu selesai."}
        
    # Hapus file CSV log lama agar uji ini mulai dari nol (0)
    if os.path.exists(DEFAULT_CSV):
        try:
            os.remove(DEFAULT_CSV)
        except Exception:
            pass

    is_analyzing = True
    background_tasks.add_task(run_traffic_monitor, req.url)
    return {"status": "started", "url": req.url}

@app.post("/api/stop")
def stop_analysis():
    global is_analyzing, traffic_app
    if is_analyzing and traffic_app:
        traffic_app.stop()
        is_analyzing = False
        
        # Tunggu sejenak agar background task (YOLO loop) selesai memproses frame terakhir 
        # dan memanggil _cleanup() yang akan mem-flush & menutup file CSV.
        import time
        time.sleep(1.5)
        
        # Pindahkan/Append data dari sesi ini ke CSV Master (Keseluruhan)
        try:
            if os.path.exists(DEFAULT_CSV):
                master_path = DEFAULT_CSV.replace("traffic_logs_buahbatu.csv", "traffic_logs_master.csv")
                master_exists = os.path.exists(master_path)
                with open(DEFAULT_CSV, "r", encoding="utf-8") as f_in:
                    lines_csv = f_in.readlines()
                    if len(lines_csv) > 1:
                        with open(master_path, "a", encoding="utf-8") as f_out:
                            if not master_exists:
                                f_out.writelines(lines_csv)
                            else:
                                f_out.writelines(lines_csv[1:])
        except Exception as e:
            print(f"Gagal memindahkan data ke master CSV: {e}")

        return {"status": "stopped", "message": "Analisis dihentikan dan data disimpan ke log master."}
    return {"status": "not_running", "message": "Tidak ada analisis yang berjalan."}

@app.post("/api/upload")
def upload_video(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            import shutil
            shutil.copyfileobj(file.file, buffer)
        # Convert path to standard string to pass to OpenCV/YOLO later
        return {"status": "success", "url": file_path, "message": f"File {file.filename} berhasil diupload"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
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

@app.get("/api/logs/latest")
def get_latest_logs():
    """Endpoint untuk membaca log CSV terakhir dan mengembalikan agregasi data untuk grafik"""
    if not os.path.exists(DEFAULT_CSV):
        return {"status": "error", "message": "File CSV tidak ditemukan.", "data": []}
    
    data = []
    try:
        with open(DEFAULT_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e), "data": []}

@app.get("/api/logs/master")
def get_master_logs():
    """Endpoint untuk membaca log Master (Keseluruhan gabungan semua uji)"""
    master_path = DEFAULT_CSV.replace("traffic_logs_buahbatu.csv", "traffic_logs_master.csv")
    if not os.path.exists(master_path):
        return {"status": "error", "message": "File Master CSV tidak ditemukan.", "data": []}
    
    data = []
    try:
        with open(master_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e), "data": []}
