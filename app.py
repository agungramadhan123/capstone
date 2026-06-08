import uvicorn
import cv2
import shutil
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from datetime import datetime, timedelta

app = FastAPI()

# 1. Konfigurasi CORS hanya sekali
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load model
# Menggunakan os.path agar lebih stabil saat deploy ke Linux
model_path = os.path.join(os.getcwd(), "best")
model = YOLO(model_path, task="detect")

# Inisialisasi video (ganti "video.mp4" dengan path video yang benar)
camera = cv2.VideoCapture("video.mp4")
latest_vehicle_count = 0

# Cache untuk update 5 menit sekali
last_detection_time = None
cached_data = None

# 3. Fungsi Generator untuk deteksi real-time
def generate_frames():
    global latest_vehicle_count, camera
    while True:
        if camera is None or not camera.isOpened():
            break
        success, frame = camera.read()
        if not success: break
        
        results = model(frame, conf=0.4)
        latest_vehicle_count = len(results[0].boxes) if len(results) > 0 else 0
        annotated_frame = results[0].plot()
        
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# 4. API Endpoint dengan logika update 5 menit
@app.get("/api/traffic")
def get_traffic_data():
    global last_detection_time, cached_data, latest_vehicle_count
    
    # Update hanya jika data kosong atau sudah lewat 5 menit
    if last_detection_time is None or (datetime.now() - last_detection_time) > timedelta(minutes=5):
        cached_data = {
            "1": {"id": "1", "name": "JL. BUAH BATU (KAMERA LIVE)", "vehicleCount": latest_vehicle_count, "maxCapacity": 100, "status": "recommended" if latest_vehicle_count < 50 else "not-recommended"},
            "2": {"id": "2", "name": "JL. SOEKARNO HATTA", "vehicleCount": 42, "maxCapacity": 100, "status": "recommended"},
            "3": {"id": "3", "name": "JL. PELAJAR PEJUANG", "vehicleCount": 65, "maxCapacity": 100, "status": "not-recommended"},
            "4": {"id": "4", "name": "JL. TERUSAN BUAH BATU", "vehicleCount": 22, "maxCapacity": 100, "status": "recommended"}
        }
        last_detection_time = datetime.now()
        
    return cached_data

@app.get('/api/video_feed')
def video_feed():
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "Video berhasil diunggah", "path": temp_path}

if __name__ == '__main__':
    # Hilangkan reload=True saat nanti deploy ke VPS agar lebih stabil
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)