
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
from ultralytics import YOLO

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Blok Sakti CORS biar Google Chrome tidak memblokir datanya
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model YOLO dari folder 'best' yang sudah kamu masukkan tadi
model = YOLO("./best")

# Gunakan webcam (angka 0) dulu untuk tes darurat malam ini
camera = cv2.VideoCapture(0)

latest_vehicle_count = 0

# 2. Tambahkan blok sakti ini agar React diizinkan mengambil video
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mengizinkan semua alamat (termasuk localhost:5173 kamu)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ... (sisa kode fungsi generator dan model YOLO kamu di bawahnya biarkan saja)

# 2. Aktifkan CORS agar frontend React bisa mengakses backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model 'best' hasil training dari Agung
# Pastikan folder bernama 'best' berada satu lokasi/sejajar dengan file app.py ini
# Diarahkan langsung ke nama foldernya menggunakan './' biar Python tahu itu folder lokal
model = YOLO("./best")

# Variabel global untuk menyimpan hitungan kendaraan terbaru secara real-time
latest_vehicle_count = 0

# 3. Fungsi Generator untuk memproses Video + Deteksi YOLOv8
def generate_frames():
    global latest_vehicle_count
    
    # diarahkan ke file video rekaman yang sudah kamu siapkan (misal video.mp4)
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            # Jika video habis, putar ulang dari awal biar terus looping
            camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Jalankan prediksi objek menggunakan model 'best'
        # conf=0.4 artinya model hanya mengambil deteksi yang tingkat yakinnya di atas 40%
        results = model(frame, conf=0.4)
        
        # Hitung berapa banyak objek/kendaraan yang terdeteksi di frame ini
        if len(results) > 0:
            latest_vehicle_count = len(results[0].boxes)
        else:
            latest_vehicle_count = 0
            
        # Gambar kotak hasil deteksi (annotated) ke atas frame video
        annotated_frame = results[0].plot()
        
        # Encode frame gambar menjadi JPG untuk dikirim via HTTP streaming
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# 4. API Endpoint untuk menyuplai Angka Hitungan Kendaraan Asli ke React
# API Endpoint yang sudah disesuaikan dengan format Object yang diminta React kamu
@app.get("/api/traffic")
def get_traffic_data():
    global latest_vehicle_count
    
    # Kita kembalikan dalam bentuk Dictionary/Object {}, bukan List []
    return {
        "1": {
            "id": "1", 
            "name": "JL. BUAH BATU (KAMERA LIVE)", 
            "vehicleCount": latest_vehicle_count, 
            "maxCapacity": 100, 
            "status": "recommended" if latest_vehicle_count < 50 else "not-recommended"
        },
        "2": {
            "id": "2", 
            "name": "JL. SOEKARNO HATTA", 
            "vehicleCount": 42, 
            "maxCapacity": 100, 
            "status": "recommended"
        },
        "3": {
            "id": "3", 
            "name": "JL. PELAJAR PEJUANG", 
            "vehicleCount": 65, 
            "maxCapacity": 100, 
            "status": "not-recommended"
        },
        "4": {
            "id": "4", 
            "name": "JL. TERUSAN BUAH BATU", 
            "vehicleCount": 22, 
            "maxCapacity": 100, 
            "status": "recommended"
        }
    }

# 5. API Endpoint untuk kirim live streaming video kotak hijau ke React
@app.get('/api/video_feed')
def video_feed():
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

# 6. Menjalankan Server
if __name__ == '__main__':
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)