# Smart Traffic Bandung

Platform berbasis Artificial Intelligence (YOLOv8 + ByteTrack) untuk mendeteksi, melacak, dan menghitung kendaraan dari CCTV (ATCS Bandung) maupun video unggahan guna membantu analisis lalu lintas secara real-time.

## Struktur Proyek
- `frontend/`: Berisi kode antarmuka web (React, Vite, Tailwind CSS).
- `backend/`: Berisi server API FastAPI untuk melayani web.
- `src/`: Berisi logika inti AI, OpenCV, perhitungan ROI (Region of Interest), dan pelacakan kendaraan.

---

## 🚀 Cara Menjalankan Aplikasi (Panduan Cepat)

Anda perlu menjalankan Frontend (Web) dan Backend (AI) secara bersamaan di dua terminal yang berbeda.

### 1. Menjalankan Frontend (Web UI)
Buka terminal pertama, lalu jalankan perintah berikut:
```bash
cd frontend
npm install
npm run dev
```
Buka browser dan akses: `http://localhost:5173`

### 2. Menjalankan Backend AI (FastAPI & YOLOv8)
Buka terminal kedua, lalu jalankan perintah berikut:
```bash

**Windows:**
```bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
cd backend
uvicorn backend.main:app --reload
```

**Mac/Linux:**
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```
Server backend akan berjalan di `http://127.0.0.1:8000`.

---
