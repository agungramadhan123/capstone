# Alur Logika dan Integrasi Frontend ke Backend

Dokumen ini menjelaskan secara menyeluruh bagaimana antarmuka pengguna (Frontend - React) berkomunikasi dengan sistem pemrosesan utama (Backend - FastAPI & YOLOv8) mulai dari bagian paling atas (Section 1) hingga bagian akhir halaman (Section 9).

Setiap seksi memiliki alur komunikasi spesifik yang dikelola menggunakan _state management_ (`Zustand`) dan permintaan HTTP (`fetch`).

---

## Section 1: Hero Section
* **Arah Integrasi:** Tidak Ada (Lokal Frontend).
* **Fungsi:** Menampilkan sambutan, judul, dan penjelasan singkat aplikasi.
* **Penjelasan:** Seksi ini murni komponen UI visual (HTML/CSS) yang dianimasikan menggunakan Framer Motion. Tidak ada data yang dikirim atau diambil dari backend pada seksi ini.

---

## Section 2: Input Source (Pemilihan Sumber Video)
* **Arah Integrasi:** Menuju _Global State_ (Zustand), yang nantinya dikirim ke Backend.
* **Fungsi:** Mengatur sumber data mana yang akan dianalisis oleh YOLOv8 (Live CCTV atau Upload Video).
* **Penjelasan:** Saat pengguna memilih salah satu kamera CCTV dari tabel, ID kamera dan URL RTSP disimpan ke dalam penyimpanan global (Zustand store). Secara konseptual, ketika analisis dimulai, data ini dapat dikirimkan ke `/api/start` sebagai _payload_ agar backend mengetahui alamat IP CCTV mana yang harus dibuka oleh `cv2.VideoCapture()`.

---

## Section 3: Live Monitoring (Inti Integrasi)
Ini adalah seksi paling krusial di mana pertukaran data dua arah dan _streaming_ video terjadi secara masif.

### 3.1. Memulai Pipeline CV (POST `/api/start`)
* **Arah:** `Frontend` ➔ `Backend (FastAPI)` ➔ `traffic_monitor.py`.
* **Fungsi:** Menginstruksikan server untuk memuat model YOLOv8 ke dalam RAM dan memulai _looping_ pembacaan video OpenCV.
* **Penjelasan:** Saat tombol "Mulai Analisis" diklik, fungsi `handleStartSim` menjalankan `fetch('http://127.0.0.1:8000/api/start', {method: 'POST'})`. Backend menerimanya dan menjalankan `TrafficMonitorApp().run()` di dalam _background thread_ agar server web tidak *freeze* (beku).

### 3.2. Streaming Video (GET `/api/video_feed`)
* **Arah:** `Backend (FastAPI Generator)` ➔ `Frontend (<img> tag)`.
* **Fungsi:** Menampilkan video hasil deteksi secara *real-time* ke antarmuka web.
* **Penjelasan:** Frontend memuat URL `http://127.0.0.1:8000/api/video_feed` langsung di atribut `src` pada tag gambar. Backend FastAPI menggunakan fungsi _Generator_ (`yield`) yang memotong-motong frame JPEG secara terus-menerus menggunakan format `multipart/x-mixed-replace`, sehingga peramban web melihatnya sebagai video bergerak padahal ia adalah tumpukan gambar.

### 3.3. Menarik Data Kendaraan (GET `/api/traffic/status`)
* **Arah:** `Backend (FastAPI)` ➔ `Frontend (useEffect Polling)`.
* **Fungsi:** Mengambil angka hitungan kendaraan dari memori backend ke layar React.
* **Penjelasan:** Selama status analisis adalah `completed`, fungsi `setInterval` di frontend akan melempar `fetch` setiap 1 detik ke backend. Backend merespons dengan format JSON berisi hitungan `{"mobil": 10, "motor": 20, ...}` hasil tarikan dari *Counter Object* YOLOv8. Angka ini dimasukkan ke _Global State_ agar komponen lain (seperti KPI Card) otomatis terperbarui.

---

## Section 4: Traffic Status (Lampu Indikator)
* **Arah Integrasi:** Internal Frontend (Mendengarkan data dari Section 3).
* **Fungsi:** Mengubah warna indikator lampu lalu lintas (Lancar, Sedang, Padat, Sangat Padat).
* **Penjelasan:** Seksi ini tidak secara langsung melakukan `fetch` ke backend. Ia hanya "mendengarkan" _Global State_ (`useAppStore(state => state.trafficCondition)`). Ketika `LiveMonitoring` berhasil mendapatkan status kepadatan dari endpoint `/api/traffic/status` setiap 1 detik, indikator warna di seksi ini langsung bereaksi dan mengganti animasinya secara otomatis.

---

## Section 5: Map Monitoring (Peta Leaflet)
* **Arah Integrasi:** Saat ini Statis (Dipersiapkan untuk `GET /api/cctv/locations`).
* **Fungsi:** Memvisualisasikan lokasi fisik kamera yang sedang memantau.
* **Penjelasan:** Peta memuat koordinat dari _library_ pihak ketiga (OpenStreetMap via Leaflet). Pada pengembangannya, data status kamera (Online/Offline) dan koordinat dapat ditarik dari _database backend_ secara dinamis.

---

## Section 6: Traffic Analytics (Grafik Recharts)
* **Arah Integrasi:** Dipersiapkan untuk HTTP `GET /api/traffic/history` (Data Logger CSV).
* **Fungsi:** Menampilkan tren historis lalu lintas berbentuk _Line Chart_.
* **Penjelasan:** Karena logika saat ini disembunyikan jika backend tidak berjalan, komponen ini sudah di-set dinamis. Pada tahap lanjut, ia akan memanggil endpoint FastAPI yang secara langsung membaca file `traffic_logs_buahbatu.csv` buatan YOLOv8, menghitung agregasi volume per jam menggunakan `pandas`, dan mengembalikannya ke React dalam wujud larik JSON agar bisa digambar oleh _Recharts_.

---

## Section 7: Insight AI (Generasi Wawasan Otomatis)
* **Arah Integrasi:** `Frontend` ➔ `Backend (GET /api/insight)` ➔ `LLM (Groq/Gemini API)`.
* **Fungsi:** Menerjemahkan angka-angka analitik yang kaku menjadi paragraf yang mudah dipahami manusia.
* **Penjelasan:** Setelah data CSV terkumpul, frontend dapat melakukan *request* ke endpoint backend. Backend kemudian mengambil ringkasan data lalu lintas hari ini, mengirimkannya melalui integrasi pihak ketiga (misalnya `groq` atau `google-generativeai`), lalu menyajikan kesimpulannya ke komponen React. Tampilannya otomatis di-blok/dikosongkan (`idle`) sebelum data ini tersedia.

---

## Section 8: Traffic Assistant (Chatbot AI)
* **Arah Integrasi:** `Frontend` ➔ `Backend (POST /api/chat)` ➔ `RAG Pipeline / LLM`.
* **Fungsi:** Asisten analitik interaktif dua arah.
* **Penjelasan:** Setiap pengguna mengetik pertanyaan dan menekan "Kirim", frontend akan mendorong string pesan tersebut ke backend. Di backend, instruksi tersebut dapat diolah mencari jawaban langsung dari database lalu lintas atau dihubungkan ke asisten LLM.

---

## Section 9: Upload History
* **Arah Integrasi:** `Frontend` ➔ `Backend (Google Drive API / Database)`.
* **Fungsi:** Sinkronisasi video unggahan dengan sistem *Cloud*.
* **Penjelasan:** Seksi ini bertugas mengambil rekaman data video yang pernah dianalisis. Komponen ini dirancang untuk memanggil endpoint FastAPI (contoh `/api/drive/history`) yang berisi skrip otentikasi Google Drive (`google-api-python-client`). Data nama file, URL g-drive, dan status proses akan dikirimkan kembali ke Frontend untuk ditampilkan dalam bentuk tabel riwayat.

---

**Kesimpulan:**
Desain aplikasi Anda saat ini menganut pola **Single Source of Truth** berbasis polling dan streaming. Pusat data berada di _Backend (FastAPI + YOLOv8)_, sedangkan _Frontend (React)_ bertindak sebagai layar pintar yang secara konstan menarik (pulling) status terbaru dari `127.0.0.1:8000` tanpa perlu memuat ulang (*refresh*) halaman peramban web.
