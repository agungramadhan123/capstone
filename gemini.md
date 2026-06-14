# PROJECT REQUIREMENT

Buat sebuah aplikasi web modern bernama **Smart Traffic Bandung** yang berfungsi untuk melakukan monitoring dan analisis lalu lintas menggunakan teknologi Computer Vision.

Aplikasi harus memiliki tampilan profesional setara produk SaaS modern (seperti Datadog, Grafana Cloud, Vercel Dashboard, atau Smart City Command Center).

Gunakan desain dark theme dengan nuansa biru gelap, neon blue, dan aksen hijau, kuning, merah untuk indikator lalu lintas.

---

# TECH STACK

Frontend:

* React 19
* Vite
* TypeScript
* Tailwind CSS
* Shadcn UI
* Framer Motion
* Lucide React
* React Query
* Recharts
* React Hook Form
* Zustand

Maps:

* Leaflet
* OpenStreetMap

Animation:

* Framer Motion
* CSS Keyframes

Backend:

* FastAPI
* Python 3.11+

Storage:

* Google Drive API

Computer Vision:

* YOLOv11
* ByteTrack

Deployment Ready:

* Docker
* Docker Compose
* akan di deploy ke vercel

---

# IMPORTANT

Saat aplikasi pertama kali dijalankan:

1. Cek seluruh dependency frontend.
2. Cek seluruh dependency backend.
3. Jika package belum tersedia maka install otomatis.
4. Buatkan file requirements.txt.
5. Buatkan package.json lengkap.
6. Jangan gunakan package deprecated.
7. Gunakan struktur project production-ready.

---

# UI CONCEPT

Gunakan layout modular vertikal.

Jangan membuat dashboard infografis satu layar.

Halaman harus dapat discroll ke bawah.

Urutan section:

1. Hero Section
2. Input Source
3. Live Monitoring
4. Traffic Status
5. Map Monitoring
6. Analytics
7. Insight AI
8. Traffic Assistant
9. Upload History

---

# SECTION 1

HERO SECTION

Tampilkan:

Smart Traffic Bandung

Deskripsi:

Platform berbasis Artificial Intelligence untuk mendeteksi, melacak, dan menghitung kendaraan dari CCTV maupun video unggahan guna membantu analisis lalu lintas secara real-time.

Tambahkan ilustrasi smart city modern.

---

# SECTION 2

INPUT SOURCE

Tampilkan dua pilihan:

Live CCTV
Upload Video

Jika memilih Live CCTV:

Munculkan tabel kamera.

Kolom:

* ID
* Lokasi
* Status
* Last Active
* Action

Tombol:

Monitor

Jika memilih Upload Video:

Munculkan uploader.

Format:

MP4
AVI
MOV

Maksimal:

2GB

---

# GOOGLE DRIVE INTEGRATION

Tambahkan integrasi Google Drive.

Alur:

Upload Video
↓
Simpan ke Google Drive
↓
Simpan metadata ke database
↓
Tampilkan histori upload

Tampilkan:

* Nama video
* Tanggal upload
* Durasi
* Status
* Link Google Drive

Tambahkan tombol:

Open Drive Folder

---

# SECTION 3

LIVE MONITORING

Tampilkan video player besar.

Tampilkan bounding box hasil deteksi.

Label:

Mobil
Motor
Bus
Truk

Tampilkan confidence score.

---

# SECTION 4

TRAFFIC STATUS

Status tidak boleh muncul sebelum proses analisis berjalan.

Gunakan state:

idle
processing
completed

Jika idle:

"Menunggu Analisis"

Jika processing:

"Sedang Menganalisis"

Jika completed:

Tampilkan:

Lancar
Sedang
Padat
Sangat Padat

Gunakan indikator lampu lalu lintas.

---

# SECTION 5

MAP MONITORING

Gunakan Leaflet.

Tampilkan marker CCTV Bandung.

Ketika marker diklik:

Tampilkan:

* Nama kamera
* Status
* Kendaraan saat ini
* Tingkat kepadatan

---

# SECTION 6

TRAFFIC ANALYTICS

Gunakan Recharts.

Tampilkan:

1. Line Chart
2. Donut Chart

Pada line chart:

Cari nilai maksimum.

Tambahkan marker khusus.

Tambahkan label:

Peak Traffic

Contoh:

18.00 WIB
1842 kendaraan

---

# SECTION 7

INSIGHT AI

Generate insight otomatis.

Contoh:

Jam tersibuk terjadi pada pukul 18.00 WIB dengan volume 1842 kendaraan.

Volume kendaraan meningkat 32% dibanding rata-rata harian.

Insight harus dibuat berdasarkan data nyata dari hasil analisis.

Jangan menggunakan data dummy setelah backend tersedia.

---

# SECTION 8

TRAFFIC ASSISTANT

Buat chatbot khusus analitik lalu lintas.

Bukan chatbot umum.

Contoh pertanyaan:

* Jam berapa paling padat hari ini?
* Kamera mana yang paling ramai?
* Bagaimana tren minggu ini?
* Bandingkan Dago dan Pasteur.

Chatbot harus menggunakan data sistem.

---

# ANIMATION REQUIREMENTS

Gunakan animasi secukupnya.

Jangan berlebihan.

1. Scroll Reveal

Semua card muncul menggunakan:

* fade in
* slide up

saat masuk viewport.

Gunakan Framer Motion.

---

2. Traffic Light Animation

Lampu lalu lintas:

* glow effect
* pulse effect

ketika status berubah.

---

3. CCTV Animation

Icon CCTV:

* bergerak kiri kanan perlahan
* durasi 4-6 detik
* infinite

---

4. Weather Animation

Jika hujan:

* animasi rintik hujan ringan

Jika cerah:

* animasi sinar matahari glow

Jika mendung:

* animasi awan bergerak pelan

---

5. Live Indicator

Badge LIVE:

* berkedip halus
* warna merah

---

6. KPI Cards

Hover:

* scale 1.03
* smooth transition

---

7. Chart Animation

Saat data pertama kali muncul:

* line chart draw animation
* smooth transition

---

# PERFORMANCE REQUIREMENTS

* Responsive desktop
* Responsive tablet
* Responsive mobile

Lazy loading:

* maps
* charts
* chatbot

Gunakan code splitting.

Target Lighthouse:

Performance > 90
Accessibility > 90

---

# CODE QUALITY

* TypeScript strict mode
* Modular architecture
* Reusable components
* Clean code
* SOLID principles
* Error handling lengkap
* Loading states lengkap
* Empty states lengkap
* Buat kode modular

Generate seluruh struktur project dan implementasi secara production-ready.
