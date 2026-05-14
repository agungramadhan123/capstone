import supervision as sv
import cv2 
import os

# 1. Load dataset
dataset = sv.DetectionDataset.from_yolo(
    images_directory_path="train/images",
    annotations_directory_path="train/labels",
    data_yaml_path="data.yaml"
)

folder_simpan = "Hasil_labelling"
os.makedirs(folder_simpan, exist_ok=True)

# 2. Siapkan D`UA annotator: Satu untuk kotak, satu untuk teks
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

print(f"Ditemukan {len(dataset)} gambar. Mulai memproses...")
print(f"Daftar kelas yang terdeteksi di data.yaml: {dataset.classes}")

for nama, image, detection in dataset:
    # Buat salinan gambar agar gambar asli tidak tertimpa di memori
    gambar_proses = image.copy()
    
    # 3. Buat daftar teks label untuk setiap objek yang terdeteksi di gambar ini
    # Ini akan mengambil nama kelas dari data.yaml berdasarkan ID-nya
    labels = [
        f"{dataset.classes[class_id]}" 
        for class_id in detection.class_id
    ]

    # 4. Gambar KOTAK-nya dulu
    gambar_proses = box_annotator.annotate(
        scene=gambar_proses, 
        detections=detection
    )
    
    # 5. Gambar TEKS LABEL-nya di atas kotak
    gambar_proses = label_annotator.annotate(
        scene=gambar_proses, 
        detections=detection, 
        labels=labels
    )
    
    # Ambil nama file dan pastikan ekstensinya benar
    nama_file_bersih = os.path.basename(nama)
    if not (nama_file_bersih.lower().endswith('.jpg') or nama_file_bersih.lower().endswith('.png')):
        nama_file_bersih = f"{nama_file_bersih}.jpg"
        
    path_simpan = os.path.join(folder_simpan, nama_file_bersih)
    
    # Simpan gambar yang sudah ada kotak + teks
    berhasil = cv2.imwrite(path_simpan, gambar_proses)
    
    if berhasil:
        print(f"Tersimpan dengan label: {path_simpan}")
    else:
        print(f"GAGAL MENYIMPAN: {path_simpan}")

print("Proses selesai! Silakan cek folder Hasil_labelling.")