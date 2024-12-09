import cv2
import numpy as np
import os
from PIL import Image

# Load the pre-trained LBPH model
model_path = 'faceDataset/training.xml'
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

# Untuk detector menggunakan file haarcascade_frontalface_default.xml
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Membaca file userinfo.txt dan membuat kamus ID ke nama
id_to_name = {}
with open('userinfo.txt', 'r') as f:
    for line in f:
        email, user_name = line.strip().split(',')
        id_to_name[str(email)] = user_name

# Menggunakan webcam bawaan (0) atau external
camera = 0

# Inisialisasi video capture
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

# Cek apakah kamera berhasil dibuka
if not video.isOpened():
    print("Error: Tidak dapat membuka kamera!")
    exit()

print("Program face recognition berjalan. Tekan 'q' untuk keluar.")

# Loop utama
while True:
    # Membaca frame dari kamera
    check, frame = video.read()
    
    # Cek apakah frame berhasil dibaca
    if not check:
        print("Error: Tidak dapat membaca frame dari kamera!")
        break
    
    # Konversi ke grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah menggunakan OpenCV
    faces = detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Proses setiap wajah yang terdeteksi
    for (x, y, w, h) in faces:
        # Gambar kotak hijau di wajah
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Potong area wajah
        face_region = gray_frame[y:y+h, x:x+w]

        # Prediksi ID menggunakan LBPH recognizer
        email, confidence = recognizer.predict(face_region)

        # Dapatkan nama berdasarkan ID (misalnya, id_to_name[email])
        name = id_to_name.get(email, user_name)

        # Menampilkan nama dan confidence score
        label = f"{name} - {confidence:.2f}%"
        cv2.putText(frame, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Tampilkan frame dengan wajah yang terdeteksi
    cv2.imshow("Face Recognition", frame)

    # Cek keyboard event
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Bersihkan
video.release()
cv2.destroyAllWindows()
