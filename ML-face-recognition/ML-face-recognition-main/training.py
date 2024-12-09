import cv2
import numpy as np
import os

# Path ke folder dataset
dataset_path = 'faceDataset'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Fungsi untuk membaca data dan label dari folder
def load_dataset(path):
    images = []
    labels = []
    label_dict = {}
    current_id = 0
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("jpg") or file.endswith("png"):
                image_path = os.path.join(root, file)
                label = os.path.basename(root)  # Nama folder sebagai label
                
                if label not in label_dict:
                    label_dict[label] = current_id
                    current_id += 1
                
                id_ = label_dict[label]
                
                # Baca gambar dalam grayscale
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                # Deteksi wajah
                faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
                
                for (x, y, w, h) in faces:
                    roi = image[y:y+h, x:x+w]  # Region of Interest (wajah saja)
                    images.append(roi)
                    labels.append(id_)
    
    return images, labels, label_dict

# Load dataset
print("Loading dataset...")
faces, ids, label_dict = load_dataset(dataset_path)
faces = [cv2.resize(face, (200, 200)) for face in faces]  # Resize wajah ke ukuran tetap

# Melatih model pengenalan wajah
print("Training model...")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(ids))

# Menyimpan model ke file training.xml
recognizer.save('faceDataset/training.xml')
print("Training selesai. Model disimpan sebagai 'training.xml'.")
