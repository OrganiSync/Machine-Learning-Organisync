import cv2
import time
import os
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import logging
from pathlib import Path
import re  # Untuk validasi email

class FaceDataCollector:
    def __init__(self, root):
        self.root = root
        self.setup_logging()
        self.initialize_ui()
        
    def setup_logging(self):
        logging.basicConfig(
            filename='face_collection.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def initialize_ui(self):
        self.root.title("Pengambilan Data Wajah")
        self.setup_window_geometry()
        self.setup_styles()
        self.create_widgets()
        
    def setup_window_geometry(self):
        window_width = 400
        window_height = 500
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
    def setup_styles(self):
        self.root.configure(bg='#f0f0f0')
        self.style = ttk.Style()
        self.style.configure('TButton', padding=10, font=('Helvetica', 12))
        self.style.configure('TLabel', background='#f0f0f0', font=('Helvetica', 12))
        self.style.configure('TEntry', font=('Helvetica', 12))
        
    def create_widgets(self):
        self.main_frame = Frame(self.root, bg='#f0f0f0', padx=20, pady=20)
        self.main_frame.pack(expand=True, fill=BOTH)
        
        # Create title
        self.create_title()
        
        # Create input frame
        self.create_input_frame()
        
        # Create instructions
        self.create_instructions()
        
    def create_title(self):
        title_label = Label(
            self.main_frame, 
            text="Pengambilan Data Wajah",
            font=("Helvetica", 18, "bold"),
            bg='#f0f0f0'
        )
        title_label.pack(pady=(0, 20))
        
    def create_input_frame(self):
        input_frame = Frame(self.main_frame, bg='#f0f0f0')
        input_frame.pack(fill=X, pady=10)
        
        # Email input
        self.create_labeled_entry(input_frame, "Email:", "email_entry")
        
        # Name input
        self.create_labeled_entry(input_frame, "Nama:", "name_entry")
        
        # Start button
        self.create_start_button(input_frame)
        
    def create_labeled_entry(self, parent, label_text, entry_name):
        label = Label(
            parent, 
            text=label_text,
            bg='#f0f0f0',
            font=("Helvetica", 12)
        )
        label.pack(anchor=W)
        
        entry = Entry(parent, font=("Helvetica", 12))
        entry.pack(fill=X, pady=(5, 10))
        setattr(self, entry_name, entry)
        
    def create_start_button(self, parent):
        start_button = Button(
            parent,
            text="Mulai Pengambilan Data",
            command=self.capture_faces,
            font=("Helvetica", 12, "bold"),
            bg='#4CAF50',
            fg='white',
            activebackground='#45a049',
            activeforeground='white',
            relief=RAISED,
            padx=20,
            pady=10
        )
        start_button.pack(pady=10)
        
    def create_instructions(self):
        instructions = """
        Petunjuk Penggunaan:
        1. Masukkan Email dan Nama
        2. Klik tombol 'Mulai Pengambilan Data'
        3. Posisikan wajah di depan kamera
        4. Tunggu hingga 100 gambar terambil
        5. Atau tekan 'q' untuk berhenti
        """
        instruction_label = Label(
            self.main_frame,
            text=instructions,
            justify=LEFT,
            bg='#f0f0f0',
            font=("Helvetica", 11)
        )
        instruction_label.pack(pady=20)
        
    def validate_inputs(self):
        email = self.email_entry.get().strip()
        user_name = self.name_entry.get().strip()
        
        if not email or not user_name:
            messagebox.showerror("Error", "Email dan Nama harus diisi!")
            return False
        
        if not self.is_valid_email(email):
            messagebox.showerror("Error", "Format Email tidak valid!")
            return False
        
        return email, user_name
        
    def is_valid_email(self, email):
        # Regex untuk validasi email sederhana
        regex = r'^\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.match(regex, email)
        
    def capture_faces(self):
        inputs = self.validate_inputs()
        if not inputs:
            return
            
        email, user_name = inputs
        try:
            self.setup_camera_and_capture(email, user_name)
        except Exception as e:
            logging.error(f"Error during face capture: {str(e)}")
            messagebox.showerror("Error", f"Terjadi kesalahan: {str(e)}")
        finally:
            self.clear_entries()
            
    def setup_camera_and_capture(self, email, user_name):
        # Initialize camera and face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        face_detection = cv2.CascadeClassifier(cascade_path)
        
        if face_detection.empty():
            raise Exception("Couldn't load face cascade classifier!")
            
        # Pastikan direktori dataset ada
        dataset_dir = Path('faceDataset')
        dataset_dir.mkdir(exist_ok=True)
        
        # Rekam info user
        with open('userinfo.txt', 'a') as f:
            f.write(f"{email},{user_name}\n")
            
        self.perform_face_capture(video, face_detection, dataset_dir, email)
        
    def perform_face_capture(self, video, face_detection, dataset_dir, email):
        image_count = 0
        while True:
            success, frame = video.read()
            
            if not success:
                raise Exception("Tidak dapat membaca kamera!")
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detection.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                # Crop area wajah
                face_img = gray[y:y+h, x:x+w]  # Pastikan ini grayscale
                
                # Buat nama file berdasarkan email yang diformat (hindari karakter ilegal)
                safe_email = email.replace('@', '_at_').replace('.', '_')
                image_path = dataset_dir / f'User.{safe_email}.{image_count}.jpg'
                cv2.imwrite(str(image_path), face_img)
                
                # Gambar persegi panjang dan teks
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Gambar ke: {image_count}/100", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Tekan 'q' untuk keluar", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                image_count += 1
                if image_count >=100:
                    break
                
            cv2.imshow("Pengambilan Data Wajah", frame)
            
            if cv2.waitKey(100) & 0xFF == ord('q') or image_count >= 100:
                break

        video.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Sukses", f"Berhasil mengambil {image_count} gambar wajah")
        
    def clear_entries(self):
        self.email_entry.delete(0, END)
        self.name_entry.delete(0, END)

if __name__ == "__main__":
    root = Tk()
    app = FaceDataCollector(root)
    root.mainloop()
