import os
import cv2
import numpy as np

# 1. CONFIGURATION
DATASET_DIR = r"C:\xampp\htdocs\facetrack\dataset"
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_images_and_labels(path):
    face_samples = []
    ids = []
    
    # Kuhanin ang listahan ng folders
    student_folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    
    label_map = {}
    
    for student_name in student_folders:
        # --- INSERTED ID MAPPING LOGIC ---
        # Kinukuha ang numero sa unahan ng folder name (e.g., "10_Jenelyn" -> ID 10)
        try:
            current_id = int(student_name.split('_')[0])
        except (ValueError, IndexError):
            print(f"⚠️ Skipping '{student_name}': Dapat magsimula sa numero (e.g., 1_Name)")
            continue
        # ----------------------------------

        label_map[current_id] = student_name
        student_path = os.path.join(path, student_name)
        
        print(f"📂 Processing: {student_name} (Assigned ID: {current_id})")
        
        for image_name in os.listdir(student_path):
            image_path = os.path.join(student_path, image_name)
            
            # Basahin ang imahe at i-grayscale
            gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if gray_img is None:
                continue
                
            # I-detect ang mukha para malinis ang training data
            faces = face_detector.detectMultiScale(gray_img)
            
            for (x, y, w, h) in faces:
                # I-crop ang mukha at idagdag sa listahan
                face_samples.append(gray_img[y:y+h, x:x+w])
                ids.append(current_id)
                
    return face_samples, np.array(ids), label_map

print("🚀 Nagsugod na ang pag-load sa mga hulagway...")
faces, ids, label_map = get_images_and_labels(DATASET_DIR)

if len(faces) == 0:
    print("❌ Error: Walay nakit-an nga mga nawong sa folder! Siguroha ang path.")
else:
    print(f"🧠 Gi-train na ang LBPH gamit ang {len(faces)} ka mga face samples...")
    recognizer.train(faces, ids)
    
    # I-save ang trained model
    recognizer.write('trainer.yml')
    print("✅ Done! Na-save na ang 'trainer.yml'.")
    
    print("\n📋 Kini ang inyong Student ID mapping:")
    for id_key in sorted(label_map.keys()):
        print(f"ID {id_key}: {label_map[id_key]}")