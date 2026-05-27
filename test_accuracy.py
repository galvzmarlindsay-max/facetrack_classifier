import os
import cv2
import numpy as np

# 1. CONFIGURATION
DATASET_DIR = r"C:\xampp\htdocs\facetrack\dataset"
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml") # I-load ang inyong trained model
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Itakda ang threshold na ginagamit niyo sa app (50%)
CONFIDENCE_THRESHOLD = 50

total_tests = 0
correct_recognitions = 0

print("🔍 Nagsisimula na ang Accuracy Testing... Pakihintay.")

student_folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]

for student_name in student_folders:
    try:
        actual_id = int(student_name.split('_')[0])
    except:
        continue
        
    student_path = os.path.join(DATASET_DIR, student_name)
    
    for image_name in os.listdir(student_path):
        image_path = os.path.join(student_path, image_name)
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if gray_img is None:
            continue
            
        faces = face_detector.detectMultiScale(gray_img)
        
        for (x, y, w, h) in faces:
            total_tests += 1
            
            # Hulaan ng AI kung kaninong mukha ito
            predicted_id, confidence = recognizer.predict(gray_img[y:y+h, x:x+w])
            match_perc = round(100 - confidence)
            
            # Tingnan kung tama ang hula at lagpas sa threshold niyo
            if match_perc >= CONFIDENCE_THRESHOLD and predicted_id == actual_id:
                correct_recognitions += 1

# 2. KALKULAHIN ANG ACCURACY
if total_tests > 0:
    accuracy_rate = (correct_recognitions / total_tests) * 100
    print("\n=============================================")
    print("📊 FINAL ACCURACY REPORT FOR THESIS DEMO")
    print("=============================================")
    print(f"Total Faces Evaluated : {total_tests}")
    print(f"Correctly Recognized  : {correct_recognitions}")
    print(f"System Accuracy Rate  : {accuracy_rate:.2f}%")
    print("=============================================")
else:
    print("❌ Walang mukha na na-detect sa dataset upang ma-test.")