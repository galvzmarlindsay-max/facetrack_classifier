import cv2
import numpy as np
from PIL import Image
import os

path = 'dataset'

# ✅ CHECK kung may dataset folder
if not os.path.exists(path):
    print("❌ ERROR: dataset folder not found!")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    faceSamples = []
    ids = []
    
    # ✅ FIX: Binago ang mga susi (Keys) para tugma sa totoong folder names mo!
    names_map = {
        "1_Hazel Mae": 1,
        "2_Fredirick": 2,
        "3_Roxan": 3,
        "4_Kristina": 4,
        "5_Meaann": 5,
        "6_Arjie": 6,
        "7_Hinayon": 7,
        "8_Brithny": 8,
        "9_Lindsay Galvez": 9,
        "10_Jenelyn": 10
    }

    print("🔄 Binabasa ang dataset folders...")

    # I-loop ang bawat folder sa loob ng dataset
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                # Kunin ang pangalan ng folder bilang label (hal. "1_Hazel Mae")
                label_name = os.path.basename(root)
                
                # Siguraduhing kasama ang folder name sa ating listahan
                if label_name in names_map:
                    id = names_map[label_name]
                    imagePath = os.path.join(root, file)
                    
                    try:
                        # I-convert sa grayscale
                        PIL_img = Image.open(imagePath).convert('L')
                        img_numpy = np.array(PIL_img, 'uint8')
                        
                        # Detect faces
                        faces = detector.detectMultiScale(img_numpy)

                        for (x, y, w, h) in faces:
                            faceSamples.append(img_numpy[y:y+h, x:x+w])
                            ids.append(id)
                            print(f"✅ Trained image: {imagePath} (Assigned ID: {id})")
                            
                    except Exception as e:
                        print(f"⚠️ Error processing {imagePath}: {e}")
                        continue
                else:
                    # Iwasan ang babala para sa main root folder
                    if label_name != "dataset" and not label_name.startswith("dataset"):
                        print(f"⚠️ Warning: Ang folder na '{label_name}' ay wala sa listahan ng names_map. Skipping...")

    print("\n👥 Name Mapping Used:", names_map)
    return faceSamples, ids

print("🔄 Training faces...")

faces, ids = getImagesAndLabels(path)

# ✅ CHECK kung may na-detect na faces
if len(faces) == 0:
    print("❌ ERROR: Walang mukhang nakita sa loob ng dataset! Siguraduhing tama ang mukha sa mga litrato.")
    exit()

recognizer.train(faces, np.array(ids))

# ✅ GINAWANG .XML IMRES NA .YML PARA MAIWASAN ANG PARSING ERROR SA GITHUB/STREAMLIT
recognizer.save('trainer.xml')

print(f"\n🎉 Training Complete! {len(np.unique(ids))} individual(s) trained successfully.")
print("📁 'trainer.xml' created successfully!")