from flask import Flask, request, jsonify
import cv2
import numpy as np
import sqlite3
import os

app = Flask(__name__)

# I-load ang iyong LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml') # Ang dati mong trained file
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/scan_face', methods=['POST'])
def scan_face():
    try:
        # 1. Tatanggapin ang image file mula sa Android (Flutter)
        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        
        if len(faces) == 0:
            return jsonify({"status": "error", "message": "Walang mukhang nakita. Humarap nang maayos."})
            
        for (x, y, w, h) in faces:
            # 2. Kilalanin ang mukha gamit ang LBPH
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            
            # Kung pasok sa accuracy threshold (halimbawa: confidence < 100)
            if confidence < 80:
                # Kuhanin ang pangalan sa database gamit ang ID
                conn = sqlite3.connect("attendance.db")
                cursor = conn.cursor()
                
                # I-insert ang record bilang 'Present'
                # (Baguhin ang query depende sa table structure mo)
                cursor.execute("INSERT INTO attendance (name, date, time, status) VALUES (?, date('now'), time('now'), 'Present')", (str(id),))
                conn.commit()
                conn.close()
                
                return jsonify({"status": "success", "student_id": id, "message": "Attendance Recorded!"})
            else:
                return jsonify({"status": "unknown", "message": "Hindi kilala ang mukha."})
                
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    # Patatakbuhin ang server sa local network niyo
    # I-type ang 'ipconfig' sa cmd para makuha ang IPv4 address ng laptop mo
    app.run(host='0.0.0.0', port=5000, debug=True)