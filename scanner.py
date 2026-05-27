import cv2
import sqlite3
from datetime import datetime
import time
import streamlit as st
import numpy as np
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# Global/Session initialization para sa models para hindi paulit-ulit na i-load
if "recognizer" not in st.session_state:
    try:
        if os.path.exists("trainer.xml"):
            st.session_state.recognizer = cv2.face.LBPHFaceRecognizer_create()
            st.session_state.recognizer.read("trainer.xml")
            st.session_state.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            st.session_state.model_loaded = True
        else:
            st.session_state.model_loaded = False
    except Exception as e:
        st.session_state.model_loaded = False

# Configuration base sa iyong dataset folders
NAMES_MAP = [
    "None",             # ID 0
    "Hazel Mae",        # ID 1
    "Fredirick",        # ID 2
    "Roxan",            # ID 3
    "Kristina",         # ID 4
    "Meaann",           # ID 5
    "Arjie",            # ID 6
    "Hinayon",          # ID 7
    "Brithny",          # ID 8
    "Lindsay Galvez",   # ID 9
    "Jenelyn"           # ID 10
]

def mark_attendance(name):
    """Isinusulat ang attendance sa SQLite Database"""
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    cursor.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, date))
    if cursor.fetchone() is None:
        cursor.execute(
            "INSERT INTO attendance (name, date, time, status) VALUES (?, ?, ?, ?)",
            (name, date, current_time, "Present")
        )
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False

def mark_absent(detected_students):
    """Minamarkahan bilang Absent ang mga hindi nakita sa buong session"""
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    all_students = NAMES_MAP[1:]
    for student in all_students:
        if student not in detected_students:
            cursor.execute("SELECT * FROM attendance WHERE name=? AND date=?", (student, date))
            if cursor.fetchone() is None:
                cursor.execute(
                    "INSERT INTO attendance (name, date, time, status) VALUES (?, ?, ?, ?)",
                    (student, date, current_time, "Absent")
                )
    conn.commit()
    conn.close()

class FaceRecognizerTransformer(VideoTransformerBase):
    def __init__(self, confidence_threshold):
        self.confidence_threshold = confidence_threshold
        self.last_mark_time = 0
        self.check_display_expiry = 0
        self.detected_students = set()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Siguraduhing naka-load ang modelo
        if not st.session_state.get("model_loaded", False):
            return frame

        img = cv2.flip(img, 1) # Mirror effect
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = st.session_state.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100)
        )

        # Basahin ang UI prediction request mula sa session state
        do_predict = st.session_state.get("do_prediction", False)

        for (x, y, w, h) in faces:
            name = "Ready to Scan"
            color = (255, 255, 255)

            if do_predict:
                name = "Scanning..."
                color = (0, 255, 255) # Yellow

                try:
                    face_roi_gray = gray[y:y+h, x:x+w]
                    id_predicted, confidence = st.session_state.recognizer.predict(face_roi_gray)
                    match_perc = round(max(0, 100 - confidence))

                    if match_perc >= self.confidence_threshold:
                        if id_predicted < len(NAMES_MAP):
                            name = NAMES_MAP[id_predicted]
                            self.detected_students.add(name)
                            
                            # I-save sa background thread ang attendance
                            current_time = time.time()
                            if current_time - self.last_mark_time > 5:
                                if mark_attendance(name):
                                    self.check_display_expiry = current_time + 1.5
                                self.last_mark_time = current_time

                            name = f"{name} ({match_perc}%)"
                            color = (0, 255, 0) # Green
                        else:
                            name = "Unknown ID"
                            color = (0, 0, 255)
                    else:
                        name = f"Unknown ({match_perc}%)"
                        color = (0, 0, 255) # Red

                    # I-update ang huling resulta para sa display persistence
                    st.session_state.last_result = name
                    st.session_state.last_color = color

                except Exception as e:
                    print(f"Prediction Error: {e}")
            else:
                name = st.session_state.get("last_result", "Ready")
                color = st.session_state.get("last_color", (255, 255, 255))

            cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
            cv2.putText(img, name, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # I-reset ang prediction trigger pagkatapos ng isang ikot
        if do_predict:
            st.session_state.do_prediction = False

        # Draw Success Checkmark
        if time.time() < self.check_display_expiry:
            cv2.line(img, (260, 260), (300, 300), (0, 255, 0), 15)
            cv2.line(img, (300, 300), (390, 190), (0, 255, 0), 15)
            cv2.putText(img, "SUCCESS", (240, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)

        return frame.from_ndarray(img, format="bgr24")

def start_scanner(placeholder, confidence_threshold):
    """Ito ang tinatawag ng iyong main app.py para ilunsad ang camera"""
    
    if not st.session_state.get("model_loaded", False):
        st.error("❌ Error: 'trainer.xml' hindi matagpuan o may sira! Patakbuhin muna ang train.py.")
        return False

    st.write("### 📷 Live Classroom WebRTC Scanner")
    st.info("💡 Pwede itong buksan sa cellphone! Pindutin ang 'Start' para buksan ang camera.")

    # Inilulunsad ang webrtc object streamer interface
    ctx = webrtc_streamer(
        key="face-recognition-scanner",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        video_transformer_factory=lambda: FaceRecognizerTransformer(confidence_threshold),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Trigger button para sa pagkilala ng mukha habang umaandar ang stream
    if ctx.state.playing:
        if st.button("📸 Capture & Identify Student", use_container_width=True):
            st.session_state.do_prediction = True
            
    # Kapag pinatay ang camera, isulat ang mga absent students
    if not ctx.state.playing and "face-recognition-scanner" in st.session_state:
        if hasattr(ctx.video_transformer, 'detected_students'):
            mark_absent(ctx.video_transformer.detected_students)

    return True