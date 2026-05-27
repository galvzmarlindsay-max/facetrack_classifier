import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
# I-import ang in-enhance nating scanner function
from scanner import start_scanner

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Student Tracking System",
    page_icon="🎓",
    layout="wide"
)

# Siguraduhing may initial state ang mga kailangan natin
if "do_prediction" not in st.session_state:
    st.session_state.do_prediction = False
if "last_result" not in st.session_state:
    st.session_state.last_result = "Ready"
if "last_color" not in st.session_state:
    st.session_state.last_color = (255, 255, 255)

# HELPER FUNCTION: Kumuha ng data mula sa SQLite para sa History
def get_attendance_data():
    conn = sqlite3.connect("attendance.db")
    # Gumawa ng table kung wala pa para maiwasan ang error
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance 
                      (name TEXT, date TEXT, time TEXT, status TEXT)''')
    conn.commit()
    
    # Basahin ang data
    df = pd.read_sql_query("SELECT name as 'Student Name', date as 'Date', time as 'Time Logged', status as 'Status' FROM attendance ORDER BY date DESC, time DESC", conn)
    conn.close()
    return df

# 2. SIDEBAR NAVIGATION
with st.sidebar:
    st.title("🛡️ Admin Panel")
    st.subheader("active teacher: Admin")
    st.write("---")
    
    # Inayos na Radio Button (May Label at maayos ang Options base sa huling error)
    choice = st.radio(
        "Navigation", 
        ["🏠 Home", "📷 Face Scanner", "📋 History Record"],
        index=0
    )
    
    st.write("---")
    # Button para i-stop ang scanner kung sakaling mag-hang
    if st.button("🛑 Force Stop Scanner", use_container_width=True):
        st.session_state.scanner_running = False
        st.rerun()

# =====================================================
# TAB 1: HOME / DASHBOARD
# =====================================================
if choice == "🏠 Home":
    st.title("Welcome to Student Tracking System 🎓")
    st.write(f"📅 **Petsa Ngayon:** {datetime.now().strftime('%B %d, %Y')}")
    st.write("---")
    
    # Kumuha ng Quick Stats mula sa Database
    df_stats = get_attendance_data()
    today_str = datetime.now().strftime("%Y-%m-%d")
    df_today = df_stats[df_stats['Date'] == today_str] if not df_stats.empty else pd.DataFrame()
    
    # Dashboard Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        total_logs = len(df_today)
        st.metric(label="Total Scans Today", value=total_logs)
    with col2:
        present_count = len(df_today[df_today['Status'] == 'Present']) if not df_today.empty else 0
        st.metric(label="Present Students", value=present_count)
    with col3:
        absent_count = len(df_today[df_today['Status'] == 'Absent']) if not df_today.empty else 0
        st.metric(label="Absent Students", value=absent_count)
        
    st.write("---")
    st.info("💡 **Paano Gamitin:** Pumunta sa **Face Scanner** tab sa kaliwa para magsimulang kumuha ng attendance ng mga estudyante gamit ang webcam.")

# =====================================================
# TAB 2: FACE SCANNER MODE
# =====================================================
elif choice == "📷 Face Scanner":
    st.title("📷 Face Scanner Mode")
    st.write("I-tapat ang mukha ng estudyante sa camera at i-click ang scan button.")
    st.write("---")
    
    # Slider para sa Confidence Threshold
    confidence_slider = st.slider("Confidence Threshold (%)", min_value=10, max_value=100, value=40, step=5)
    
    # Layout para sa Camera at Control Buttons
    col_cam, col_ctrl = st.columns([2, 1])
    
    with col_cam:
        st.subheader("Webcam Live Feed")
        # Dito papasok ang video feed mula sa scanner.py
        video_placeholder = st.empty()
        
    with col_ctrl:
        st.subheader("Controls")
        st.write("Gamitin ang mga button sa ibaba para mag-trigger ng facial recognition.")
        
        # Pangunahing Button para mag-scan
        if st.button("📸 Capture & Identify Student", type="primary", use_container_width=True):
            st.session_state.do_prediction = True
            
        st.write("---")
        st.write("**Current Status:**")
        # Ipakita kung ano ang huling resulta ng scan sa gilid
        res = st.session_state.get("last_result", "Ready")
        if "FOUND" in res or "Ready" not in res and "Unknown" not in res:
            st.success(f"Result: {res}")
        elif "Unknown" in res:
            st.error(f"Result: {res}")
        else:
            st.info(f"Status: {res}")

    # PATAKBUHIN ANG SCANNER LOOP
    # Ang loop na ito ay hindi mag-ba-block dahil gumagamit ito ng video_placeholder
    start_scanner(video_placeholder, confidence_slider)

# =====================================================
# TAB 3: HISTORY RECORD
# =====================================================
elif choice == "📋 History Record":
    st.title("📋 Attendance History Record")
    st.write("Narito ang listahan ng mga estudyanteng pumasok at na-record sa system.")
    st.write("---")
    
    # Kumuha ng pinakahuling data
    df = get_attendance_data()
    
    if df.empty:
        st.warning("⚠️ Wala pang laman ang attendance database sa kasalukuyan.")
    else:
        # Search Bar Feature
        search_query = st.text_input("🔍 Maghanap ng Estudyante sa Pangalan:")
        if search_query:
            df = df[df['Student Name'].str.contains(search_query, case=False, na=False)]
            
        # I-display ang Table sa Streamlit
        st.dataframe(df, use_container_width=True)
        
        # Export Feature sa CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Attendance Report (CSV)",
            data=csv,
            file_name=f"Attendance_Report_{datetime.now().strftime('%Y-%m-%d')}.csv",
            mime='text/csv',
            use_container_width=True
        )