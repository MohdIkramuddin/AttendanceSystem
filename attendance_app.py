import cv2
import face_recognition
import numpy as np
import sqlite3
import datetime
import pickle
import os
import csv
import io
from flask import Flask, render_template_string, request, redirect, url_for, Response, send_file, flash

# Indian Time Zone (IST = UTC+5:30)
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

# ==========================================
# CONFIGURATION & DATABASE SETUP
# ==========================================
app = Flask(__name__)
app.secret_key = 'academic_secret_key'  # Change for production
DB_NAME = 'attendance_system.db'

def init_db():
    """Initialize the SQLite database with necessary tables."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Students Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            course TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Attendance Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            date_str TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    ''')
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

# ==========================================
# GLOBAL VARIABLES FOR RECOGNITION
# ==========================================
# We cache embeddings in memory to avoid DB reads on every frame
known_face_encodings = []
known_face_ids = []
known_face_names = []

def load_encodings():
    """Load all student embeddings from DB into memory."""
    global known_face_encodings, known_face_ids, known_face_names
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, name, embedding FROM students")
    rows = c.fetchall()
    
    known_face_encodings = []
    known_face_ids = []
    known_face_names = []
    
    for r in rows:
        s_id, s_name, s_emb_blob = r
        try:
            encoding = pickle.loads(s_emb_blob)
            known_face_encodings.append(encoding)
            known_face_ids.append(s_id)
            known_face_names.append(s_name)
        except Exception as e:
            print(f"Error loading student {s_id}: {e}")
            
    conn.close()
    print(f"Loaded {len(known_face_encodings)} student profiles.")

# Load initially
load_encodings()

# ==========================================
# CORE LOGIC
# ==========================================

def register_attendance(student_id):
    """Log attendance if not already logged recently."""
    now = datetime.datetime.now(IST)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")  # Store time in HH:MM:SS format
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")  # Full timestamp in IST
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Check if already present today (or within specific session timeframe)
    # Simple logic: One entry per student per day
    c.execute("SELECT * FROM attendance WHERE student_id = ? AND date_str = ?", (student_id, date_str))
    data = c.fetchone()
    
    if not data:
        c.execute("INSERT INTO attendance (student_id, date_str, timestamp) VALUES (?, ?, ?)", (student_id, date_str, timestamp_str))
        conn.commit()
        print(f"Attendance recorded for {student_id} at {time_str}")
    
    conn.close()

def generate_frames():
    """
    Video streaming generator function.
    Captures webcam, detects faces, draws boxes, and yields MJPEG frames.
    """
    # 0 is usually the default webcam. Change to 1 or video file path if needed.
    camera = cv2.VideoCapture(0) 
    
    if not camera.isOpened():
        print("Error: Could not access the camera.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
            
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names_to_display = []

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            s_id = None

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    s_id = known_face_ids[best_match_index]
                    
                    # Record Attendance in background
                    register_attendance(s_id)

            face_names_to_display.append(f"{name}")

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names_to_display):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==========================================
# HTML TEMPLATES (Embedded)
# ==========================================

CSS = """
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;900&family=Inter:wght@400;600;700&family=Playfair+Display:wght@700;900&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<style>
    * {
        margin: 0;
        padding: 0;
    }

    body {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 25%, #FEC868 50%, #30B0C0 75%, #4A90E2 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        min-height: 100vh;
        font-family: 'Poppins', 'Inter', sans-serif;
        color: #2c3e50;
        position: relative;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    body::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(255, 107, 107, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 80% 80%, rgba(42, 144, 226, 0.1) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
    }

    .navbar {
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E53 100%) !important;
        box-shadow: 0 8px 32px rgba(255, 107, 107, 0.3);
        border-bottom: 3px solid #FF1744;
        backdrop-filter: blur(10px);
        animation: slideDown 0.6s ease-out;
    }

    @keyframes slideDown {
        from {
            transform: translateY(-100%);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }

    .navbar-brand {
        font-weight: 900;
        font-size: 1.8rem;
        font-family: 'Playfair Display', serif;
        color: #fff !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }

    .navbar-brand:hover {
        transform: scale(1.05);
        text-shadow: 3px 3px 8px rgba(0,0,0,0.3);
    }

    .nav-link {
        font-weight: 700;
        font-size: 1.1rem;
        color: #fff !important;
        position: relative;
        transition: all 0.3s ease;
        margin: 0 10px;
    }

    .nav-link::before {
        content: '';
        position: absolute;
        bottom: -5px;
        left: 0;
        width: 0;
        height: 3px;
        background: linear-gradient(90deg, #FFD700, #FF1744);
        transition: width 0.3s ease;
    }

    .nav-link:hover {
        transform: translateY(-3px);
        color: #FFD700 !important;
    }

    .nav-link:hover::before {
        width: 100%;
    }

    .card {
        border: none;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        backdrop-filter: blur(20px);
        background: linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(240,248,255,0.95) 100%);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 2px solid rgba(255,255,255,0.5);
        position: relative;
        overflow: hidden;
    }

    .card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,215,0,0.3) 0%, transparent 70%);
        transition: all 0.6s ease;
        opacity: 0;
    }

    .card:hover {
        transform: translateY(-12px) scale(1.02);
        box-shadow: 0 30px 80px rgba(255,107,107,0.3);
    }

    .card:hover::before {
        opacity: 1;
    }

    .card-header {
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E53 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 18px 18px 0 0 !important;
        padding: 25px !important;
        box-shadow: 0 4px 15px rgba(255,107,107,0.2);
    }

    .card-title {
        font-weight: 800 !important;
        font-size: 1.6rem !important;
        font-family: 'Playfair Display', serif;
        letter-spacing: 0.5px;
    }

    .btn-primary {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF1744 100%);
        border: none;
        border-radius: 25px;
        padding: 14px 35px;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        color: white;
        box-shadow: 0 8px 25px rgba(255,23,68,0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .btn-primary::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255,255,255,0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }

    .btn-primary:hover {
        background: linear-gradient(135deg, #FF1744 0%, #C51162 100%);
        transform: translateY(-4px);
        box-shadow: 0 15px 40px rgba(255,23,68,0.4);
    }

    .btn-primary:hover::before {
        width: 300px;
        height: 300px;
    }

    .btn-primary:active {
        transform: translateY(-2px);
    }

    .stats-card {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 50%, #FEC868 100%);
        color: white;
        border-radius: 20px;
        padding: 35px 25px;
        margin-bottom: 20px;
        box-shadow: 0 15px 50px rgba(255,107,107,0.3);
        border: 2px solid rgba(255,255,255,0.3);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.6s ease-out;
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .stats-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.6s ease;
    }

    .stats-card:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 0 25px 70px rgba(255,107,107,0.4);
    }

    .stats-card:hover::before {
        left: 100%;
    }

    .stats-card h3 {
        font-size: 2.8rem;
        font-weight: 900;
        font-family: 'Playfair Display', serif;
        margin: 15px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    .stats-card p {
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .stats-card i {
        font-size: 2.5rem;
        animation: floatIcon 3s ease-in-out infinite;
    }

    @keyframes floatIcon {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    .table {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    .table thead th {
        background: linear-gradient(90deg, #FF6B6B 0%, #FF1744 100%);
        color: white;
        border: none;
        font-weight: 700;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 20px;
    }

    .table tbody tr {
        transition: all 0.3s ease;
        border-bottom: 2px solid #f0f0f0;
    }

    .table tbody tr:hover {
        background: linear-gradient(90deg, rgba(255,107,107,0.05) 0%, rgba(255,215,0,0.05) 100%);
        transform: scale(1.01);
    }

    .table td {
        padding: 18px;
        font-weight: 500;
        vertical-align: middle;
    }

    .badge {
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.95rem;
        animation: pulse 2s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    .badge.bg-primary {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF1744 100%) !important;
        box-shadow: 0 4px 15px rgba(255,107,107,0.3);
    }

    .form-control {
        border-radius: 15px;
        border: 2px solid #FF8E53;
        transition: all 0.3s ease;
        padding: 12px 20px;
        font-weight: 500;
        font-size: 1.05rem;
        background: rgba(255,255,255,0.95);
    }

    .form-control:focus {
        border-color: #FF1744;
        box-shadow: 0 0 0 0.3rem rgba(255,23,68,0.25);
        background: white;
        transform: scale(1.01);
    }

    .form-control::placeholder {
        color: #999;
        font-weight: 500;
    }

    .form-label {
        font-weight: 700;
        color: #FF6B6B;
        font-size: 1.1rem;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .video-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        border: 3px solid #FF6B6B;
        animation: videoGlow 3s ease-in-out infinite;
    }

    @keyframes videoGlow {
        0%, 100% { box-shadow: 0 20px 60px rgba(0,0,0,0.3), 0 0 30px rgba(255,107,107,0.3); }
        50% { box-shadow: 0 20px 60px rgba(0,0,0,0.3), 0 0 50px rgba(255,107,107,0.5); }
    }

    .video-container img {
        border-radius: 15px;
        animation: scaleIn 0.8s ease-out;
    }

    @keyframes scaleIn {
        from {
            transform: scale(0.95);
            opacity: 0;
        }
        to {
            transform: scale(1);
            opacity: 1;
        }
    }

    .alert {
        border-radius: 15px;
        border: 2px solid;
        border-left: 5px solid;
        font-weight: 600;
        font-size: 1.05rem;
        animation: slideInAlert 0.5s ease-out;
    }

    @keyframes slideInAlert {
        from {
            transform: translateX(-30px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    .alert-info {
        background: linear-gradient(135deg, rgba(74,144,226,0.1) 0%, rgba(48,176,192,0.1) 100%);
        border-color: #4A90E2;
        color: #1a5490;
    }

    .alert-success {
        background: linear-gradient(135deg, rgba(76,175,80,0.1) 0%, rgba(56,142,60,0.1) 100%);
        border-color: #4CAF50;
        color: #2e7d32;
    }

    .container {
        animation: fadeIn 0.8s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .form-text {
        font-weight: 500;
        color: #FF8E53;
    }

    .btn-close {
        filter: invert(1);
    }

    /* Text shadows for better readability */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 800;
        letter-spacing: 0.5px;
    }

    .d-grid {
        animation: slideUp 0.6s ease-out;
    }

    @keyframes slideUp {
        from {
            transform: translateY(20px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }

    /* Responsive animations */
    @media (max-width: 768px) {
        .stats-card {
            animation: fadeInUp 0.6s ease-out;
            margin-bottom: 15px;
        }

        .card {
            margin-bottom: 20px;
        }

        .navbar-brand {
            font-size: 1.4rem;
        }

        .nav-link {
            font-size: 1rem;
            margin: 5px 0;
        }
    }
</style>
"""

LAYOUT = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Attendance System</title>
    {CSS}
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            // Add animation to all cards on page load
            const cards = document.querySelectorAll('.card');
            cards.forEach((card, index) => {{
                card.style.animation = `fadeInUp 0.6s ease-out ${{index * 0.1}}s both`;
            }});

            // Add ripple effect to buttons
            document.querySelectorAll('.btn').forEach(button => {{
                button.addEventListener('click', function(e) {{
                    const ripple = document.createElement('span');
                    const rect = this.getBoundingClientRect();
                    const size = Math.max(rect.width, rect.height);
                    const x = e.clientX - rect.left - size / 2;
                    const y = e.clientY - rect.top - size / 2;
                    
                    ripple.style.width = ripple.style.height = size + 'px';
                    ripple.style.left = x + 'px';
                    ripple.style.top = y + 'px';
                    ripple.classList.add('ripple');
                    this.appendChild(ripple);
                    
                    setTimeout(() => ripple.remove(), 600);
                }});
            }});
        }});
    </script>
    <style>
        .ripple {{
            position: absolute;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.6);
            transform: scale(0);
            animation: ripple-animation 0.6s ease-out;
            pointer-events: none;
        }}

        @keyframes ripple-animation {{
            to {{
                transform: scale(4);
                opacity: 0;
            }}
        }}
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="fas fa-user-check me-2"></i>🎓 Academic Attendance
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" style="border-color: rgba(255,255,255,0.5);">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="/"><i class="fas fa-tachometer-alt me-1"></i>Dashboard</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/register"><i class="fas fa-user-plus me-1"></i>Register</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/live"><i class="fas fa-video me-1"></i>Live Monitor</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>
    <div class="container mt-5">
        {{% with messages = get_flashed_messages() %}}
            {{% if messages %}}
                {{% for message in messages %}}
                    <div class="alert alert-success alert-dismissible fade show" role="alert" style="margin-bottom: 30px;">
                        <i class="fas fa-check-circle me-2"></i>
                        <strong>Success!</strong> {{{{ message }}}}
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                    </div>
                {{% endfor %}}
            {{% endif %}}
        {{% endwith %}}
        {{% block content %}}{{ % endblock %}}
    </div>
</body>
</html>
"""

TEMPLATE_DASHBOARD = LAYOUT.replace("{% block content %}{ % endblock %}", """
    <h1 style="font-size: 3rem; font-family: 'Playfair Display', serif; color: white; text-shadow: 3px 3px 6px rgba(0,0,0,0.3); margin-bottom: 40px; letter-spacing: 1px;">Dashboard Overview</h1>
    <div class="row mb-4">
        <div class="col-md-4">
            <div class="stats-card text-center">
                <i class="fas fa-users mb-3"></i>
                <h3>{{ total_students }}</h3>
                <p class="mb-0">Total Students</p>
            </div>
        </div>
        <div class="col-md-4">
            <div class="stats-card text-center">
                <i class="fas fa-calendar-check mb-3"></i>
                <h3>{{ today_attendance }}</h3>
                <p class="mb-0">Present Today</p>
            </div>
        </div>
        <div class="col-md-4">
            <div class="stats-card text-center">
                <i class="fas fa-chart-line mb-3"></i>
                <h3>{{ attendance_rate }}%</h3>
                <p class="mb-0">Attendance Rate</p>
            </div>
        </div>
    </div>
    <div class="card">
        <div class="card-header">
            <h2 class="card-title mb-0"><i class="fas fa-list me-3"></i>Attendance Report</h2>
        </div>
        <div class="card-body" style="padding: 30px;">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <p style="font-weight: 600; font-size: 1.1rem; color: #666; mb-0;">📊 Recent attendance records</p>
                <a href="/export" class="btn btn-primary">
                    <i class="fas fa-download me-2"></i>Download CSV Report
                </a>
            </div>
            <div class="table-responsive">
                <table class="table table-hover">
                    <thead>
                        <tr>
                            <th><i class="fas fa-calendar me-2"></i>Date</th>
                            <th><i class="fas fa-clock me-2"></i>Time</th>
                            <th><i class="fas fa-id-card me-2"></i>Student ID</th>
                            <th><i class="fas fa-user me-2"></i>Name</th>
                            <th><i class="fas fa-graduation-cap me-2"></i>Course</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for row in attendance_log %}
                        <tr style="animation: slideInLeft 0.5s ease-out;">
                            <td><span style="font-weight: 600;">{{ row[2] }}</span></td>
                            <td><span style="font-weight: 600;">{{ row[1].split(' ')[1] if ' ' in row[1] else row[1] }}</span></td>
                            <td><span class="badge bg-primary">{{ row[3] }}</span></td>
                            <td><strong>{{ row[4] }}</strong></td>
                            <td><span style="color: #FF6B6B; font-weight: 700;">{{ row[5] }}</span></td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    <style>
        @keyframes slideInLeft {
            from {
                transform: translateX(-20px);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
    </style>
""")

TEMPLATE_REGISTER = LAYOUT.replace("{% block content %}{ % endblock %}", """
    <div class="row justify-content-center">
        <div class="col-md-8">
            <div style="text-align: center; margin-bottom: 40px; animation: fadeInDown 0.8s ease-out;">
                <h1 style="font-size: 3rem; font-family: 'Playfair Display', serif; color: white; text-shadow: 3px 3px 6px rgba(0,0,0,0.3); letter-spacing: 1px;">Register New Student</h1>
                <p style="font-size: 1.2rem; color: white; font-weight: 600; margin-top: 10px;">Add a student to the attendance system</p>
            </div>
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title mb-0"><i class="fas fa-user-plus me-3"></i>Student Registration Form</h2>
                </div>
                <div class="card-body" style="padding: 40px;">
                    <p style="color: #666; font-size: 1.1rem; margin-bottom: 30px; font-weight: 500;">📸 Upload a clear, front-facing photo to generate the facial embedding for attendance recognition.</p>
                    <form action="/register" method="POST" enctype="multipart/form-data">
                        <div class="row">
                            <div class="col-md-6 mb-4">
                                <label for="student_id" class="form-label">
                                    <i class="fas fa-id-card me-2"></i>Student ID
                                </label>
                                <input type="text" class="form-control" id="student_id" name="student_id" required placeholder="e.g. S1024" style="font-size: 1.05rem;">
                            </div>
                            <div class="col-md-6 mb-4">
                                <label for="course" class="form-label">
                                    <i class="fas fa-graduation-cap me-2"></i>Course Code
                                </label>
                                <input type="text" class="form-control" id="course" name="course" required placeholder="e.g. CS101" style="font-size: 1.05rem;">
                            </div>
                        </div>
                        <div class="mb-4">
                            <label for="name" class="form-label">
                                <i class="fas fa-user me-2"></i>Full Name
                            </label>
                            <input type="text" class="form-control" id="name" name="name" required placeholder="John Doe" style="font-size: 1.05rem;">
                        </div>
                        <div class="mb-4">
                            <label for="file" class="form-label">
                                <i class="fas fa-camera me-2"></i>Profile Photo
                            </label>
                            <input type="file" class="form-control" id="file" name="file" accept="image/*" required style="padding: 15px; font-size: 1.05rem;">
                            <div class="form-text">✨ Please upload a clear, front-facing photo for best recognition results.</div>
                        </div>
                        <div class="d-grid">
                            <button type="submit" class="btn btn-primary btn-lg" style="padding: 16px; font-size: 1.2rem;">
                                <i class="fas fa-save me-2"></i>Register Student
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </div>
    <style>
        @keyframes fadeInDown {
            from {
                transform: translateY(-30px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
    </style>
""")

TEMPLATE_LIVE = LAYOUT.replace("{% block content %}{ % endblock %}", """
    <style>
        .live-monitor-container {
            display: flex;
            flex-direction: column;
            height: calc(100vh - 200px);
            gap: 20px;
        }

        .video-wrapper {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            min-height: 400px;
        }

        .video-wrapper img {
            max-width: 100%;
            max-height: 100%;
            height: auto;
            width: auto;
            object-fit: contain;
        }

        .legend-section {
            flex-shrink: 0;
        }

        @media (max-width: 768px) {
            .live-monitor-container {
                height: auto;
            }
            
            .video-wrapper {
                min-height: 300px;
            }

            h1 {
                font-size: 2rem !important;
            }

            .card-title {
                font-size: 1.3rem !important;
            }
        }

        @media (max-width: 480px) {
            .live-monitor-container {
                gap: 15px;
            }

            .video-wrapper {
                min-height: 250px;
            }

            h1 {
                font-size: 1.5rem !important;
            }

            .card-title {
                font-size: 1.1rem !important;
            }

            .alert {
                font-size: 0.95rem !important;
            }
        }
    </style>

    <h1 style="font-size: 3rem; font-family: 'Playfair Display', serif; color: white; text-shadow: 3px 3px 6px rgba(0,0,0,0.3); margin-bottom: 30px; letter-spacing: 1px;">Live Recognition Monitor</h1>
    
    <div class="card" style="height: 100%; display: flex; flex-direction: column;">
        <div class="card-header">
            <h2 class="card-title mb-0"><i class="fas fa-video me-3"></i>Real-time Face Recognition</h2>
        </div>
        <div class="card-body" style="padding: 20px; flex: 1; display: flex; flex-direction: column;">
            <div class="alert alert-info" style="font-size: 1rem; margin-bottom: 20px;">
                <i class="fas fa-info-circle me-2"></i>
                <strong>Ensure the camera is connected and active.</strong> Recognized students will be automatically logged for attendance.
            </div>
            
            <div class="video-wrapper" style="flex: 1;">
                <div class="video-container" style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">
                    <img class="img-fluid rounded" src="{{ url_for('video_feed') }}" alt="Live Feed" style="max-width: 100%; max-height: 100%; object-fit: contain;">
                </div>
            </div>

            <div class="legend-section mt-4" style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; border: 2px dashed rgba(255,107,107,0.3);">
                <p style="font-size: 1.1rem; font-weight: 600; margin-bottom: 15px; color: white; text-align: center;">📊 Legend:</p>
                <div class="row">
                    <div class="col-md-6 col-sm-6 text-center mb-2">
                        <span style="font-size: 1rem; font-weight: 700;">
                            <i class="fas fa-square" style="color: #00ff00; margin-right: 8px;"></i>
                            Green = Recognized
                        </span>
                    </div>
                    <div class="col-md-6 col-sm-6 text-center mb-2">
                        <span style="font-size: 1rem; font-weight: 700;">
                            <i class="fas fa-square" style="color: #ff0000; margin-right: 8px;"></i>
                            Red = Unknown
                        </span>
                    </div>
                </div>
            </div>
        </div>
    </div>
""")

# ==========================================
# ROUTES
# ==========================================

@app.route('/')
def dashboard():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Get attendance log
    query = '''
        SELECT a.id, a.timestamp, a.date_str, s.id, s.name, s.course 
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        ORDER BY a.timestamp DESC
    '''
    c.execute(query)
    attendance_log = c.fetchall()
    
    # Get total students
    c.execute("SELECT COUNT(*) FROM students")
    total_students = c.fetchone()[0]
    
    # Get today's attendance
    today = datetime.datetime.now(IST).strftime("%Y-%m-%d")
    c.execute("SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date_str = ?", (today,))
    today_attendance = c.fetchone()[0]
    
    # Calculate attendance rate (today's attendance / total students * 100)
    attendance_rate = 0
    if total_students > 0:
        attendance_rate = round((today_attendance / total_students) * 100, 1)
    
    conn.close()
    return render_template_string(TEMPLATE_DASHBOARD, 
                                attendance_log=attendance_log,
                                total_students=total_students,
                                today_attendance=today_attendance,
                                attendance_rate=attendance_rate)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        s_id = request.form['student_id']
        name = request.form['name']
        course = request.form['course']
        file = request.files['file']
        
        if file:
            # Read image file into numpy array
            in_memory_file = io.BytesIO(file.read())
            file_bytes = np.asarray(bytearray(in_memory_file.read()), dtype=np.uint8)
            # Decode image using opencv but keep it for face_recognition
            # We need to reload the bytes pointer
            in_memory_file.seek(0)
            
            # Using face_recognition to load directly
            image = face_recognition.load_image_file(in_memory_file)
            
            # Generate encoding
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) > 0:
                embedding = encodings[0]
                serialized_embedding = pickle.dumps(embedding)
                
                try:
                    conn = sqlite3.connect(DB_NAME)
                    c = conn.cursor()
                    c.execute("INSERT INTO students (id, name, course, embedding) VALUES (?, ?, ?, ?)",
                              (s_id, name, course, serialized_embedding))
                    conn.commit()
                    conn.close()
                    
                    # Reload global encodings
                    load_encodings()
                    flash(f"Student {name} registered successfully!")
                    return redirect(url_for('dashboard'))
                except sqlite3.IntegrityError:
                    flash("Error: Student ID already exists.")
            else:
                flash("Error: No face detected in the photo. Please try again.")
        
    return render_template_string(TEMPLATE_REGISTER)

@app.route('/live')
def live():
    return render_template_string(TEMPLATE_LIVE)

@app.route('/video_feed')
def video_feed():
    """Route that returns the streaming response."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/export')
def export_csv():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        SELECT a.date_str, a.timestamp, s.id, s.name, s.course 
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        ORDER BY a.timestamp DESC
    ''')
    rows = c.fetchall()
    conn.close()
    
    # Generate CSV
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Date', 'Timestamp', 'Student ID', 'Name', 'Course'])
    cw.writerows(rows)
    output = io.BytesIO()
    output.write(si.getvalue().encode('utf-8'))
    output.seek(0)
    
    return send_file(output, mimetype="text/csv", as_attachment=True, download_name="attendance_report.csv")

if __name__ == '__main__':
    # Threaded is required for video streaming to work alongside other requests
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)