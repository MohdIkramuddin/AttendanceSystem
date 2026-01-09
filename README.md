# AttendanceSystem
Project Introduction: Face Recognition Attendance System

Overview

The Face Recognition Attendance System is a modern, biometrics-based application designed to automate student attendance tracking. Moving away from traditional manual roll calls—which are time-consuming and prone to errors—this system leverages computer vision technology to identify students instantly via a live camera feed and record their attendance in real-time.

How It Works

The system operates in three simple stages:

Registration: An administrator uploads a student's photo and details (Name, ID, Course). The system analyzes the face and creates a unique mathematical signature (facial embedding).

Recognition: A camera continuously monitors the classroom entrance. When a face is detected, the system compares it against the stored signatures.

Logging: If a match is found with high confidence, the student is marked as "Present" in the database along with the exact date and timestamp.

Key Features

Contactless & Fast: Records attendance in milliseconds without physical interaction.

Live Monitoring: Provides a real-time video feed showing recognized names and status.

Secure Storage: Stores abstract facial embeddings rather than raw images for efficiency.

Reporting: Generates downloadable CSV reports for teachers to review daily or monthly attendance.

Technology Stack

The system is built using a robust Python-based stack:

Core Logic: Python 3.11

Computer Vision: OpenCV (Video capture & processing) & face_recognition (dlib-based AI model).

Web Interface: Flask (Backend server) with HTML5/CSS3 (Frontend).

Database: SQLite (Lightweight, serverless data management).

Objective

This project aims to demonstrate the practical application of Artificial Intelligence in educational administration, providing a low-cost, high-efficiency solution for attendance management.
