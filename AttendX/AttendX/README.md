
# 🤖 AttendX — Integrated Face Attendance System
> An automated, AI-powered attendance tracking application built with a Flask backend, responsive HTML5 frontend, and OpenCV face recognition algorithms.

---

## 🌐 Live Project & Links
* 🚀 **Live Deployment:** [View Live Web App](https://attendx-kimy.onrender.com) *(Note: Device webcam required for full functionality)*
* 📂 **Repository Link:** [GitHub Repo](https://github.com/arpita7470/Attendx)

---

## 🛠️ Tech Stack & Architecture

| Backend & AI | Frontend UI | Tools & Environment |
| :--- | :--- | :--- |
| **Python** (Core Logic) | **HTML5** & **CSS3** | **Git** & **GitHub** |
| **Flask** (Web Framework) | Custom Responsive UI | **Gunicorn** (Production Server) |
| **OpenCV** (Computer Vision) | Dynamic CSS Keyframes | **Render** (Cloud Deployment) |
| **LBPH Face Recognizer** | JavaScript (Fetch API) | **CSV** (Lightweight Storage) |

---

## 📁 Project Structure

```text
AppendX_Integrated/
├── app.py                         # Core Flask backend and API routing
├── requirements.txt               # Application dependencies & packages
├── haarcascade_frontalface_default.xml # OpenCV Haar Cascade file for face detection
├── templates/
│   └── index.html                 # Main frontend user interface
├── TrainingImage/                 # Dataset directory (stores captured student faces)
├── TrainingImageLabel/            # Model directory (stores trained Trainner.yml)
├── StudentDetails/                # Database directory (stores studentdetails.csv)
└── Attendance/                    # Analytics directory (stores subject-wise CSV logs)

---

## 🎯 Key Features

> ### 📋 System Capabilities at a Glance
>
> | Feature | Operational Description | Core Tech/Logic |
> | :--- | :--- | :--- |
> | **1. Seamless Registration** | Input details to trigger the webcam and automatically capture **50 distinct facial angles** for the dataset. | OpenCV Haar Cascade |
> | **2. LBPH Model Training** | Train the face recognition engine directly from the UI dashboard with a single click. | Local Binary Patterns Histograms |
> | **3. Real-Time Scanning** | Stream live video to detect and match faces against the database, displaying recognized names instantly. | OpenCV Video Capture |
> | **4. Auto-Attendance Logs** | Terminate the stream to automatically export timestamped logs into subject-wise files. | Python CSV Integration |
> | **5. Record Analytics** | Query complete attendance history by subject to generate dynamic tables with calculated **attendance percentages**. | Dynamic Data Mapping |
> | **6. Interactive UI Scanner** | Visual UI circle that **expands and pulses with a glowing green aura** immediately upon successful face detection. | CSS Keyframe Animations |

---




