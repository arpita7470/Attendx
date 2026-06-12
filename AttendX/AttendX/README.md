



http:# AppendX — Integrated Face Attendance System


## 📁 Project Structure
```
AppendX_Integrated/
├── app.py                          ← Flask backend (main server)
├── requirements.txt
├── haarcascade_frontalface_default.xml
├── templates/
│   └── index.html                  ← Frontend (browser me khulta hai)
├── TrainingImage/                  ← Student face images yahan save hongi
├── TrainingImageLabel/             ← Trained model (Trainner.yml)
├── StudentDetails/                 ← studentdetails.csv
└── Attendance/                     ← Subject-wise attendance CSVs
```

---

## 🎯 Features

### 1. REGISTER Button
- Enrollment number aur naam daalo
- Camera shuru hoga, 50 face images capture hongi automatically
- Phir "Train Model" click karo — LBPH model train ho jayega

### 2. TAKE ATTENDANCE Button
- Subject name daalo
- Camera shuru hoga, faces recognize honge
- Recognized student ka naam screen pe dikhega
- "Band Karo" press karo → attendance CSV me save ho jayegi

### 3. VIEW RECORDS Button
- Subject name daalkar attendance records dekho
- Table me date-wise columns aur attendance percentage dikhegi

### 4. Face Detection Circle (Scanner)
- Navbar ke neeche wala circle icon **bada hoga aur green glow ke saath blink karega**
  jab bhi koi face camera me detect hoga
- Recognized student ka naam bhi circle ke neeche dikhega
## 🌐 Live Demo
Aap mera live project yahan dekh sakte hain: [AttendX Live Application](https://attendx-kimy.onrender.com)
---

## ⚠️ Notes
- Pehle **Register** karo, phir **Train** karo, phir **Attendance** lo
- Webcam connected honi chahiye
- Ek baar train karne ke baad naye students register karne par dobara train karna hoga
