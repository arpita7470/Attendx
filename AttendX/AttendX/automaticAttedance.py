import cv2
import os
import csv
import numpy as np
import pandas as pd
import datetime
import time
import tkinter as tk
from tkinter import *


def subjectChoose(text_to_speech):
    """Open subject selection window and start face recognition attendance."""

    def fillAttendance():
        subject = subject_var.get().strip()
        if not subject:
            tk.messagebox.showwarning("Warning", "Please enter subject name!")
            return

        subject_win.destroy()
        takeAttendance(subject, text_to_speech)

    def checkSheets():
        subject = subject_var.get().strip()
        if not subject:
            tk.messagebox.showwarning("Warning", "Please enter subject name!")
            return
        attendance_file = f"./Attendance/{subject}.csv"
        if os.path.exists(attendance_file):
            df = pd.read_csv(attendance_file)
            showSheet(df, subject)
        else:
            tk.messagebox.showinfo("Info", f"No attendance sheet found for {subject}")

    subject_win = tk.Tk()
    subject_win.title("Subject...")
    subject_win.geometry("600x300")
    subject_win.configure(background="#1c1c1c")
    subject_win.resizable(0, 0)

    tk.Label(
        subject_win, text="Enter the Subject Name",
        bg="#333333", fg="green", font=("Verdana", 20, "bold"),
        width=30, height=2
    ).pack(fill=X)

    tk.Label(
        subject_win, text="Enter Subject", bg="#1c1c1c", fg="yellow",
        font=("Verdana", 14), width=12, height=2
    ).place(x=60, y=80)

    subject_var = tk.StringVar()
    subject_entry = tk.Entry(
        subject_win, width=20, bd=5, bg="#333333", fg="yellow",
        relief=RIDGE, font=("Verdana", 16, "bold"), textvariable=subject_var
    )
    subject_entry.place(x=250, y=85)

    fill_btn = tk.Button(
        subject_win, text="Fill Attendance", command=fillAttendance,
        bd=5, font=("Verdana", 14, "bold"), bg="#333333", fg="yellow",
        height=1, width=12, relief=RIDGE
    )
    fill_btn.place(x=160, y=160)

    check_btn = tk.Button(
        subject_win, text="Check Sheets", command=checkSheets,
        bd=5, font=("Verdana", 14, "bold"), bg="#333333", fg="yellow",
        height=1, width=12, relief=RIDGE
    )
    check_btn.place(x=340, y=160)

    subject_win.mainloop()


def takeAttendance(subject, text_to_speech):
    """Use trained model to recognize faces and mark attendance."""

    trainimagelabel_path = "./TrainingImageLabel/Trainner.yml"
    haarcasecade_path = "haarcascade_frontalface_default.xml"
    studentdetail_path = "./StudentDetails/studentdetails.csv"

    if not os.path.exists(trainimagelabel_path):
        text_to_speech("Model not found. Please train images first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainimagelabel_path)

    detector = cv2.CascadeClassifier(haarcasecade_path)
    
    if detector.empty():
        text_to_speech("Face detector file not found.")
        return

    # Load student details - FIX: handle duplicate enrollments
    student_map = {}
    if os.path.exists(studentdetail_path):
        df = pd.read_csv(studentdetail_path)
        # Drop duplicates, keep last entry for each enrollment
        df = df.drop_duplicates(subset=["Enrollment"], keep="last")
        for _, row in df.iterrows():
            student_map[int(row["Enrollment"])] = str(row["Name"])

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)

    attendance_set = set()
    today = str(datetime.date.today())

    text_to_speech(f"Starting attendance for {subject}. Press ESC to stop.")

    while True:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # FIX: Better detection parameters - same as takeImage
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            # FIX: Resize face to match training size (200x200)
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))

            try:
                id_, confidence = recognizer.predict(face_roi)
                
                # FIX: Confidence threshold raised to 85 (more lenient)
                # In LBPH: lower confidence = better match
                if confidence < 85:
                    name = student_map.get(id_, f"ID:{id_}")
                    label = f"{name} ({100 - int(confidence)}%)"
                    color = (0, 255, 0)
                    attendance_set.add((id_, name))
                else:
                    label = f"Unknown ({100 - int(confidence)}%)"
                    color = (0, 0, 255)
            except Exception as e:
                label = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Show count of recognized students
        cv2.putText(img, f"Recognized: {len(attendance_set)} | Press ESC to save",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow(f"Attendance: {subject}", img)

        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

    saveAttendance(subject, attendance_set, today)
    text_to_speech(f"Attendance saved for {len(attendance_set)} student(s) in {subject}.")


def saveAttendance(subject, attendance_set, today):
    """Save attendance records to CSV file for the subject."""

    os.makedirs("./Attendance", exist_ok=True)
    attendance_file = f"./Attendance/{subject}.csv"

    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file, index_col=0)
    else:
        df = pd.DataFrame(columns=["Enrollment", "Name"])
        df.set_index("Enrollment", inplace=True)

    if today not in df.columns:
        df[today] = 0

    for (enrollment, name) in attendance_set:
        if enrollment not in df.index:
            df.loc[enrollment] = 0
            df.loc[enrollment, "Name"] = name
        df.loc[enrollment, today] = 1

    date_cols = [c for c in df.columns if c != "Name" and c != "Attendance%"]
    if date_cols:
        df["Attendance%"] = (df[date_cols].sum(axis=1) / len(date_cols) * 100).round(1).astype(str) + "%"

    df.to_csv(attendance_file)


def showSheet(df, subject):
    """Show attendance sheet in a new window."""
    sheet_win = tk.Tk()
    sheet_win.title(f"Attendance - {subject}")
    sheet_win.configure(background="#1c1c1c")

    tk.Label(
        sheet_win, text=f"Attendance Sheet: {subject}",
        bg="#1c1c1c", fg="yellow", font=("Verdana", 16, "bold")
    ).pack(pady=10)

    frame = tk.Frame(sheet_win, bg="#1c1c1c")
    frame.pack(padx=10, pady=10)

    for j, col in enumerate(df.columns):
        tk.Label(
            frame, text=col, bg="#333333", fg="yellow",
            font=("Verdana", 10, "bold"), bd=2, relief=RIDGE, width=12
        ).grid(row=0, column=j, sticky="nsew")

    for i, row in df.iterrows():
        tk.Label(
            frame, text=str(i), bg="#1c1c1c", fg="white",
            font=("Verdana", 10), bd=1, relief=RIDGE, width=12
        ).grid(row=i+1, column=0, sticky="nsew")
        for j, val in enumerate(row):
            tk.Label(
                frame, text=str(val), bg="#1c1c1c", fg="white",
                font=("Verdana", 10), bd=1, relief=RIDGE, width=12
            ).grid(row=i+1, column=j+1, sticky="nsew")

    sheet_win.mainloop()
