import cv2
import os
import csv
import numpy as np
import pandas as pd
import datetime
import time


def TakeImage(enrollment, name, haarcasecade_path, trainimage_path, message, err_screen, text_to_speech):
    """Capture face images from webcam and save them for training."""

    if enrollment == "" or name == "":
        err_screen()
        return

    # Load Haar Cascade face detector
    detector = cv2.CascadeClassifier(haarcasecade_path)
    
    if detector.empty():
        message.configure(text="Error: Haar cascade file not found!")
        text_to_speech("Error loading face detector.")
        return

    # Create directory for this student
    student_img_path = os.path.join(trainimage_path, f"{enrollment}_{name}")
    if not os.path.exists(student_img_path):
        os.makedirs(student_img_path)

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)

    sample_num = 0
    max_images = 70  # Increased for better training accuracy

    message.configure(text=f"Capturing images for {name}... Please look at camera.")

    while True:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Better detection parameters - more sensitive
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            sample_num += 1
            
            # FIX: Save CROPPED face (grayscale) resized to standard size
            face_crop = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_crop, (200, 200))
            
            img_path = os.path.join(student_img_path, f"{enrollment}_{name}_{sample_num}.jpg")
            cv2.imwrite(img_path, face_resized)

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"Captured: {sample_num}/{max_images}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(img, f"ID: {enrollment} | {name}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if len(faces) == 0:
            cv2.putText(img, "No face detected - adjust lighting/position", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Taking Images - Press ESC to stop", img)

        if cv2.waitKey(1) == 27 or sample_num >= max_images:
            break

    cam.release()
    cv2.destroyAllWindows()

    if sample_num == 0:
        message.configure(text="No face detected! Check lighting and camera position.")
        text_to_speech("No face detected. Please try again with better lighting.")
        return

    # FIX: Prevent duplicate enrollment entries in CSV
    studentdetail_path = "./StudentDetails/studentdetails.csv"
    os.makedirs("./StudentDetails", exist_ok=True)

    already_exists = False
    if os.path.isfile(studentdetail_path):
        existing_df = pd.read_csv(studentdetail_path)
        if int(enrollment) in existing_df["Enrollment"].values:
            already_exists = True

    if not already_exists:
        file_exists = os.path.isfile(studentdetail_path)
        with open(studentdetail_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["Enrollment", "Name"])
            writer.writerow([enrollment, name])

    message.configure(text=f"{sample_num} Images Saved for {name}! Now click Train Image.")
    text_to_speech(f"{sample_num} images saved for {name}. Please click Train Image now.")
