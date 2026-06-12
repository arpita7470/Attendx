import cv2
import os
import numpy as np
from PIL import Image


def TrainImage(haarcasecade_path, trainimage_path, trainimagelabel_path, message, text_to_speech):
    """Train LBPH face recognizer on saved student images."""

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(haarcasecade_path)

    message.configure(text="Training in progress... Please wait.")
    
    faces, ids = getImagesAndLabels(trainimage_path, detector)

    if len(faces) == 0:
        message.configure(text="No images found! Please take images first.")
        text_to_speech("No images found. Please register students first.")
        return

    recognizer.train(faces, np.array(ids))

    # Save trained model
    os.makedirs(os.path.dirname(trainimagelabel_path), exist_ok=True)
    recognizer.write(trainimagelabel_path)

    unique_students = len(set(ids))
    total_images = len(faces)
    message.configure(text=f"Training done! {unique_students} student(s), {total_images} images used.")
    text_to_speech(f"Training complete for {unique_students} students. You can now take attendance.")


def getImagesAndLabels(trainimage_path, detector):
    """Read all training images and return face arrays with their IDs."""
    faces = []
    ids = []

    if not os.path.exists(trainimage_path):
        return faces, ids

    for student_folder in os.listdir(trainimage_path):
        folder_path = os.path.join(trainimage_path, student_folder)

        if not os.path.isdir(folder_path):
            continue

        # Extract enrollment ID from folder name (format: enrollment_name)
        parts = student_folder.split("_")
        try:
            student_id = int(parts[0])
        except ValueError:
            continue

        for img_file in os.listdir(folder_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(folder_path, img_file)
            try:
                pil_img = Image.open(img_path).convert('L')  # Grayscale
                img_arr = np.array(pil_img, 'uint8')
                
                # FIX: Resize to standard size (must match what takeImage saves)
                img_arr = cv2.resize(img_arr, (200, 200))
                
                faces.append(img_arr)
                ids.append(student_id)
            except Exception as e:
                print(f"Error reading {img_path}: {e}")

    return faces, ids
