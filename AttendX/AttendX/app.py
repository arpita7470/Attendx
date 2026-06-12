"""
AttendX - Flask Backend (ULTRA SPEED OPTIMIZED)
"""
from flask import Flask, render_template, request, jsonify, Response
import cv2, os, csv, numpy as np, pandas as pd, datetime, threading, time
from PIL import Image

app = Flask(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
HAAR_PATH           = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
TRAIN_IMG_PATH      = os.path.join(BASE_DIR, "TrainingImage")
TRAIN_LABEL_PATH    = os.path.join(BASE_DIR, "TrainingImageLabel", "Trainner.yml")
STUDENT_DETAIL_PATH = os.path.join(BASE_DIR, "StudentDetails", "studentdetails.csv")
ATTENDANCE_PATH     = os.path.join(BASE_DIR, "Attendance")

for d in [TRAIN_IMG_PATH, os.path.dirname(TRAIN_LABEL_PATH),
          os.path.dirname(STUDENT_DETAIL_PATH), ATTENDANCE_PATH]:
    os.makedirs(d, exist_ok=True)

# ── ULTRA SPEED CONSTANTS ──────────────────────────────────────────────────────
CAM_W        = 640
CAM_H        = 480
JPEG_Q       = 60
DETECT_SKIP  = 1         # har frame pe detect karo
REGISTER_MAX = 70
SCALE_FACTOR = 1.0       # FIX: shrink mat karo - full size pe detect karo

# ── Shared state ───────────────────────────────────────────────────────────────
camera_active    = False
current_frame    = None
frame_lock       = threading.Lock()
rec_status       = {"detected": False, "name": "", "confidence": 0}
camera_mode      = None
register_data    = {}
attendance_data  = {"marked": set(), "student_map": {}, "subject": "", "today": ""}
train_status     = {"running": False, "done": False, "msg": "", "ok": False}
live_attendance  = []

# ── Pre-load detector once at startup ─────────────────────────────────────────
_detector = cv2.CascadeClassifier(HAAR_PATH)


# ══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/register/start", methods=["POST"])
def register_start():
    global camera_active, camera_mode, register_data
    d = request.json
    enr, name = d.get("enrollment","").strip(), d.get("name","").strip()
    if not enr or not name:
        return jsonify({"success": False, "message": "Fill Enrollment and Name!"})
    if not enr.isdigit():
        return jsonify({"success": False, "message": "Enrollment in numbers only!"})
    if camera_active:
        return jsonify({"success": False, "message": "Camera is already working!"})
    register_data = {"enrollment": enr, "name": name, "count": 0, "max": REGISTER_MAX}
    camera_mode   = "register"
    camera_active = True
    threading.Thread(target=camera_loop, daemon=True).start()
    return jsonify({"success": True, "message": f"{name} registration started!"})


@app.route("/api/register/stop", methods=["POST"])
def register_stop():
    global camera_active
    camera_active = False
    time.sleep(0.2)
    enr   = register_data.get("enrollment","")
    name  = register_data.get("name","")
    count = register_data.get("count", 0)
    if count > 0:
        exists = os.path.isfile(STUDENT_DETAIL_PATH)
        with open(STUDENT_DETAIL_PATH, "a", newline="") as f:
            w = csv.writer(f)
            if not exists: w.writerow(["Enrollment","Name"])
            w.writerow([enr, name])
        return jsonify({"success": True,
                        "message": f"{name}'s {count} images saved successfully.",
                        "count": count})
    return jsonify({"success": False, "message": "image not captured yet."})


@app.route("/api/train", methods=["POST"])
def train_model():
    global train_status
    if train_status["running"]:
        return jsonify({"success": False, "message": "Training is going on..."})
    train_status = {"running": True, "done": False, "msg": "Training started...", "ok": False}
    threading.Thread(target=_train_worker, daemon=True).start()
    return jsonify({"success": True, "message": "Training started in background !"})


def _train_worker():
    global train_status
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8
        )
        faces, ids = [], []
        for folder in os.listdir(TRAIN_IMG_PATH):
            fp = os.path.join(TRAIN_IMG_PATH, folder)
            if not os.path.isdir(fp): continue
            try: sid = int(folder.split("_")[0])
            except: continue
            for img_file in sorted(os.listdir(fp)):
                if not img_file.lower().endswith((".jpg",".jpeg",".png")): continue
                try:
                    img_path = os.path.join(fp, img_file)
                    # cv2 se load karo (PIL nahi) — exactly same as attendance predict
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None: continue
                    img = cv2.resize(img, (100, 100))
                    faces.append(img)
                    ids.append(sid)
                except Exception as ex:
                    print(f"Skip {img_file}: {ex}")
        if not faces:
            train_status = {"running":False,"done":True,"ok":False,
                            "msg":"image not found! Register first."}
            return
        recognizer.train(faces, np.array(ids))
        recognizer.write(TRAIN_LABEL_PATH)
        train_status = {"running":False,"done":True,"ok":True,
                        "msg":f"Training complete! {len(set(ids))} student(s), {len(faces)} images."}
    except Exception as e:
        train_status = {"running":False,"done":True,"ok":False,"msg":f"Error: {e}"}


@app.route("/api/train/status")
def train_status_api():
    return jsonify(train_status)


@app.route("/api/attendance/start", methods=["POST"])
def attendance_start():
    global camera_active, camera_mode, attendance_data, live_attendance
    d = request.json
    subject = d.get("subject","").strip()
    if not subject:
        return jsonify({"success": False, "message": "Subject Name!"})
    if not os.path.exists(TRAIN_LABEL_PATH):
        return jsonify({"success": False, "message": "Data not found! Save data first."})
    if camera_active:
        return jsonify({"success": False, "message": "Camera is already in working!"})
    smap = {}
    if os.path.exists(STUDENT_DETAIL_PATH):
        df = pd.read_csv(STUDENT_DETAIL_PATH)
        df = df.drop_duplicates(subset=["Enrollment"], keep="last")
        for _, row in df.iterrows():
            smap[int(row["Enrollment"])] = str(row["Name"])
    live_attendance = []
    attendance_data = {"subject": subject, "student_map": smap,
                       "marked": set(), "today": str(datetime.date.today())}
    camera_mode   = "attendance"
    camera_active = True
    threading.Thread(target=camera_loop, daemon=True).start()
    return jsonify({"success": True, "message": f"{subject} attendance started!"})


@app.route("/api/attendance/stop", methods=["POST"])
def attendance_stop():
    global camera_active
    camera_active = False
    time.sleep(0.2)
    subject = attendance_data.get("subject","")
    marked  = attendance_data.get("marked", set())
    today   = attendance_data.get("today","")
    if subject and marked:
        _save_attendance(subject, marked, today)
        names = [name for (_, name) in marked]
        return jsonify({"success":True,"message":f"{len(marked)} student(s) ki attendance saved successfully!","count":len(marked),"names":names,"subject":subject})
    return jsonify({"success":True,"message":"Session stopped. Face not found.","count":0,"names":[],"subject":subject})


@app.route("/api/attendance/records")
def get_records():
    subject = request.args.get("subject","").strip()
    if not subject:
        files = [f.replace(".csv","") for f in os.listdir(ATTENDANCE_PATH) if f.endswith(".csv")]
        return jsonify({"success": True, "subjects": files})
    att_file = os.path.join(ATTENDANCE_PATH, f"{subject}.csv")
    if not os.path.exists(att_file):
        return jsonify({"success": False, "message": f"'{subject}' records not found."})
    df = pd.read_csv(att_file, index_col=0)
    df.index.name = "Enrollment"
    df = df.reset_index()
    return jsonify({"success":True,"data":df.to_dict(orient="records"),"columns":list(df.columns)})


@app.route("/api/attendance/live")
def get_live_attendance():
    return jsonify({"success": True, "data": live_attendance})


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def get_status():
    return jsonify({
        "camera_active":  camera_active,
        "camera_mode":    camera_mode,
        "rec_status":     rec_status,
        "register_count": register_data.get("count", 0),
        "register_max":   register_data.get("max", REGISTER_MAX),
        "train_status":   train_status,
        "live_count":     len(live_attendance),
    })


# ══════════════════════════════════════════════════════════════════════════════
#  ULTRA-FAST CAMERA LOOP
# ══════════════════════════════════════════════════════════════════════════════
def camera_loop():
    global current_frame, camera_active, rec_status, register_data, live_attendance

    recognizer = None
    if camera_mode == "attendance" and os.path.exists(TRAIN_LABEL_PATH):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(TRAIN_LABEL_PATH)

    # DSHOW on Windows = much faster camera open
    backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
    cam = cv2.VideoCapture(0, backend)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    enc_params  = [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q]
    frame_count = 0
    last_faces  = []

    while camera_active:
        ret, frame = cam.read()
        if not ret:
            time.sleep(0.005)
            continue

        frame_count += 1
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # always compute

        if frame_count % DETECT_SKIP == 0:
            gray_eq = cv2.equalizeHist(gray_full)

            detected = _detector.detectMultiScale(
                gray_eq,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(detected) > 0:
                last_faces = [(int(x), int(y), int(w), int(h))
                              for (x,y,w,h) in detected]
            else:
                last_faces = []

        rec_status["detected"] = len(last_faces) > 0

        for (x, y, w, h) in last_faces:

            if camera_mode == "register":
                count = register_data.get("count", 0)
                mx    = register_data.get("max", REGISTER_MAX)
                if count < mx:
                    enr  = register_data["enrollment"]
                    name = register_data["name"]
                    path = os.path.join(TRAIN_IMG_PATH, f"{enr}_{name}")
                    os.makedirs(path, exist_ok=True)
                    face_img = cv2.resize(gray_full[y:y+h, x:x+w], (100, 100))
                    cv2.imwrite(os.path.join(path, f"{enr}_{count+1}.jpg"), face_img)
                    register_data["count"] = count + 1
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,100), 2)
                cv2.putText(frame, f"{register_data.get('count',0)}/{mx}",
                            (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,100), 2)
                if register_data.get("count",0) >= mx:
                    camera_active = False

            elif camera_mode == "attendance" and recognizer:
                try:
                    face_roi = cv2.resize(gray_full[y:y+h, x:x+w], (100, 100))
                    id_, conf = recognizer.predict(face_roi)

                    if conf < 85:  # FIX: 70 → 85 (zyada lenient, better recognition)
                        name = attendance_data["student_map"].get(id_, f"ID:{id_}")
                        key  = (id_, name)
                        if key not in attendance_data["marked"]:
                            attendance_data["marked"].add(key)
                            now_str = datetime.datetime.now().strftime("%H:%M:%S")
                            live_attendance.append({
                                "enrollment": id_,
                                "name": name,
                                "time": now_str
                            })
                        rec_status["name"]       = name
                        rec_status["confidence"] = round(float(conf), 1)
                        color, label = (0,255,100), f"{name} ({int(100-conf)}%)"
                    else:
                        rec_status["name"] = "Unknown"
                        color, label = (0,0,255), f"Unknown ({int(100-conf)}%)"
                except Exception as e:
                    color, label = (255,165,0), "Err"
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                cv2.putText(frame, label, (x, y-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ok, buf = cv2.imencode(".jpg", frame, enc_params)
        if ok:
            with frame_lock:
                current_frame = buf.tobytes()

    cam.release()
    rec_status["detected"] = False
    rec_status["name"]     = ""


def generate_frames():
    last = None
    while True:
        with frame_lock:
            frame = current_frame
        if frame and frame is not last:
            last = frame
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + frame + b"\r\n")
        else:
            time.sleep(0.01)


def _save_attendance(subject, marked_set, today):
    att_file = os.path.join(ATTENDANCE_PATH, f"{subject}.csv")
    df = pd.read_csv(att_file, index_col=0) if os.path.exists(att_file) \
         else pd.DataFrame(columns=["Enrollment","Name"]).set_index("Enrollment")
    if today not in df.columns: df[today] = 0
    for (enr, name) in marked_set:
        if enr not in df.index:
            df.loc[enr] = 0
            df.loc[enr,"Name"] = name
        df.loc[enr, today] = 1
    date_cols = [c for c in df.columns if c not in ("Name","Attendance%")]
    if date_cols:
        df["Attendance%"] = (df[date_cols].sum(axis=1)/len(date_cols)*100).round(1).astype(str)+"%"
    df.to_csv(att_file)


@app.route("/api/attendance/delete", methods=["POST"])
def delete_attendance():
    subject = request.json.get("subject", "").strip()
    if not subject:
        return jsonify({"success": False, "message": "Subject name !"})
    att_file = os.path.join(ATTENDANCE_PATH, f"{subject}.csv")
    if not os.path.exists(att_file):
        return jsonify({"success": False, "message": f"'{subject}' file not found."})
    os.remove(att_file)
    return jsonify({"success": True, "message": f"'{subject}' attendance deleted!"})

@app.route("/api/students")
def get_students():
    if not os.path.exists(STUDENT_DETAIL_PATH):
        return jsonify({"success": True, "students": []})
    df = pd.read_csv(STUDENT_DETAIL_PATH)
    df = df.drop_duplicates(subset=["Enrollment"], keep="last")
    return jsonify({"success": True, "students": df.to_dict(orient="records")})

@app.route("/api/attendance/export/<subject>")
def export_attendance(subject):
    att_file = os.path.join(ATTENDANCE_PATH, f"{subject}.csv")
    if not os.path.exists(att_file):
        return jsonify({"success": False, "message": "File not found."})
    from flask import send_file
    return send_file(att_file, as_attachment=True, download_name=f"{subject}_attendance.csv")


if __name__ == "__main__":
    print("\n🚀 AttendX ULTRA FAST → http://127.0.0.1:5000\n")
    app.run(debug=False, threaded=True, host="0.0.0.0", port=5000, use_reloader=False)
