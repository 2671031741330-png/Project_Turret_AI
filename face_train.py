import cv2
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# ============================================================ #
#                         PCA9685 SERVO                        #
# ============================================================ #
# Import PCA9685 servo driver library for controlling
# pan-tilt servo motors through I2C communication.

from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685

# ============================================================ #
#                         SYSTEM SETTINGS                      #
# ============================================================ #
# Main configuration for face collection, camera settings,
# dataset generation, image quality, and detection thresholds.

TARGET_IMAGES = 500
CAPTURE_INTERVAL = 0.25

FRAME_WIDTH = 854
FRAME_HEIGHT = 480
CAMERA_FPS = 30

FACE_SIZE = 224
MIN_FACE_SIZE = (80, 80)

BLUR_THRESHOLD = 80.0
SAVE_SHARPNESS_THRESHOLD = 120.0
DNN_CONFIDENCE = 0.55

BASE_DIR = Path("face_dataset")
MODELS_DIR = Path("models")

# Model files are now stored in models/
DNN_PROTO = MODELS_DIR / "deploy.prototxt"
DNN_MODEL = MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
FRONTAL_HAAR = MODELS_DIR / "haarcascade_frontalface_default.xml"
PROFILE_HAAR = MODELS_DIR / "haarcascade_profileface.xml"

PERSON_NAME = input("Enter person name: ").strip()
if PERSON_NAME == "":
    PERSON_NAME = "PEOPLE"

PERSON_DIR = BASE_DIR / PERSON_NAME

# ============================================================
#                 IMAGE TARGET PER POSE
# ============================================================

POSE_TARGETS = {
    "CENTER": 300,
    "LEFT": 100,
    "RIGHT": 100,
}

POSES = list(POSE_TARGETS.keys())
TARGET_IMAGES = sum(POSE_TARGETS.values())

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
ORANGE = (0, 165, 255)

# ============================================================ #
#                        SERVO SETTINGS                        #
# ============================================================ #
# Configure PCA9685 channels, servo angle limits,
# default startup angles, and reverse directions.

# Initialize I2C communication for PCA9685.
i2c = busio.I2C(SCL, SDA)

# Create PCA9685 object.
pca = PCA9685(i2c)

# Set PWM frequency for servo motors.
pca.frequency = 50

# Servo channel assignment.
PAN_CH = 0
TILT_CH = 1

# Maximum servo angle range.
PAN_MAX_ANGLE = 270
TILT_MAX_ANGLE = 180

# Initial startup position.
START_PAN_ANGLE = 135
START_TILT_ANGLE = 90

# Reverse servo direction if required.
SERVO_REVERSED_PAN = True
SERVO_REVERSED_TILT = False


# Convert angle value to PCA9685 duty cycle.
def angle_to_duty(angle, max_angle=180):

    pulse = 500 + (angle / max_angle) * 1900

    return int((pulse / 20000.0) * 65535)


# Send servo angle command to PCA9685.
def set_servo(channel, angle, reversed=False, max_angle=180):

    angle = max(0, min(max_angle, angle))

    real_angle = max_angle - angle if reversed else angle

    duty = angle_to_duty(real_angle, max_angle)

    pca.channels[channel].duty_cycle = duty

# ============================================================ #
#                    CHECK REQUIRED FILES                     #
# ============================================================ #
# Verify that all required AI model files exist before startup.

# Validate required DNN and Haar model files.
def check_required_files():
    required_files = [
        DNN_PROTO,
        DNN_MODEL,
        FRONTAL_HAAR,
        PROFILE_HAAR,
    ]

    missing = [str(f) for f in required_files if not f.exists()]

    if missing:
        print("[ERROR] Missing required model files:")
        for f in missing:
            print(" -", f)
        print("\nPlease download model files into the models/ folder first.")
        sys.exit(1)


check_required_files()

# ============================================================ #
#                  HAAR CASCADE POSE DETECTION               #
# ============================================================ #
# Loads frontal and profile Haar cascades for pose estimation.

FRONTAL_CASCADE = cv2.CascadeClassifier(str(FRONTAL_HAAR))
PROFILE_CASCADE = cv2.CascadeClassifier(str(PROFILE_HAAR))

if FRONTAL_CASCADE.empty():
    print(f"[ERROR] Cannot load frontal cascade: {FRONTAL_HAAR}")
    sys.exit(1)

if PROFILE_CASCADE.empty():
    print(f"[ERROR] Cannot load profile cascade: {PROFILE_HAAR}")
    sys.exit(1)


# Measure image sharpness using Laplacian variance.
def is_blurry(img, threshold=BLUR_THRESHOLD):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < threshold, score


# Check whether a file is an augmented image variant.
def is_augmented_file(name):
    suffixes = [
        "_dark_gamma",
        "_bright",
        "_contrast",
        "_pose_left",
        "_pose_right",
    ]
    return any(s in name for s in suffixes)


# Load available face detection systems.
def load_detectors():
    detectors = {}

    try:
        net = cv2.dnn.readNetFromCaffe(str(DNN_PROTO), str(DNN_MODEL))
        detectors["dnn"] = net
        print("[OK] DNN detector loaded")
    except Exception as e:
        print("[WARN] DNN unavailable:", e)

    haar = cv2.CascadeClassifier(str(FRONTAL_HAAR))
    if not haar.empty():
        detectors["haar"] = haar
        print("[OK] Haar detector loaded")
    else:
        print(f"[WARN] Haar detector unavailable: {FRONTAL_HAAR}")

    return detectors


# Detect faces using OpenCV DNN SSD model.
def detect_faces_dnn(net, frame):
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
    )

    net.setInput(blob)
    detections = net.forward()
    faces = []

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]

        if conf >= DNN_CONFIDENCE:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            fw = x2 - x1
            fh = y2 - y1

            if fw >= MIN_FACE_SIZE[0] and fh >= MIN_FACE_SIZE[1]:
                faces.append((x1, y1, fw, fh, float(conf)))

    return faces


# Detect faces using Haar Cascade fallback detector.
def detect_faces_haar(cascade, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    raw = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=MIN_FACE_SIZE,
    )

    faces = []
    for x, y, w, h in raw:
        faces.append((x, y, w, h, 0.80))

    return faces


# Main face detection pipeline.
def detect_faces(detectors, frame):
    faces = []

    if "dnn" in detectors:
        faces = detect_faces_dnn(detectors["dnn"], frame)

    if not faces and "haar" in detectors:
        faces = detect_faces_haar(detectors["haar"], frame)

    return sorted(faces, key=lambda f: f[2] * f[3], reverse=True)


# Apply gamma correction for brightness augmentation.
def adjust_gamma(image, gamma=1.5):
    table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in np.arange(256)
    ]).astype("uint8")

    return cv2.LUT(image, table)


# Rotate image to simulate different head poses.
def rotate_image(image, angle):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_REPLICATE,
    )


# Generate augmented face dataset variations.
def augment_face(face_img):
    variants = []

    dark_gamma = adjust_gamma(face_img, gamma=2.0)
    bright = cv2.convertScaleAbs(face_img, alpha=1.0, beta=25)
    contrast = cv2.convertScaleAbs(face_img, alpha=1.15, beta=0)
    pose_left = rotate_image(face_img, -8)
    pose_right = rotate_image(face_img, 8)

    variants.append(("dark_gamma", dark_gamma))
    variants.append(("bright", bright))
    variants.append(("contrast", contrast))
    variants.append(("pose_left", pose_left))
    variants.append(("pose_right", pose_right))

    return variants


# Crop and resize detected face region.
def crop_face(frame, box):
    x, y, w, h, conf = box

    margin_x = int(w * 0.22)
    margin_y = int(h * 0.28)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(frame.shape[1], x + w + margin_x)
    y2 = min(frame.shape[0], y + h + margin_y)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return cv2.resize(crop, (FACE_SIZE, FACE_SIZE))


# Estimate head pose using frontal/profile cascades.
def detect_head_pose_simple(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    front_faces = FRONTAL_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=MIN_FACE_SIZE,
    )

    if len(front_faces) > 0:
        return "CENTER"

    left_faces = PROFILE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=MIN_FACE_SIZE,
    )

    if len(left_faces) > 0:
        return "LEFT"

    flipped_gray = cv2.flip(gray, 1)

    right_faces = PROFILE_CASCADE.detectMultiScale(
        flipped_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=MIN_FACE_SIZE,
    )

    if len(right_faces) > 0:
        return "RIGHT"

    return None


# Draw all overlay UI elements on the preview window.
def draw_ui(
    frame,
    original_count,
    total_count,
    faces,
    fps,
    status_msg,
    blur_score=None,
    current_pose=None,
    detected_pose=None,
    pose_count=0,
):
    h, w = frame.shape[:2]

    for x, y, fw, fh, conf in faces:
        color = GREEN if conf >= 0.75 else YELLOW

        cv2.rectangle(
            frame,
            (x, y),
            (x + fw, y + fh),
            color,
            2,
        )

        cv2.putText(
            frame,
            f"{conf:.0%}",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    pct = min(original_count / TARGET_IMAGES, 1.0)
    bar_w = w - 40
    filled = int(bar_w * pct)

    cv2.rectangle(frame, (20, h - 50), (20 + bar_w, h - 30), (60, 60, 60), -1)
    cv2.rectangle(frame, (20, h - 50), (20 + filled, h - 30), GREEN, -1)
    cv2.rectangle(frame, (20, h - 50), (20 + bar_w, h - 30), WHITE, 1)

    cv2.putText(
        frame,
        f"Original {original_count}/{TARGET_IMAGES} | Total {total_count}",
        (20, h - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        WHITE,
        2,
    )

    cv2.putText(
        frame,
        f"FACE COLLECTOR | {PERSON_NAME}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        CYAN,
        2,
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.0f}",
        (w - 100, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        WHITE,
        1,
    )

    cv2.putText(
        frame,
        status_msg,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        ORANGE,
        2,
    )

    if blur_score is not None:
        cv2.putText(
            frame,
            f"Sharpness: {blur_score:.1f}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            WHITE,
            1,
        )

    if current_pose is not None:
        cv2.putText(
            frame,
            f"TARGET POSE: {current_pose} "
            f"{pose_count}/{POSE_TARGETS[current_pose]}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            CYAN,
            2,
        )

    if detected_pose is not None:
        cv2.putText(
            frame,
            f"DETECTED: {detected_pose}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            WHITE,
            2,
        )

    cv2.putText(
        frame,
        "[SPACE] Start/Pause   [Q] Quit",
        (20, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        WHITE,
        1,
    )

    return frame


# Main face dataset collection workflow.
def collect_faces():
    PERSON_DIR.mkdir(parents=True, exist_ok=True)

    detectors = load_detectors()
    if not detectors:
        print("[ERROR] No face detector loaded")
        return

    existing = list(PERSON_DIR.glob("*.jpg"))
    original_count = len([f for f in existing if not is_augmented_file(f.name)])
    total_count = len(existing)

    print("=" * 60)
    print("FACE DATASET COLLECTOR")
    print(f"Person       : {PERSON_NAME}")
    print(f"Output       : {PERSON_DIR}")
    print(f"Existing     : {original_count} original / {total_count} total")
    print(f"Target       : {TARGET_IMAGES} original images")
    print(f"Pose list    : {POSES}")
    print("Pose targets :")
    for pose, count in POSE_TARGETS.items():
        print(f" - {pose}: {count}")
    print("=" * 60)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot access camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # ============================================================ #
    #                    INITIAL SERVO POSITION                    #
    # ============================================================ #
    # Move pan-tilt servos to startup position before
    # entering the main face collection loop.

    print("[INIT] Set Servo Startup Position")

    # Set PAN servo to 135 degrees.
    set_servo(
        PAN_CH,
        START_PAN_ANGLE,
        SERVO_REVERSED_PAN,
        PAN_MAX_ANGLE,
    )

    # Set TILT servo to 90 degrees.
    set_servo(
        TILT_CH,
        START_TILT_ANGLE,
        SERVO_REVERSED_TILT,
        TILT_MAX_ANGLE,
    )

    # Allow servos time to reach position.
    time.sleep(1)

    paused = True
    last_cap = time.time()
    fps_time = time.time()
    fps_count = 0
    fps = 0.0

    status_msg = "Press SPACE to start"
    blur_score = None
    detected_pose = None

    current_pose_index = 0
    pose_count = 0

    cv2.namedWindow("Face Collector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Collector", 960, 560)

    # ============================================================ #
    #                         MAIN LOOP                          #
    # ============================================================ #
    # Main runtime loop for camera capture, face detection,
    # dataset collection, augmentation, and UI updates.
    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Failed to read camera frame")
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = cv2.flip(frame, 1)

        now = time.time()
        fps_count += 1

        if now - fps_time >= 1.0:
            fps = fps_count / (now - fps_time)
            fps_count = 0
            fps_time = now

        faces = detect_faces(detectors, frame)
        current_pose = POSES[current_pose_index]

        if not paused:
            detected_pose = detect_head_pose_simple(frame)

        if not paused and faces and (now - last_cap) >= CAPTURE_INTERVAL:
            best_face = faces[0]

            if detected_pose is None:
                status_msg = f"Move face to: {current_pose}"

            elif detected_pose != current_pose:
                status_msg = f"Need {current_pose}, detected {detected_pose}"

            else:
                face_img = crop_face(frame, best_face)

                if face_img is not None:
                    blurry, blur_score = is_blurry(face_img)

                    if blurry or blur_score < SAVE_SHARPNESS_THRESHOLD:
                        status_msg = f"Too blurry - hold still ({blur_score:.1f})"

                    else:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                        filename = PERSON_DIR / (
                            f"{PERSON_NAME}_{current_pose}_{timestamp}.jpg"
                        )

                        cv2.imwrite(
                            str(filename),
                            face_img,
                            [cv2.IMWRITE_JPEG_QUALITY, 95],
                        )

                        original_count += 1
                        total_count += 1
                        pose_count += 1

                        for suffix, aug in augment_face(face_img):
                            aug_file = PERSON_DIR / (
                                f"{PERSON_NAME}_{current_pose}_{timestamp}_{suffix}.jpg"
                            )

                            cv2.imwrite(
                                str(aug_file),
                                aug,
                                [cv2.IMWRITE_JPEG_QUALITY, 90],
                            )

                            total_count += 1

                        last_cap = now
                        status_msg = f"Saved {current_pose} {pose_count}/{POSE_TARGETS}"

                        if pose_count >= POSE_TARGETS[current_pose]:
                            pose_count = 0
                            current_pose_index += 1

                            if current_pose_index >= len(POSES):
                                current_pose_index = len(POSES) - 1
                                status_msg = "Complete!"

                            else:
                                next_pose = POSES[current_pose_index]
                                status_msg = f"Next pose: {next_pose}"
                                print(f"[INFO] Change pose to: {next_pose}")

                        if original_count % 20 == 0:
                            print(
                                f"[INFO] Saved original "
                                f"{original_count}/{TARGET_IMAGES}"
                            )

        elif not paused and not faces:
            status_msg = "No face detected"

        if original_count >= TARGET_IMAGES:
            status_msg = "Complete!"

            display = draw_ui(
                frame.copy(),
                original_count,
                total_count,
                faces,
                fps,
                status_msg,
                blur_score,
                current_pose,
                detected_pose,
                pose_count,
            )

            cv2.imshow("Face Collector", display)
            cv2.waitKey(2000)
            break

        display = draw_ui(
            frame.copy(),
            original_count,
            total_count,
            faces,
            fps,
            status_msg,
            blur_score,
            current_pose,
            detected_pose,
            pose_count,
        )

        if paused:
            cv2.putText(
                display,
                "PAUSED",
                (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                YELLOW,
                3,
            )

        cv2.imshow("Face Collector", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            print("[INFO] Quit")
            break

        if key == ord(" "):
            paused = not paused

            if paused:
                status_msg = "Paused"
                print("[INFO] Paused")
            else:
                status_msg = f"Collecting pose: {current_pose}"
                print(f"[INFO] Collecting started: {current_pose}")

    cap.release()
    cv2.destroyAllWindows()

    final_total = len(list(PERSON_DIR.glob("*.jpg")))
    final_original = len(
        [f for f in PERSON_DIR.glob("*.jpg") if not is_augmented_file(f.name)]
    )

    print("=" * 60)
    print("COLLECTION COMPLETE")
    print(f"Person          : {PERSON_NAME}")
    print(f"Original images : {final_original}")
    print(f"Total images    : {final_total}")
    print(f"Folder          : {PERSON_DIR}")
    print("=" * 60)


# ============================================================ #
#                         PROGRAM START                        #
# ============================================================ #
# Entry point for the face dataset collection system.

if __name__ == "__main__":
    collect_faces()
