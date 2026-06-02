import cv2
import dlib
import numpy as np
import os
import time
import signal
import sys
from gpiozero import OutputDevice

# ============================================================ #
#                    PCA9685 SERVO SYSTEM                      #
# ============================================================ #
# This section initializes the PCA9685 servo controller.
# It controls PAN, TILT, and MOVE servo channels.

from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685

i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 50

PAN_CH = 0
TILT_CH = 1
MOVE_CH = 2

PAN_MAX_ANGLE = 270
TILT_MAX_ANGLE = 180

pan_angle = 135
tilt_angle = 90
move_angle = 0

move_triggered = False

move_loop_time = 0
move_state = 0

SERVO_REVERSED_PAN = True
SERVO_REVERSED_TILT = False


# Convert servo angle into PCA9685 duty cycle value.
def angle_to_duty(angle, max_angle=180):
    pulse = 500 + (angle / max_angle) * 1900
    return int((pulse / 20000.0) * 65535)


# Send angle command to selected servo channel.
def set_servo(channel, angle, reversed=False, max_angle=180):
    angle = max(0, min(max_angle, angle))
    real = max_angle - angle if reversed else angle
    duty = angle_to_duty(real, max_angle)
    pca.channels[channel].duty_cycle = duty


# Stop all servo PWM signals immediately.
def stop_all():
    pca.channels[PAN_CH].duty_cycle = 0
    pca.channels[TILT_CH].duty_cycle = 0


# ============================================================ #
#                         SYSTEM CONFIG                        #
# ============================================================ #
# Main configuration for face tracking, camera setup,
# detection intervals, thresholds, and optimization settings.

PEOPLE_DIR = "face_dataset"
MODELS_DIR = "models"

THRESHOLD = 0.43
DETECT_INTERVAL = 15
KNOWN_PRIORITY_INTERVAL = 10
MAX_LOST = 5
UNKNOWN_GRACE = 60

# FPS optimization
FRAME_WIDTH = 854
FRAME_HEIGHT = 480
CAMERA_FPS = 30

# Model files are now stored in models/
HAAR = os.path.join(MODELS_DIR, "haarcascade_frontalface_alt.xml")
PREDICTOR = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")
MODEL = os.path.join(MODELS_DIR, "dlib_face_recognition_resnet_model_v1.dat")

# Minimum box size
MIN_BOX_W = 25
MIN_BOX_H = 25

# No movement reset
#NO_MOVEMENT_THRESH = 5       # px
#NO_MOVEMENT_TIMEOUT = 10.0   # sec
last_box = None
#last_movement_time = time.time()

# ============================================================ #
#                    BUZZER & UNKNOWN ALARM                   #
# ============================================================ #
# Controls warning sounds when an unknown face is detected.

BUZZER_PIN = 22
UNKNOWN_ALARM_DELAY = 10.0
RAPID_BEEP_ON = 0.08
RAPID_BEEP_OFF = 0.08

buzzer = OutputDevice(BUZZER_PIN, active_high=False, initial_value=False)
buzzer.off()

unknown_start_time = None

buzzer_mode = "off"          # off / timed / rapid
buzzer_next_time = 0
buzzer_off_time = 0
buzzer_is_on = False
buzzer_last_second = -1

# Auto scan when no face detected
SCAN_MIN_ANGLE = 30
SCAN_MAX_ANGLE = 240
SCAN_STEP = 0.2
SCAN_INTERVAL = 0.03
SCAN_TILT_ANGLE = 90

scan_direction = 1
last_scan_time = time.time()
scan_mode_started = False


# Ensure face tracking box stays inside frame boundaries.
def normalize_box(box, frame):
    x, y, w, h = map(int, box)
    H, W = frame.shape[:2]

    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))

    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    return (int(x), int(y), int(w), int(h))


# Calculate movement distance between two tracking boxes.
def box_movement(b1, b2):
    if b1 is None or b2 is None:
        return 9999
    x1, y1, w1, h1 = b1
    cx1 = x1 + w1 // 2
    cy1 = y1 + h1 // 2
    x2, y2, w2, h2 = b2
    cx2 = x2 + w2 // 2
    cy2 = y2 + h2 // 2
    return np.hypot(cx1 - cx2, cy1 - cy2)


# ============================================================ #
#                     TRACKER VALIDATION                      #
# ============================================================ #
# Validates tracking boxes to prevent invalid or broken data.

def box_outside_frame(box, frame):
    x, y, w, h = map(int, box)
    H, W = frame.shape[:2]
    cx = x + w // 2
    cy = y + h // 2
    if cx < 0 or cx > W or cy < 0 or cy > H:
        return True

    inside_w = min(x + w, W) - max(x, 0)
    inside_h = min(y + h, H) - max(y, 0)
    if inside_w <= 0 or inside_h <= 0:
        return True

    inside_area = inside_w * inside_h
    box_area = w * h
    return (inside_area / box_area) < 0.6


def invalid_box(box, frame):
    x, y, w, h = map(int, box)
    H, W = frame.shape[:2]

    if w < 30 or h < 30:
        return True
    if w > W * 0.55 or h > H * 0.70:
        return True

    ratio = w / float(h)
    if ratio < 0.5 or ratio > 2.0:
        return True

    return False


# Recognize a face from a selected tracking region.
def recognize_face_from_box(frame, box):
    x, y, w, h = map(int, box)
    H, W = frame.shape[:2]

    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    face = frame[y:y + h, x:x + w]
    if face.size == 0:
        return None, 999

    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    dets = detector(rgb, 1)
    if len(dets) == 0:
        return None, 999

    shape = sp(rgb, dets[0])
    desc = np.array(rec_model.compute_face_descriptor(rgb, shape))

    best_name = None
    best_dist = 999

    for name, ref in KNOWN_FACES.items():
        d = np.linalg.norm(ref - desc)
        if d < best_dist:
            best_dist = d
            best_name = name

    return best_name, best_dist


# ============================================================ #
#                        PID CONTROLLER                       #
# ============================================================ #
# Smooth servo tracking system using PID calculations.

Kp_x_fast, Ki_x_fast, Kd_x_fast = 0.017, 0.00001, 0.0015
Kp_y_fast, Ki_y_fast, Kd_y_fast = 0.011, 0.00001, 0.0012

Kp_x_prec, Ki_x_prec, Kd_x_prec = 0.010, 0.000005, 0.0011
Kp_y_prec, Ki_y_prec, Kd_y_prec = 0.0058, 0.000005, 0.0007

FAST_ERROR_X = 130
FAST_ERROR_Y = 100

integral_x = integral_y = 0
prev_error_x = prev_error_y = 0
derivative_x = derivative_y = 0

DERIV_ALPHA_X = 0.9
DERIV_ALPHA_Y = 0.92
INTEGRAL_LIMIT = 800

deadband_x = 20
deadband_y = 25

MAX_STEP_X_FAST = 3.2
MAX_STEP_Y_FAST = 1.2

MAX_STEP_X_PREC = 1.2
MAX_STEP_Y_PREC = 0.9

SERVO_SMOOTH_FAST = 0.75
SERVO_SMOOTH_PREC = 0.60
SERVO_UPDATE_INTERVAL = 0.025
last_servo_time = time.time()

prev_time = time.time()
mode_text = "IDLE"

# ============================================================ #
#                         KALMAN FILTER                        #
# ============================================================ #
# Predicts face movement for smoother tracking.

kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
kalman.statePre = np.array([[0], [0], [0], [0]], np.float32)
kalman_initialized = False

# ============================================================ #
#                      FACE RECOGNITION MODELS                #
# ============================================================ #
# Loads Haar Cascade, Dlib predictor, and face embeddings.

required_files = [HAAR, PREDICTOR, MODEL]
for f in required_files:
    if not os.path.exists(f):
        print(f"[ERROR] Missing file: {f}")
        sys.exit(1)

if not os.path.exists(PEOPLE_DIR):
    print(f"[ERROR] Missing dataset folder: {PEOPLE_DIR}")
    sys.exit(1)

face_cascade = cv2.CascadeClassifier(HAAR)
if face_cascade.empty():
    print(f"[ERROR] Cannot load Haar cascade: {HAAR}")
    sys.exit(1)

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(PREDICTOR)
rec_model = dlib.face_recognition_model_v1(MODEL)

KNOWN_FACES = {}

# Load embeddings
for person_name in os.listdir(PEOPLE_DIR):
    person_path = os.path.join(PEOPLE_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    embeddings = []
    for img_name in os.listdir(person_path):
        if not img_name.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = detector(rgb, 1)
        if len(dets) == 0:
            continue

        shape = sp(rgb, dets[0])
        desc = np.array(rec_model.compute_face_descriptor(rgb, shape))
        embeddings.append(desc)

    if embeddings:
        KNOWN_FACES[person_name] = np.mean(embeddings, axis=0)

print("Known people:", list(KNOWN_FACES.keys()))

# ============================================================ #
#                        TRACKER SYSTEM                       #
# ============================================================ #
# Handles KNOWN and UNKNOWN face trackers.

KNOWN_TRACKER = None
UNKNOWN_TRACKER = None

CURRENT_KNOWN_NAME = None

KNOWN_LOST = 0
UNKNOWN_LOST = 0

LAST_KNOWN_FRAME = -999

# ============================================================ #
#                         CAMERA SYSTEM                       #
# ============================================================ #
# Initializes and configures the USB camera.

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)

if not cap.isOpened():
    print("[ERROR] Cannot access camera")
    sys.exit(1)

frame_id = 0

# ============================================================ #
#                     INITIAL SERVO POSITION                  #
# ============================================================ #
# Set default startup angles for all servos.

set_servo(PAN_CH, pan_angle, SERVO_REVERSED_PAN, PAN_MAX_ANGLE)
set_servo(TILT_CH, tilt_angle, SERVO_REVERSED_TILT, TILT_MAX_ANGLE)
set_servo(MOVE_CH, move_angle)
time.sleep(1)


# ============================================================ #
#                         CLEANUP SYSTEM                      #
# ============================================================ #
# Safely shutdown servos, buzzer, camera, and windows.

# Shutdown handler for CTRL+C or program exit.
def cleanup(sig=None, frame=None):
    try:
        buzzer.off()
    except Exception:
        pass

    try:
        stop_all()
        pca.deinit()
    except Exception:
        pass

    try:
        cap.release()
    except Exception:
        pass

    cv2.destroyAllWindows()
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)


# ============================================================ #
#                       BUZZER FUNCTIONS                      #
# ============================================================ #
# Helper functions for timed and rapid buzzer alerts.

# Disable all buzzer activity.
def stop_buzzer_alarm():
    global buzzer_mode, buzzer_is_on, buzzer_next_time
    global buzzer_off_time, buzzer_last_second

    buzzer.off()
    buzzer_mode = "off"
    buzzer_is_on = False
    buzzer_next_time = 0
    buzzer_off_time = 0
    buzzer_last_second = -1


# Start buzzer for a fixed duration.
def start_timed_beep(duration):
    global buzzer_mode, buzzer_is_on, buzzer_off_time

    buzzer.on()
    buzzer_mode = "timed"
    buzzer_is_on = True
    buzzer_off_time = time.time() + duration


# Start continuous rapid beep alarm.
def start_rapid_beep():
    global buzzer_mode, buzzer_is_on, buzzer_next_time

    buzzer_mode = "rapid"
    buzzer_is_on = False
    buzzer_next_time = time.time()


# Update buzzer state machine.
def update_buzzer():
    global buzzer_mode, buzzer_is_on, buzzer_next_time

    now = time.time()

    if buzzer_mode == "timed":
        if now >= buzzer_off_time:
            buzzer.off()
            buzzer_mode = "off"
            buzzer_is_on = False
        return

    if buzzer_mode == "rapid":
        if now < buzzer_next_time:
            return

        if buzzer_is_on:
            buzzer.off()
            buzzer_is_on = False
            buzzer_next_time = now + RAPID_BEEP_OFF
        else:
            buzzer.on()
            buzzer_is_on = True
            buzzer_next_time = now + RAPID_BEEP_ON

# ============================================================ #
#                           MAIN LOOP                         #
# ============================================================ #
# Main runtime loop for face detection, tracking,
# servo movement, alarms, and visualization.

while True:
    ret, frame = cap.read()
    if not ret:
        break

    update_buzzer()
    # ====== MOVE_CH LOOP ======
# Controls repeated MOVE_CH servo movement sequence.

    if move_triggered:
        now_move = time.time()

        # 
        if move_state == 0:
            set_servo(MOVE_CH, 45)
            move_state = 1
            move_loop_time = now_move

        #
        elif move_state == 1:
            if now_move - move_loop_time >= 0.2:
                set_servo(MOVE_CH, 0)
                move_state = 2
                move_loop_time = now_move

        #
        elif move_state == 2:
            if now_move - move_loop_time >= 2.0:
                move_state = 0
                move_loop_time = now_move
    
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame = cv2.flip(frame, 1)

    frame_id += 1

    H, W = frame.shape[:2]
    cx_frame = W // 2
    cy_frame = H // 2

    cv2.drawMarker(frame, (cx_frame, cy_frame),
                   (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    active_box = None
    tracking_type = None

    # ====================================================
    #                 UPDATE KNOWN TRACKER
    # ====================================================
    if KNOWN_TRACKER is not None:
        ok, box = KNOWN_TRACKER.update(frame)
        if not ok or box_outside_frame(box, frame) or invalid_box(box, frame):
            KNOWN_LOST += 1
        else:
            KNOWN_LOST = 0
            active_box = box
            tracking_type = "known"
            LAST_KNOWN_FRAME = frame_id

            x, y, w, h = map(int, box)
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
            cv2.putText(frame, CURRENT_KNOWN_NAME, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if KNOWN_LOST > MAX_LOST:
            print("Known lost:", CURRENT_KNOWN_NAME)
            KNOWN_TRACKER = None
            CURRENT_KNOWN_NAME = None
            KNOWN_LOST = 0

    # ====================================================
    #                UPDATE UNKNOWN TRACKER
    # ====================================================
    if active_box is None and UNKNOWN_TRACKER is not None:
        ok, box = UNKNOWN_TRACKER.update(frame)
        if not ok or box_outside_frame(box, frame) or invalid_box(box, frame):
            UNKNOWN_LOST += 1
        else:
            UNKNOWN_LOST = 0
            if frame_id - LAST_KNOWN_FRAME > UNKNOWN_GRACE:
                active_box = box
                tracking_type = "unknown"
                x, y, w, h = map(int, box)
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 0, 255), 2)
                cv2.putText(frame, "UNKNOWN", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if UNKNOWN_LOST > MAX_LOST:
            print("UNKNOWN RESET")
            UNKNOWN_TRACKER = None
            UNKNOWN_LOST = 0

            unknown_start_time = None
            stop_buzzer_alarm()
            set_servo(MOVE_CH, 0)
            move_triggered = False
            move_state = 0

    # ====================================================
    #                 KNOWN PRIORITY SCAN
    # ====================================================
    if tracking_type == "unknown" and frame_id % KNOWN_PRIORITY_INTERVAL == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            if w < MIN_BOX_W or h < MIN_BOX_H:
                continue

            box = normalize_box((x, y, w, h), frame)
            best_name, best_dist = recognize_face_from_box(frame, box)

            if best_dist < THRESHOLD:
                print("SWITCH UNKNOWN -> KNOWN:", best_name)

                UNKNOWN_TRACKER = None
                UNKNOWN_LOST = 0

                KNOWN_TRACKER = cv2.legacy.TrackerCSRT_create()
                KNOWN_TRACKER.init(frame, box)
                CURRENT_KNOWN_NAME = best_name
                KNOWN_LOST = 0
                LAST_KNOWN_FRAME = frame_id

                active_box = box
                tracking_type = "known"

                kalman_initialized = False
                integral_x = integral_y = 0
                prev_error_x = prev_error_y = 0
                derivative_x = derivative_y = 0
                last_box = None
                mode_text = "KNOWN PRIORITY"

                break

    # ====================================================
    #                    FACE DETECTION
    # ====================================================
    if active_box is None and frame_id % DETECT_INTERVAL == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            if KNOWN_TRACKER is not None:
                break

            if w < MIN_BOX_W or h < MIN_BOX_H:
                continue

            box = normalize_box((x, y, w, h), frame)
            best_name, best_dist = recognize_face_from_box(frame, box)

            if best_dist < THRESHOLD:
                print("TRACK KNOWN:", best_name)

                KNOWN_TRACKER = cv2.legacy.TrackerCSRT_create()
                KNOWN_TRACKER.init(frame, box)
                CURRENT_KNOWN_NAME = best_name
                KNOWN_LOST = 0
                LAST_KNOWN_FRAME = frame_id
                break

            elif UNKNOWN_TRACKER is None and frame_id - LAST_KNOWN_FRAME > UNKNOWN_GRACE:
                print("TRACK UNKNOWN")

                UNKNOWN_TRACKER = cv2.legacy.TrackerCSRT_create()
                UNKNOWN_TRACKER.init(frame, box)
                UNKNOWN_LOST = 0
                break

    # ============================================================
    #                       SERVO CONTROL
    # ============================================================
    if active_box is not None:
        scan_mode_started = False

        # ================= UNKNOWN ALARM TIMER =================
        if tracking_type == "unknown":
            if unknown_start_time is None:
                unknown_start_time = time.time()
                buzzer_last_second = -1
                stop_buzzer_alarm()

            unknown_duration = time.time() - unknown_start_time
            current_sec = int(unknown_duration)

            if current_sec in [10, 11, 12, 13] and current_sec != buzzer_last_second:
                beep_duration = (current_sec - 9) * 0.1
                print(f"[ALARM] UNKNOWN {current_sec} sec -> beep {beep_duration:.1f} sec")
                start_timed_beep(beep_duration)
                buzzer_last_second = current_sec

            elif current_sec >= 14 and buzzer_mode != "rapid":
                print("[ALARM] UNKNOWN 14 sec -> rapid beep until target lost")
                start_rapid_beep()
                buzzer_last_second = current_sec

            # ===== MOVE SERVO AT 15 SEC =====
            if current_sec >= 15 and not move_triggered:
                print("[MOVE] Servo MOVE_CH -> LOOP 40 <-> 0")

                move_triggered = True
                move_loop_time = time.time()
                move_state = 0

        else:
            unknown_start_time = None
            stop_buzzer_alarm()
            set_servo(MOVE_CH, 0)
            move_triggered = False
            move_state = 0

        # ====================================================
        #         FIX SMALL BOX + NO MOVEMENT RESET
        # ====================================================
      #  movement = box_movement(active_box, last_box)

       # if movement < NO_MOVEMENT_THRESH:
          #  if time.time() - last_movement_time > NO_MOVEMENT_TIMEOUT:
              #  print("[RESET] No movement for 10 sec -> reset tracker")

               # KNOWN_TRACKER = None
               # UNKNOWN_TRACKER = None
               # CURRENT_KNOWN_NAME = None
              #  KNOWN_LOST = 0
             #   UNKNOWN_LOST = 0

               # unknown_start_time = None
               # stop_buzzer_alarm()
                #set_servo(MOVE_CH, 0)
              #  move_triggered = False
            #    move_state = 0
                
               # kalman_initialized = False
               # integral_x = integral_y = 0
              #  prev_error_x = prev_error_y = 0
               # derivative_x = derivative_y = 0
               # last_box = None
                #mode_text = "RESET"

               # continue
        #else:
            #last_movement_time = time.time()

        last_box = active_box

        # ====================================================
        #        Compute face_x / face_y for Kalman
        # ====================================================
        x, y, w, h = map(int, active_box)
        face_x = x + w // 2
        face_y = y + h // 2

        # ====================================================
        #                        KALMAN
        # ====================================================
        measurement = np.array([[np.float32(face_x)],
                                [np.float32(face_y)]])

        if not kalman_initialized:
            kalman.statePre = np.array([[face_x],
                                        [face_y],
                                        [0],
                                        [0]], np.float32)
            kalman_initialized = True

        kalman.predict()
        estimated = kalman.correct(measurement)

        use_x = int(estimated[0])
        use_y = int(estimated[1])

        cv2.circle(frame, (use_x, use_y), 6, (255, 0, 0), -1)
        cv2.putText(frame, mode_text,
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 0), 2)

        # ====================================================
        #                        PID
        # ====================================================
        error_x = cx_frame - use_x
        error_y = use_y - cy_frame

        now = time.time()
        dt = now - prev_time
        dt = max(0.02, min(dt, 0.08))
        prev_time = now

        if abs(error_x) > deadband_x or abs(error_y) > deadband_y:
            # ===== 2-MODE PID =====
            if abs(error_x) > FAST_ERROR_X or abs(error_y) > FAST_ERROR_Y:
                Kp_x, Ki_x, Kd_x = Kp_x_fast, Ki_x_fast, Kd_x_fast
                Kp_y, Ki_y, Kd_y = Kp_y_fast, Ki_y_fast, Kd_y_fast
                max_step_x = MAX_STEP_X_FAST
                max_step_y = MAX_STEP_Y_FAST
                servo_smooth = SERVO_SMOOTH_FAST
                mode_text = "FAST"
            else:
                Kp_x, Ki_x, Kd_x = Kp_x_prec, Ki_x_prec, Kd_x_prec
                Kp_y, Ki_y, Kd_y = Kp_y_prec, Ki_y_prec, Kd_y_prec
                max_step_x = MAX_STEP_X_PREC
                max_step_y = MAX_STEP_Y_PREC
                servo_smooth = SERVO_SMOOTH_PREC
                mode_text = "PRECISION"

            # PAN
            integral_x += error_x * dt
            integral_x = max(-INTEGRAL_LIMIT, min(INTEGRAL_LIMIT, integral_x))

            dx = (error_x - prev_error_x) / dt if dt > 0 else 0
            derivative_x = DERIV_ALPHA_X * derivative_x + (1 - DERIV_ALPHA_X) * dx

            out_x = Kp_x * error_x + Ki_x * integral_x + Kd_x * derivative_x
            out_x = max(-max_step_x, min(max_step_x, out_x))

            target_pan = pan_angle + out_x
            pan_angle = pan_angle + (target_pan - pan_angle) * servo_smooth

            # TILT
            integral_y += error_y * dt
            integral_y = max(-INTEGRAL_LIMIT, min(INTEGRAL_LIMIT, integral_y))

            dy = (error_y - prev_error_y) / dt if dt > 0 else 0
            derivative_y = DERIV_ALPHA_Y * derivative_y + (1 - DERIV_ALPHA_Y) * dy

            out_y = Kp_y * error_y + Ki_y * integral_y + Kd_y * derivative_y
            out_y = max(-max_step_y, min(max_step_y, out_y))

            target_tilt = tilt_angle - out_y
            tilt_angle = tilt_angle + (target_tilt - tilt_angle) * servo_smooth

            # Clamp limits
            pan_angle = max(30, min(240, pan_angle))
            tilt_angle = max(70, min(105, tilt_angle))

            if time.time() - last_servo_time >= SERVO_UPDATE_INTERVAL:
                set_servo(PAN_CH, pan_angle, SERVO_REVERSED_PAN, PAN_MAX_ANGLE)
                set_servo(TILT_CH, tilt_angle, SERVO_REVERSED_TILT, TILT_MAX_ANGLE)
                last_servo_time = time.time()

            prev_error_x = error_x
            prev_error_y = error_y

        else:
            mode_text = "LOCK"
            integral_x *= 0.8
            integral_y *= 0.8
            derivative_x *= 0.5
            derivative_y *= 0.5
            prev_error_x = error_x
            prev_error_y = error_y

    else:
        # No face detected -> enter auto scan mode
        mode_text = "SCAN"
        kalman_initialized = False
        integral_x = integral_y = 0
        prev_error_x = prev_error_y = 0
        derivative_x = derivative_y = 0
        last_box = None

        # Reset TILT/Y only once when entering scan mode
        if not scan_mode_started:
            tilt_angle = SCAN_TILT_ANGLE
            set_servo(TILT_CH, tilt_angle, SERVO_REVERSED_TILT)
            scan_mode_started = True
            print("[SCAN MODE] Reset tilt to", SCAN_TILT_ANGLE)

        now = time.time()
        if now - last_scan_time >= SCAN_INTERVAL:
            pan_angle += SCAN_STEP * scan_direction

            if pan_angle >= SCAN_MAX_ANGLE:
                pan_angle = SCAN_MAX_ANGLE
                scan_direction = -1
            elif pan_angle <= SCAN_MIN_ANGLE:
                pan_angle = SCAN_MIN_ANGLE
                scan_direction = 1

            set_servo(PAN_CH, pan_angle, SERVO_REVERSED_PAN, PAN_MAX_ANGLE)
            set_servo(TILT_CH, tilt_angle, SERVO_REVERSED_TILT, TILT_MAX_ANGLE)

            last_scan_time = now
    
    cv2.imshow("Face Tracking Pan-Tilt", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cleanup()
