#imports
import time
import os
import threading
import urllib.request
from collections import deque
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import tkinter as tk
from tkinter import messagebox


# setting up camera for recognition
CAMERA_INDEX = 0  # change this if your camera isn't at index 0

MODEL_PATH = "hand_landmarker.task" #this is the path where the model will be downloaded and saved
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
WINDOW_NAME = "Image"

# Download the hand landmark model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Initialize the hand landmarker
base_options = mp_python.BaseOptions(
    model_asset_path=MODEL_PATH,
    delegate=mp_python.BaseOptions.Delegate.CPU,)

options = vision.HandLandmarkerOptions(
    #we can set the number of hands to detect here
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.IMAGE,
)

# Initialize the hand landmarker
detector = vision.HandLandmarker.create_from_options(options)

# coordinates of the hand landmarks and their connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Fingertip landmark indices (thumb, index, middle, ring, pinky tips).
# We use the centroid of these as the "contact point" for hit detection,
# since a smash/swipe motion leads with the fingers, not the wrist.
FINGERTIP_IDS = [4, 8, 12, 16, 20]


# Separate tracking for each hand so a left-hand swipe never affects the right hand's state.
prev_right_wrist_y = None
prev_left_wrist_y = None
last_right_click_time = 0
last_left_click_time = 0
CLICK_COOLDOWN = 1.0        # seconds between allowed clicks (per hand)
MOVE_THRESHOLD = 0.06       # normalized y-distance (0-1 range) that counts as a "down" swipe

FLIP_HANDEDNESS_LABELS = False


def draw_landmarks(img, hand_landmarks):
    height, width, _ = img.shape
    points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in hand_landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(img, points[start], points[end], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)


def show_click_dialog(hand_label):
    # runs in its own thread so it doesn't freeze the camera feed
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Object Clicked", f"{hand_label} hand clicked")
    root.destroy()


def click_right(wrist_y):
    # when the RIGHT hand moves down in vertical direction, an "object clicked" dialog pops up
    global prev_right_wrist_y, last_right_click_time

    now = time.time()
    prev_y = prev_right_wrist_y
    prev_right_wrist_y = wrist_y

    if prev_y is None:
        return  # first frame we've seen this hand, nothing to compare yet

    delta_y = wrist_y - prev_y  # positive = moved DOWN (y grows downward in image coords)

    if delta_y > MOVE_THRESHOLD and (now - last_right_click_time) > CLICK_COOLDOWN:
        last_right_click_time = now
        threading.Thread(target=show_click_dialog, args=("Right",), daemon=True).start()


def click_left(wrist_y):
    # when the LEFT hand moves down in vertical direction, an "object clicked" dialog pops up
    global prev_left_wrist_y, last_left_click_time

    now = time.time()
    prev_y = prev_left_wrist_y
    prev_left_wrist_y = wrist_y

    if prev_y is None:
        return  # first frame we've seen this hand, nothing to compare yet

    delta_y = wrist_y - prev_y  # positive = moved DOWN (y grows downward in image coords)

    if delta_y > MOVE_THRESHOLD and (now - last_left_click_time) > CLICK_COOLDOWN:
        last_left_click_time = now
        threading.Thread(target=show_click_dialog, args=("Left",), daemon=True).start()


cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

if not cap.isOpened():
    print(f"Could not open camera at index {CAMERA_INDEX}. "
          f"Try a different CAMERA_INDEX (0, 1, 2...) and re-run.")


def main():
    start_time = time.time()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        attempt = 0
        success, img = cap.read()

        while not success and attempt < 5:
            time.sleep(0.2)
            success, img = cap.read()
            attempt += 1

        if not success:
            print("Failed to capture image after 5 attempts.")
            break

        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                draw_landmarks(img, hand_landmarks)
                wrist_y = hand_landmarks[0].y  # landmark 0 = wrist

                label = handedness[0].category_name  # "Left" or "Right"

                if label == "Right":
                    click_left(wrist_y)
                elif label == "Left":
                    click_right(wrist_y)

        cv2.imshow(WINDOW_NAME, img)

        # 'q' key OR the window's X button both end the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    for _ in range(4):
        cv2.waitKey(1)
    detector.close()

# ------------- Merging with the interface ------------------
# Rolling window of recent fingertip-centroid y positions per hand, used to
# detect a downward swipe as a net trend rather than a single-frame delta.
# A single-frame comparison is fragile: detection jitter or a slow inference
# call on any one frame can make an in-progress swipe register as "not
# moving" even while the hand is clearly heading down. Looking at the net
# change over the last few detections smooths that out.
_Y_HISTORY_LEN = 5
_y_history = {"Left": deque(maxlen=_Y_HISTORY_LEN), "Right": deque(maxlen=_Y_HISTORY_LEN)}
MOVE_DOWN_THRESHOLD = 0.015  # net normalized downward movement over the window to count as a swipe


def get_hand_state(frame):
    frame_h, frame_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    state = {
        "left_hand":  {"x": None, "y": None, "moving_down": False, "present": False, "landmarks": []},
        "right_hand": {"x": None, "y": None, "moving_down": False, "present": False, "landmarks": []},
    }

    seen_labels = set()

    if result.hand_landmarks:
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label = handedness[0].category_name
            seen_labels.add(label)

            # Contact point = centroid of the fingertips, not the wrist.
            # A smash/swipe leads with the fingers, so this point tracks much
            # closer to where the hand actually meets the falling object.
            tip_x = sum(hand_landmarks[i].x for i in FINGERTIP_IDS) / len(FINGERTIP_IDS)
            tip_y = sum(hand_landmarks[i].y for i in FINGERTIP_IDS) / len(FINGERTIP_IDS)
            px, py = int(tip_x * frame_w), int(tip_y * frame_h)

            # collect pixel coordinates for ALL 21 landmarks (fingertips, joints, wrist)
            landmark_points = [
                (int(lm.x * frame_w), int(lm.y * frame_h)) for lm in hand_landmarks
            ]

            history = _y_history[label]
            history.append(tip_y)

            # Net movement from the oldest to newest sample in the window.
            moving_down = len(history) >= 2 and (history[-1] - history[0]) > MOVE_DOWN_THRESHOLD

            key = "left_hand" if label == "Left" else "right_hand"
            state[key] = {
                "x": px, "y": py,
                "moving_down": moving_down,
                "present": True,
                "landmarks": landmark_points,
            }

    # Clear history for any hand that dropped out of frame this call, so a
    # gap in detection can't be misread as movement once it reappears.
    for label in ("Left", "Right"):
        if label not in seen_labels:
            _y_history[label].clear()

    return state