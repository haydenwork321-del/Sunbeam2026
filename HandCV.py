#imports
import time
import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# setting up camera for recognition
CAMERA_INDEX = 0  # change this if your camera isn't at index 0

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
WINDOW_NAME = "Image"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO,
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_landmarks(img, hand_landmarks):
    h, w, _ = img.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(img, points[start], points[end], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)


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
            for hand_landmarks in result.hand_landmarks:
                draw_landmarks(img, hand_landmarks)

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


if __name__ == "__main__":
    main()