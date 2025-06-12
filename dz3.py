import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import os

YOUR_NAME = "VLADIMIR"
YOUR_SURNAME = "LISITSYN"
REFERENCE_FACE_IMAGE = "owner.jpg"

FINGER_HISTORY_LENGTH = 5
STABLE_FINGER_COUNT_THRESHOLD = 3
finger_history = deque(maxlen=FINGER_HISTORY_LENGTH)

mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

def prepare_training_data(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise Exception(f"Cannot load reference face image from {image_path}")
    return [img], [1]

def train_on_reference():
    faces, labels = prepare_training_data(REFERENCE_FACE_IMAGE)
    face_recognizer.train(faces, np.array(labels))
    return faces[0].shape

def count_fingers(hand_landmarks, hand_label):
    fingers = []

    for tip_id in [8, 12, 16]:
        is_up = hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y
        fingers.append(int(is_up))

    return sum(fingers)

def detect_emotion(face_img):
    emotions = ["Happy", "Neutral", "Sad"]
    mean_intensity = np.mean(face_img)
    if mean_intensity > 100:
        return emotions[0]
    elif mean_intensity > 70:
        return emotions[1]
    else:
        return emotions[2]

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.6) as face_detection, \
     mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1) as hands:

    trained = os.path.exists(REFERENCE_FACE_IMAGE)
    face_shape = None
    if trained:
        face_shape = train_on_reference()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        recognized_label = 0
        face_text_pos = (50, 50)
        faces_results = face_detection.process(img_rgb)

        if faces_results.detections:
            for detection in faces_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = x1 + int(bbox.width * w)
                y2 = y1 + int(bbox.height * h)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face_roi = img_gray[y1:y2, x1:x2]

                if trained and face_roi.size > 0 and face_shape:
                    face_resized = cv2.resize(face_roi, (face_shape[1], face_shape[0]))
                    label, confidence = face_recognizer.predict(face_resized)
                    recognized_label = label if confidence < 70 else 0

                color = (0, 255, 0) if recognized_label == 1 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                face_text_pos = (x1, max(y1 - 10, 30))

        hand_results = hands.process(img_rgb)
        stable_fingers = 0

        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            handedness = hand_results.multi_handedness[0].classification[0].label
            current_count = count_fingers(hand_landmarks, handedness)
            finger_history.append(current_count)

            most_common = max(set(finger_history), key=finger_history.count)
            if finger_history.count(most_common) >= STABLE_FINGER_COUNT_THRESHOLD:
                stable_fingers = most_common
            else:
                stable_fingers = 0

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            finger_history.clear()

        if faces_results.detections:
            if recognized_label == 1:
                if stable_fingers == 1:
                    cv2.putText(frame, YOUR_NAME, face_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                elif stable_fingers == 2:
                    cv2.putText(frame, YOUR_SURNAME, face_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                elif stable_fingers == 3:
                    face_gray = img_gray[y1:y2, x1:x2]
                    if face_gray.size > 0:
                        emotion = detect_emotion(face_gray)
                        cv2.putText(frame, emotion, face_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", face_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(frame, "Press 's' to save face, 'ESC' to exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("Face & Hand Recognition", frame)

        key = cv2.waitKey(1)
        if key == ord('s') and faces_results.detections:
            for detection in faces_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = x1 + int(bbox.width * w)
                y2 = y1 + int(bbox.height * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                face_img = img_gray[y1:y2, x1:x2]
                if face_img.size > 0:
                    face_img_resized = cv2.resize(face_img, (200, 200))
                    cv2.imwrite(REFERENCE_FACE_IMAGE, face_img_resized)
                    print("Saved face image. Re-training model...")
                    trained = True
                    face_shape = train_on_reference()
                    print("Done.")
                    break

        elif key == 27:
            break

cap.release()
cv2.destroyAllWindows()
