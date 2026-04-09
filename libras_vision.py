import cv2
import time
import mediapipe as mp
import extract
import joblib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)

detector = vision.HandLandmarker.create_from_options(options)

knn_clf = joblib.load('models/knn_model.joblib')
label_encoder = joblib.load('models/label_encoder.joblib')

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),# Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

cap = cv2.VideoCapture(0)
print("Iniciando captura... Pressione 'q' ou 'ESC' para sair.")

while cap.isOpened():
    
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    timestamp_ms = int(time.time() * 1000)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect_for_video(mp_image, timestamp_ms)
    
    if detection_result.hand_landmarks:
        
        # --- Sign processing ---
    
        landmarks_arr = extract.extract_relative_coords(detection_result)
        
        result = knn_clf.predict(landmarks_arr.reshape(1, -1))
        sign = label_encoder.inverse_transform(result)

        print(sign[0])
        
        # --- Landmarks visual processing ---
        
        h, w, c = frame.shape
        
        for hand_landmarks in detection_result.hand_landmarks:
            
            for connection in HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_lm = hand_landmarks[start_idx]
                end_lm = hand_landmarks[end_idx]
                
                start_x, start_y = int(start_lm.x * w), int(start_lm.y * h)
                end_x, end_y = int(end_lm.x * w), int(end_lm.y * h)
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

            for landmark in hand_landmarks:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    cv2.imshow('Libras Vision - Hand Tracking', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()