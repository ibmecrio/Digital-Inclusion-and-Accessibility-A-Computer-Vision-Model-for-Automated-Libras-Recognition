import math
import numpy as np

def extract_relative_coords(detection, time_since_last_frame_ms, last_wrist_position):
    if not detection.hand_landmarks:
        return np.array([]), None
    
    landmarks = detection.hand_landmarks[0]
    treated_landmarks = list()
    
    handedness = detection.handedness[0][0].category_name
    current_wrist_position = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    
    # Subtracts point 0 coordinates from every point,
    # normalizing the position.
    for idx, coords in enumerate(landmarks):
        x_normalized = coords.x-landmarks[0].x
        y_normalized = coords.y-landmarks[0].y
        
        if idx == 0:
            z_normalized = 0.0
        else:
            z_normalized = coords.z
        
        # Mirrors the hand for consistency in training.
        if handedness == 'Left':
            x_normalized = -x_normalized
        
        treated_landmarks.append(x_normalized)
        treated_landmarks.append(y_normalized)
        treated_landmarks.append(z_normalized)
    
    # Estabilishing wrist velocity (not scaled yet)
    if last_wrist_position is not None and len(last_wrist_position) == 3:
        
        delta_time_s = (time_since_last_frame_ms / 1000.0) + 1e-6
        wrist_velocity = (current_wrist_position - last_wrist_position) / delta_time_s
        
        if handedness == 'Left':
            wrist_velocity[0] = -wrist_velocity[0]
            
    else:
        wrist_velocity = np.array([0.0, 0.0, 0.0])
        
    treated_landmarks.extend(wrist_velocity)
    
    # Rescales based on the distance between landmark 0 (wrist)
    # and landmark 9 (middle finger base)
    x9 = treated_landmarks[9*3]
    y9 = treated_landmarks[(9*3)+1]
    z9 = treated_landmarks[(9*3)+2]
    
    scale_factor = math.sqrt(x9**2 + y9**2 + z9**2)
    
    if scale_factor == 0:
        scale_factor = 1.0
        
    treated_landmarks = np.array(treated_landmarks)
    treated_landmarks = treated_landmarks / scale_factor
    
    return treated_landmarks, current_wrist_position