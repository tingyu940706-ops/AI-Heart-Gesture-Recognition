import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def get_pose_label(landmarks, width, height):
    def to_pixel(landmark):
        return int(landmark.x * width), int(landmark.y * height)

    left_wrist = to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_WRIST])
    right_wrist = to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])
    left_shoulder = to_pixel(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
    right_shoulder = to_pixel(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
    nose = to_pixel(landmarks[mp_pose.PoseLandmark.NOSE])

    wrist_dist = np.linalg.norm(np.array(left_wrist) - np.array(right_wrist))
    shoulder_dist = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
    chest_y = (left_shoulder[1] + right_shoulder[1]) // 2

    if abs(left_wrist[1] - chest_y) < 60 and abs(right_wrist[1] - chest_y) < 60 and wrist_dist < shoulder_dist * 0.8:
        return "Chest Heart"
    if left_wrist[1] < nose[1] and right_wrist[1] < nose[1] and wrist_dist < shoulder_dist * 2:
        return "Overhead Heart"
    if abs(left_wrist[1] - chest_y) < 60 and abs(right_wrist[1] - chest_y) < 60 and wrist_dist < shoulder_dist * 1.5:
        return "Small Heart"
    return None

def process_heart_pose_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    label = None
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        label = get_pose_label(result.pose_landmarks.landmark, frame.shape[1], frame.shape[0])
        if label:
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
    return frame, label
