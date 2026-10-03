import cv2
import time
import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import joblib
import pandas as pd
import warnings
warnings.filterwarnings("ignore") 

import mediapipe as mp

from landmark_extractor import (
    get_elbow_angle, get_shoulder_angle, get_elbow_horizontal_offset,
    get_knee_angle, get_hip_angle, get_torso_angle,
    get_body_angle, get_hip_alignment
)
from exercise_counter import ExerciseCounter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_full.task"

def load_exercise_models(exercise_name):
    try:
        classifier = joblib.load(PROJECT_ROOT / "models" / f"{exercise_name}_classifier.pkl")
        label_encoder = joblib.load(PROJECT_ROOT / "models" / f"{exercise_name}_label_encoder.pkl")
        return classifier, label_encoder
    except FileNotFoundError:
        print(f"❌ Missing models for {exercise_name}. Please run `python src/train_model.py`.")
        sys.exit(1)

def get_feedback(features, is_correct, exercise_name):
    if is_correct:
        return "Good form."
    
    if exercise_name == "bicep_curl":
        if features["left_elbow_offset"] > 0.15:
            return "Keep your elbow closer to your body."
        elif features["left_shoulder_angle"] > 35:
            return "Keep your upper arm stable."
        else:
            return "Inconsistent arm position."
            
    elif exercise_name == "squat":
        if features["torso_angle"] > 40:
            return "Keep your chest up."
        elif features["left_knee_angle"] < 70:
            return "Don't squat too deep (protect knees)."
        else:
            return "Check your stance and depth."
            
    elif exercise_name == "pushup":
        if abs(features["hip_alignment"]) > 0.1:
            return "Keep your body straight."
        elif features["left_elbow_angle"] > 90:
            return "Go deeper on the push-up."
        else:
            return "Inconsistent push-up form."

class ExerciseSelectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FormFit AI - Select Exercise")
        self.root.geometry("300x250")
        self.selected_exercise = None
        
        # Bring window to front
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)
        
        ttk.Label(root, text="Which exercise do you want to perform?", font=("Arial", 12)).pack(pady=15)
        
        ttk.Button(root, text="Bicep Curls", command=lambda: self.select("bicep_curl")).pack(fill='x', padx=50, pady=5)
        ttk.Button(root, text="Squats", command=lambda: self.select("squat")).pack(fill='x', padx=50, pady=5)
        ttk.Button(root, text="Push-ups", command=lambda: self.select("pushup")).pack(fill='x', padx=50, pady=5)
        
    def select(self, exercise):
        self.selected_exercise = exercise
        self.root.destroy()

def main():
    root = tk.Tk()
    app = ExerciseSelectorApp(root)
    root.mainloop()
    
    exercise_name = app.selected_exercise
    if not exercise_name:
        print("No exercise selected. Exiting.")
        return

    classifier, label_encoder = load_exercise_models(exercise_name)
    
    # Configure Counter
    if exercise_name == "bicep_curl":
        counter = ExerciseCounter(up_threshold=60, down_threshold=160)
    elif exercise_name == "squat":
        counter = ExerciseCounter(up_threshold=160, down_threshold=90)
    else: # pushup
        counter = ExerciseCounter(up_threshold=160, down_threshold=90) # roughly same logic

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        sys.exit(1)

    print(f"✅ Starting {exercise_name} Tracking!")

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000)
            
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            height, width, _ = frame.shape
            
            cv2.rectangle(frame, (0, 0), (450, 320), (0, 0, 0), -1)
            cv2.putText(frame, f"FORMFIT AI - {exercise_name.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if result and result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                
                # Extract relevant features based on exercise
                feature_dict = {}
                
                # Check visibility
                is_visible = False
                missing_message = ""
                VISIBILITY_THRESHOLD = 0.5
                
                def is_landmark_visible(idx):
                    return landmarks[idx].visibility > VISIBILITY_THRESHOLD

                if exercise_name == "bicep_curl":
                    required_indices = [11, 13, 15, 23]
                    is_visible = all(is_landmark_visible(i) for i in required_indices)
                    missing_message = "Arm or hip not fully visible."
                elif exercise_name == "squat":
                    required_indices = [11, 12, 23, 24, 25, 26, 27, 28]
                    is_visible = all(is_landmark_visible(i) for i in required_indices)
                    missing_message = "Move back so your full body is visible."
                elif exercise_name == "pushup":
                    required_indices = [11, 12, 13, 14, 15, 16, 23, 27]
                    is_visible = all(is_landmark_visible(i) for i in required_indices)
                    missing_message = "Full body not visible."

                if is_visible:
                    left_shoulder = landmarks[11]
                    right_shoulder = landmarks[12]
                    left_elbow = landmarks[13]
                    right_elbow = landmarks[14]
                    left_wrist = landmarks[15]
                    left_hip = landmarks[23]
                    right_hip = landmarks[24]
                    left_knee = landmarks[25]
                    right_knee = landmarks[26]
                    left_ankle = landmarks[27]
                    right_ankle = landmarks[28]

                    main_angle = None

                    if exercise_name == "bicep_curl":
                        feature_dict["left_elbow_angle"] = get_elbow_angle(left_shoulder, left_elbow, left_wrist)
                        feature_dict["left_shoulder_angle"] = get_shoulder_angle(left_hip, left_shoulder, left_elbow)
                        feature_dict["left_elbow_offset"] = get_elbow_horizontal_offset(left_shoulder, left_elbow)
                        main_angle = feature_dict["left_elbow_angle"]
                        
                    elif exercise_name == "squat":
                        feature_dict["left_knee_angle"] = get_knee_angle(left_hip, left_knee, left_ankle)
                        feature_dict["right_knee_angle"] = get_knee_angle(right_hip, right_knee, right_ankle)
                        feature_dict["left_hip_angle"] = get_hip_angle(left_shoulder, left_hip, left_knee)
                        feature_dict["right_hip_angle"] = get_hip_angle(right_shoulder, right_hip, right_knee)
                        feature_dict["torso_angle"] = get_torso_angle(left_shoulder, left_hip)
                        main_angle = feature_dict["left_knee_angle"]
                        
                    elif exercise_name == "pushup":
                        feature_dict["left_elbow_angle"] = get_elbow_angle(left_shoulder, left_elbow, left_wrist)
                        feature_dict["right_elbow_angle"] = get_elbow_angle(right_shoulder, right_elbow, landmarks[16])
                        feature_dict["body_angle"] = get_body_angle(left_shoulder, left_hip, left_ankle)
                        feature_dict["hip_alignment"] = get_hip_alignment(left_shoulder, left_hip, left_ankle)
                        main_angle = feature_dict["left_elbow_angle"]

                    reps, state, _ = counter.update(main_angle)
                    
                    feature_vector = pd.DataFrame([feature_dict])
                    pred_idx = classifier.predict(feature_vector)[0]
                    predicted_label = label_encoder.inverse_transform([pred_idx])[0]
                    is_correct = (predicted_label == "correct")
                    
                    form_text = "GOOD" if is_correct else "INCORRECT"
                    form_color = (0, 255, 0) if is_correct else (0, 0, 255)
                    feedback = get_feedback(feature_dict, is_correct, exercise_name)
                    
                else:
                    form_text = "WAITING"
                    form_color = (255, 165, 0)
                    feedback = missing_message
                    reps = counter.reps
                    state = "--"

                for landmark in landmarks:
                    if landmark.visibility > VISIBILITY_THRESHOLD:
                        x, y = int(landmark.x * width), int(landmark.y * height)
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                cv2.putText(frame, f"Form: {form_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, form_color, 2)
                cv2.putText(frame, f"Feedback: {feedback}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                cv2.putText(frame, f"Repetitions: {reps}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"State: {state}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

            else:
                cv2.putText(frame, "No person detected.", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("FormFit AI", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()