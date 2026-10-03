# FormFit AI

## 1. Project Title
FormFit AI - Real-time Bicep Curl Analysis System

## 2. Problem Statement
Many individuals perform strength training exercises like bicep curls with incorrect form, which reduces the effectiveness of the exercise and increases the risk of injury. There is a need for an automated, accessible system that provides real-time feedback on exercise form using only a standard webcam.

## 3. Features
*   **Live Pose Detection:** Uses MediaPipe to track body landmarks in real-time.
*   **Automated Rep Counting:** Accurately counts repetitions based on elbow angle transitions.
*   **Form Classification:** Uses a Machine Learning model (Random Forest/Logistic Regression) to classify form as "Correct" or "Incorrect".
*   **Real-time Feedback:** Provides specific actionable text feedback based on incorrect joint angles and offsets.

## 4. Technology Stack
*   **Language:** Python 3
*   **Computer Vision:** OpenCV, MediaPipe
*   **Machine Learning:** scikit-learn, joblib
*   **Data Handling:** pandas, numpy
*   **Visualization:** matplotlib, seaborn

## 5. System Architecture
```
Webcam -> OpenCV -> MediaPipe Pose -> Landmarks -> Feature Extraction -> ML Classifier -> Form Result -> Feedback + Rep Counter -> Live Dashboard
```

## 6. Methodology
The system captures live video, extracts key body landmarks (shoulder, elbow, wrist, hip), calculates biomechanical features (angles, offsets), and feeds them into a trained ML classifier to determine form correctness. Concurrently, a state machine counts repetitions by monitoring the elbow angle.

## 7. Dataset Description
The dataset (`data/bicep_curl_dataset.csv`) contains samples of correct and incorrect bicep curls.
Features:
*   `left_elbow_angle`: Angle at the elbow joint.
*   `left_shoulder_angle`: Angle at the shoulder joint relative to the hip and elbow.
*   `left_elbow_offset`: Horizontal distance between the shoulder and elbow.
*   `label`: "correct" or "incorrect".

## 8. Feature Extraction
Features are extracted using vector math on the X and Y coordinates of the landmarks provided by MediaPipe. The angles are calculated using the dot product formula, and horizontal offset is a simple coordinate subtraction.

## 9. ML Model
The project trains a Logistic Regression and a Random Forest Classifier on the dataset. The model with the highest F1-score is selected and saved (`form_classifier.pkl`) for real-time inference.

## 10. Rep-counting Approach
The `BicepCurlCounter` uses a moving average to smooth the elbow angle. It tracks state transitions between "UP" (angle <= 60) and "DOWN" (angle >= 160) to count one full repetition.

## 11. Feedback Mechanism
If the form is predicted as "incorrect", the system analyzes the current angles and offsets to provide specific feedback, such as "Keep your elbow closer to your body" or "Keep your upper arm stable."

## 12. Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 13. How to train the model
Run the training script to generate the ML model and confusion matrix:
```bash
python3 src/train_model.py
```

## 14. How to run the application
Run the live pose detector:
```bash
python3 src/pose_detector.py
```

## 15. Expected Output
The application will open a webcam window. As you perform bicep curls, it will display the extracted angles, current repetition count, form classification (CORRECT/INCORRECT), and live feedback.

## 16. Project Structure
```
FormFit-AI-main/
├── data/
│   └── bicep_curl_dataset.csv
├── models/
│   ├── pose_landmarker_full.task
│   ├── form_classifier.pkl
│   └── label_encoder.pkl
├── results/
│   └── confusion_matrix.png
├── src/
│   ├── bicep_curl_counter.py
│   ├── camera_test.py
│   ├── data_collector.py
│   ├── landmark_extractor.py
│   ├── train_model.py
│   └── pose_detector.py
├── README.md
└── requirements.txt
```

## 17. Limitations
*   Requires good lighting and full visibility of the upper body.
*   The prototype currently analyzes only the left arm.
*   Feedback is based on heuristic thresholds which may vary slightly across different body types.

## 18. Future Scope
*   Extend analysis to other exercises (squats, pushups).
*   Add 3D depth analysis using `z` coordinates for better accuracy.
*   Implement a full UI for tracking workout history.
