# FormFit AI - Academic Report (Milestone 2)

## 1. Title
FormFit AI - Real-time Bicep Curl Analysis System

## 2. Abstract
Strength training is beneficial but can lead to injury if performed with incorrect form. FormFit AI is a real-time computer vision system that uses an ordinary webcam to track user movements and provide live feedback. By leveraging MediaPipe for pose detection and a Random Forest classification model, the system analyzes the bicep curl exercise, counts repetitions, and categorizes form as correct or incorrect, providing specific actionable feedback to the user.

## 3. Introduction
Proper form is essential in weightlifting to maximize muscle activation and minimize injury risk. However, not everyone has access to a personal trainer. This project introduces FormFit AI, an accessible software solution that acts as a virtual spotter for the bicep curl.

## 4. Problem Domain Identification and Analysis
The problem domain is sports science and human-computer interaction (HCI). Novice lifters often struggle to maintain proper posture and joint alignment, leading to suboptimal workouts and strain injuries. The need for accessible, low-cost biomechanical analysis is critical.

## 5. Problem Statement
To design and implement a real-time exercise form analysis system that uses a standard webcam to detect body landmarks, calculate joint angles, and provide immediate feedback on bicep curl mechanics using machine learning techniques.

## 6. Existing Systems
Existing solutions include:
1. Wearable sensors (e.g., IMUs on joints), which are expensive and intrusive.
2. Premium gym equipment with built-in 3D cameras (e.g., Tonal), which are costly and require specialized hardware.
3. Simple mobile apps that count reps using gyroscope data but cannot visually analyze form.

## 7. Limitations of Existing Systems
* High hardware costs.
* Intrusiveness of wearable tech.
* Lack of specific form correction feedback in simple rep-counting apps.

## 8. Feasibility Study
*   **Technical feasibility:** High. MediaPipe provides robust, lightweight pose tracking, and scikit-learn offers efficient ML algorithms that can run real-time on standard laptops.
*   **Operational feasibility:** High. The system only requires a webcam and runs locally without needing complex setup.
*   **Economic feasibility:** High. Open-source libraries are used, keeping software costs at zero.

## 9. Proposed System
FormFit AI processes video frames, extracts human pose landmarks, calculates biomechanical features (elbow angle, shoulder angle, offset), counts valid repetitions, and classifies the form in real-time, providing textual feedback on an interactive dashboard.

## 10. Objectives
*   Extract human pose landmarks accurately.
*   Calculate relevant joint angles and offsets.
*   Count bicep curl repetitions reliably.
*   Train a machine learning classifier to differentiate between correct and incorrect form.
*   Provide real-time visual feedback.

## 11. Methodology
The project follows an iterative development methodology:
1. Data collection of correct and incorrect forms.
2. Feature engineering (calculating angles from x,y coordinates).
3. Model training and evaluation.
4. Integration into a live OpenCV video pipeline.
5. Implementation of a state-machine based rep counter.

## 12. System Architecture
```
Webcam -> OpenCV -> MediaPipe Pose -> Landmarks -> Feature Extraction -> ML Classifier -> Form Result -> Feedback + Rep Counter -> Live Dashboard
```

## 13. Data Collection
A dataset of 154 samples was collected locally using `data_collector.py`, representing both correct form and intentionally incorrect form (e.g., swinging the arm, excessive elbow movement). 

## 14. Feature Extraction
Features extracted include:
*   `left_elbow_angle`
*   `left_shoulder_angle`
*   `left_elbow_offset`

## 15. Algorithms and Techniques
*   **MediaPipe Blazepose:** For landmark extraction.
*   **Vector Mathematics (Dot Product):** To calculate angles between joints.
*   **Moving Average:** To smooth noisy landmark data for the rep counter.
*   **Random Forest Classifier:** An ensemble learning method used for form classification.

## 16. Machine Learning Model
A Random Forest Classifier was trained using 80% of the dataset, achieving a balance between accuracy and generalization. The data was scaled/encoded, and classes were balanced using `class_weight='balanced'`.

## 17. Rep Counting
The rep counter uses a state machine. It transitions to "UP" when the smoothed elbow angle goes below 60 degrees, and transitions to "DOWN" when it exceeds 160 degrees, incrementing the rep count on the DOWN transition.

## 18. Form Classification & Feedback System
The Random Forest model classifies the live feature vector. If deemed incorrect, a heuristic feedback system checks the specific angles (e.g., offset > 0.15) and advises the user (e.g., "Keep your elbow closer to your body").

## 19. Implementation
The final implementation synchronizes the MediaPipe asynchronous callback with an OpenCV rendering loop. The model and label encoder are loaded via `joblib`, ensuring the live application matches the training pipeline exactly.

## 20. Experimental Results
*   **Model Selected:** Random Forest
*   **Accuracy:** 0.7742
*   **Precision:** 0.7707
*   **Recall:** 0.7742
*   **F1-Score:** 0.7716

## 21. Confusion Matrix
The confusion matrix is saved in `results/confusion_matrix.png`, visualizing the true positives, false positives, true negatives, and false negatives of the test set.

## 22. Limitations
*   System is sensitive to lighting conditions and camera angle.
*   Only tracks the left arm currently.
*   This is an academic prototype and is not a certified medical or injury prevention diagnostic tool.

## 23. Future Scope
*   Extend to full-body exercises like squats and deadlifts.
*   Incorporate temporal sequences (LSTMs) for form analysis across the entire movement rather than frame-by-frame.
*   Develop a robust frontend UI.

## 24. Conclusion
FormFit AI demonstrates the feasibility of combining open-source computer vision and machine learning for accessible, real-time exercise form analysis, successfully fulfilling the objectives of the proposed system.

## 25. References
*   [ADD SOURCE: MediaPipe Pose documentation]
*   [ADD SOURCE: Scikit-learn documentation]
*   [ADD SOURCE: OpenCV documentation]
