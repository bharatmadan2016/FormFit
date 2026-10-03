import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-source-for-stubs]
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def train_exercise(csv_name, features, label_col, model_name):
    data_path = PROJECT_ROOT / "data" / csv_name
    if not data_path.exists():
        print(f"❌ Dataset not found at: {data_path}")
        return
        
    print(f"\n--- Training {model_name} ---")
    if csv_name.endswith('.csv') and 'dataset' in csv_name and csv_name != 'bicep_curl_dataset.csv':
        df = pd.read_csv(data_path, sep='\t').dropna()
    else:
        df = pd.read_csv(data_path).dropna()
        
    X = df[features]
    y = df[label_col]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(le, models_dir / f"{model_name}_label_encoder.pkl")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', max_depth=5)
    rf_model.fit(X_train, y_train)
    
    joblib.dump(rf_model, models_dir / f"{model_name}_classifier.pkl")
    print(f"Model saved to: {model_name}_classifier.pkl")

def main():
    print("==========================================")
    print("     FormFit AI - Multi-Model Training")
    print("==========================================")
    
    # 1. Bicep Curls
    train_exercise("bicep_curl_dataset.csv", ['left_elbow_angle', 'left_shoulder_angle', 'left_elbow_offset'], 'label', "bicep_curl")
    
    # 2. Squats
    train_exercise("squat_dataset.csv", ['left_knee_angle', 'right_knee_angle', 'left_hip_angle', 'right_hip_angle', 'torso_angle'], 'form', "squat")
    
    # 3. Push-ups
    train_exercise("push-up_curl_dataset.csv", ['left_elbow_angle', 'right_elbow_angle', 'body_angle', 'hip_alignment'], 'form', "pushup")
    
    print("==========================================")

if __name__ == "__main__":
    main()
