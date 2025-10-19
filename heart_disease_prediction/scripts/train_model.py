import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

# Preprocess function
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Encode categorical features
    df = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])
    
    # Features and target
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    return X, y

if __name__ == "__main__":
    X, y = preprocess_data('../data/heart.csv')
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Save model
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/heart_model.pkl')
    print("💾 Model saved as '../models/heart_model.pkl'")
