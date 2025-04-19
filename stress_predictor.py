import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# Constants
DATA_PATH = "student_lifestyle_dataset.csv"
MODEL_PATH = "stress_model.h5"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "encoder.pkl"

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Handle missing values
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
            
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def encode_features(df):
    """Encode categorical features"""
    le = LabelEncoder()
    df['Stress_Level'] = le.fit_transform(df['Stress_Level'])
    
    # Save the encoder for later use
    joblib.dump(le, ENCODER_PATH)
    
    return df, le

def scale_features(df):
    """Scale numerical features"""
    features_to_scale = [
        'Study_Hours_Per_Day', 
        'Extracurricular_Hours_Per_Day', 
        'Sleep_Hours_Per_Day',
        'Social_Hours_Per_Day', 
        'Physical_Activity_Hours_Per_Day', 
        'GPA'
    ]
    
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    # Save the scaler for later use
    joblib.dump(scaler, SCALER_PATH)
    
    return df, scaler

def explore_data(df):
    """Perform EDA and visualize data"""
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.savefig('static/correlation_matrix.png')
    plt.close()
    
    # Stress level distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Stress_Level', data=df)
    plt.title('Stress Level Distribution')
    plt.savefig('static/stress_distribution.png')
    plt.close()

def build_model(X_train):
    """Build and compile the neural network model"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """Train the model and evaluate performance"""
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    
    plt.savefig('static/training_history.png')
    plt.close()
    
    # Evaluate on test set
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'Moderate', 'High'], 
                yticklabels=['Low', 'Moderate', 'High'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('static/confusion_matrix.png')
    plt.close()
    
    return model

def save_model(model):
    """Save the trained model"""
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def main():
    # Step 1: Load and preprocess data
    df = load_and_preprocess_data()
    if df is None:
        return
    
    # Step 2: Encode categorical features
    df, le = encode_features(df)
    
    # Step 3: Scale numerical features
    df, scaler = scale_features(df)
    
    # Step 4: Explore data
    explore_data(df)
    
    # Step 5: Prepare features and target
    X = df.drop(['Student_ID', 'Stress_Level'], axis=1)
    y = df['Stress_Level']
    
    # Step 6: Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 7: Build and train model
    model = build_model(X_train)
    model = train_and_evaluate(model, X_train, y_train, X_test, y_test)
    
    # Step 8: Save model
    save_model(model)

if __name__ == "__main__":
    main()