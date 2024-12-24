
# Gesture Recognition Project

## Overview
This project implements a hand gesture recognition system using MediaPipe and TensorFlow. The system is designed to recognize specific hand gestures and map them to corresponding actions or characters. It includes steps for data preprocessing, model training, evaluation, and real-time gesture recognition.

## Features
- **Hand Detection and Landmark Extraction**: Utilizes MediaPipe Hands for detecting and extracting hand landmarks.
- **Gesture Classification**: Implements a neural network using TensorFlow for classifying gestures.
- **Model Persistence**: Saves and loads trained models for reuse.
- **Performance Metrics**: Provides confusion matrix and accuracy plots for model evaluation.

## Requirements
To run this project, ensure you have the following installed:
- Python 3.7+
- OpenCV
- MediaPipe
- TensorFlow
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

Install the required libraries using:
```bash
pip install opencv-python mediapipe tensorflow numpy scikit-learn matplotlib seaborn tqdm
```

## Files
- `Gesture Recognition.ipynb`: The main Jupyter Notebook containing the implementation.
- `gesture_model.keras`: The trained model file (generated after training).
- `labels.pkl`: The label encoder file for mapping gestures to classes.

## Usage
1. **Data Preparation**:
   - Collect hand gesture images and preprocess them using MediaPipe Hands.
   - Split the data into training and testing sets.

2. **Model Training**:
   - Train the model using the neural network defined in the notebook.
   - Save the trained model and label encoder for future use.

3. **Model Evaluation**:
   - Use the provided evaluation functions to generate confusion matrices and accuracy plots.

4. **Real-time Recognition**:
   - Use a webcam feed to recognize gestures in real-time by loading the saved model.

## Example Code
```python
from HandGestureRecognizer import HandGestureRecognizer

# Initialize recognizer
recognizer = HandGestureRecognizer()

# Train model
recognizer.train(data_dir='path_to_data', epochs=10)

# Save model
recognizer.save_model('models/')

# Load and use model
recognizer.load_model('models/gesture_model.keras', 'models/labels.pkl')
result = recognizer.predict(image)
print(f"Predicted Gesture: {result}")
```

## Output
- **Confusion Matrix**: Visual representation of prediction accuracy across classes.
- **Real-time Predictions**: Predicted gestures displayed on the webcam feed.
