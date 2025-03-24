from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from src.preprocessing import preprocess_data
from src.model import build_model, train_model
from src.prediction import predict_single
from tensorflow.keras.models import load_model
import joblib
import os

app = Flask(__name__)

# Load pre-trained model and preprocessors
model = load_model("models/drug_prediction_model.tf")
label_encoders = joblib.load("models/label_encoders.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = predict_single(model, preprocessor, label_encoders, data)
    return jsonify({"predicted_drug": prediction})

@app.route('/upload', methods=['POST'])
def upload_data():
    file = request.files['file']
    new_df = pd.read_csv(file)
    X_new, y_new, _, _ = preprocess_data(new_df, label_encoders, preprocessor, is_new_data=True)
    
    X_train = np.load("data/train/X_train.npy")
    y_train = np.load("data/train/y_train.npy")
    X_combined = np.vstack((X_train, X_new))
    y_combined = np.hstack((y_train, y_new))
    
    num_classes = len(label_encoders["drug"].classes_)
    model, _ = train_model(model, X_combined, y_combined, X_combined, y_combined)
    model.save("models/drug_prediction_model.tf")
    return jsonify({"message": "Model retrained successfully"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)