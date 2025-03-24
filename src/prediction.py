import numpy as np
from tensorflow.keras.models import load_model
import joblib
import pandas as pd


def predict_single(model, preprocessor, label_encoders, data_point):
    df = pd.DataFrame([data_point], columns=["disease", "age", "gender", "severity"])
    for col in ["disease", "gender", "severity"]:
        df[col] = label_encoders[col].transform([df[col][0]])
    X = preprocessor.transform(df)
    pred_probs = model.predict(X)
    pred_class = np.argmax(pred_probs, axis=1)[0]
    return label_encoders["drug"].inverse_transform([pred_class])[0]

def predict_multiple(model, preprocessor, label_encoders, data_points):
    predictions = []
    for data_point in data_points:
        predictions.append(predict_single(model, preprocessor, label_encoders, data_point))
    return predictions

def main():
    # Load the model and preprocessor
    model = load_model("model.h5")
    preprocessor = joblib.load("preprocessor.pkl")

