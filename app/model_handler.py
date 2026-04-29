import joblib
import pandas as pd

MODEL_PATH = "models/model_v1.pkl"
FEATURES_PATH = "models/features_v1.pkl"

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

def predict_default(data: dict):
    X = pd.DataFrame([data])
    X = X[features]

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "model_version": "v1"
    }
