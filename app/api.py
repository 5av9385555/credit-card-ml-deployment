from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# загрузка модели
model = joblib.load("models/model_v1.pkl")


@app.route("/")
def home():
    return jsonify({"message": "Credit Card ML API работает"})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = np.array([
            data["age"],
            data["income"],
            data["balance"]
        ]).reshape(1, -1)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
