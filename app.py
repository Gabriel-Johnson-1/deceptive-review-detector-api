from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load model + vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def home():
    return "API running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    review = data.get("review")
    user_label = data.get("user_label")  # what user thinks

    # Transform text
    X = vectorizer.transform([review])

    # Predict
    pred = model.predict(X)[0]
    prob = model.decision_function(X)[0]

    return jsonify({
        "prediction": int(pred),
        "confidence": float(prob),
        "user_label": user_label
    })

if __name__ == "__main__":
    app.run()