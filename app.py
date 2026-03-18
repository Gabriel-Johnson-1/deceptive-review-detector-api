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
    try:
        data = request.json
        review = data.get("review")
        user_label = data.get("user_label")

        if not review:
            return jsonify({"error": "Missing review"}), 400

        X = vectorizer.transform([review])
        pred = model.predict(X)[0]  # now pred is 'deceptive' or 'truthful'

        return jsonify({
            "prediction": pred,           # keep as string
            "user_label": user_label
        })
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()