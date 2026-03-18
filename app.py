from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from pymongo import MongoClient
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Load SVM model + vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

# Connect to MongoDB Atlas
# Put your connection string in environment variable MONGO_URI
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client["review_db"]
collection = db["submissions"]

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

        # Make prediction
        X = vectorizer.transform([review])
        pred = model.predict(X)[0]
        try:
            confidence = float(model.decision_function(X)[0])
        except:
            confidence = None

        # Save to MongoDB
        submission_doc = {
            "review": review,
            "user_label": user_label,
            "model_prediction": pred,
        }
        collection.insert_one(submission_doc)

        return jsonify({
            "prediction": pred,
            "user_label": user_label
        })
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()