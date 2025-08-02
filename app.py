from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("chat_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

@app.route("/")
def home():
    return "Chat Moderation AI is Running!"

@app.route("/moderate", methods=["POST"])
def moderate():
    msg = request.json.get("message", "")
    X = vectorizer.transform([msg])
    pred = model.predict(X)[0]
    label = "poor_sportsmanship" if pred == 1 else "normal"
    return jsonify({"label": label})
