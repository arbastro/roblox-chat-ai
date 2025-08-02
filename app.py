from flask import Flask, request, jsonify
import joblib
import os
import traceback

app = Flask(__name__)

@app.route("/")
def home():
    return "Chat moderation AI is running!"

@app.route("/moderate", methods=["POST"])
def moderate():
    try:
        data = request.get_json()
        msg = data.get("message", "")
        vec = vectorizer.transform([msg])
        pred = model.predict(vec)[0]
        label = "poor_sportsmanship" if pred == 1 else "normal"
        return jsonify({"label": label})
    except Exception as e:
        print("❌ Error while predicting:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Load model and vectorizer at startup
try:
    print("📦 Loading model and vectorizer...")
    model = joblib.load("chat_model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    print("✅ Models loaded successfully!")
except Exception as e:
    print("❌ Failed to load model/vectorizer:")
    traceback.print_exc()
    exit(1)

# Start server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port)
