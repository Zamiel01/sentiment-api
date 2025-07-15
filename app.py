from flask import Flask, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)

# Load model from HuggingFace Hub (downloaded automatically on first run)
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def map_sentiment(label):
    label = label.lower()
    if "1 star" in label or "2 star" in label:
        return "negative"
    elif "3 star" in label:
        return "neutral"
    else:
        return "positive"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("feedback_text", "")
    if not text:
        return jsonify({"error": "No feedback_text provided"}), 400
    
    result = sentiment_model(text[:512])[0]
    sentiment = map_sentiment(result["label"])
    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
