from flask import Flask, request, jsonify
from transformers import pipeline
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)

# Load sentiment model
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def map_sentiment(label):
    label = label.lower()
    if "1 star" in label or "2 star" in label:
        return "negative"
    elif "3 star" in label:
        return "neutral"
    else:
        return "positive"

# Initialize BERTopic model with sentence-transformers embeddings
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
topic_model = BERTopic(embedding_model=sentence_model, language="english")

# Urgent keywords list (can expand)
URGENT_KEYWORDS = ["emergency", "pain", "neglect", "urgent", "infection", "danger", "unsafe", "critical"]

def flag_urgent(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in URGENT_KEYWORDS)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    # Single feedback case
    if isinstance(data, dict) and "feedback_text" in data:
        text = data["feedback_text"]
        if not text:
            return jsonify({"error": "Empty feedback_text"}), 400
        result = sentiment_model(text[:512])[0]
        sentiment = map_sentiment(result["label"])
        return jsonify({"feedback": text, "sentiment": sentiment})

    # Batch feedback case
    elif isinstance(data, dict) and "feedback_list" in data:
        feedback_list = data["feedback_list"]
        if not isinstance(feedback_list, list) or not feedback_list:
            return jsonify({"error": "'feedback_list' must be a non-empty list"}), 400
        results = []
        for text in feedback_list:
            result = sentiment_model(text[:512])[0]
            sentiment = map_sentiment(result["label"])
            results.append({"feedback": text, "sentiment": sentiment})
        return jsonify(results)

    else:
        return jsonify({"error": "Provide 'feedback_text' or 'feedback_list' in JSON body"}), 400

@app.route("/topics", methods=["POST"])
def topics():
    data = request.get_json()
    if not data or "feedback_list" not in data:
        return jsonify({"error": "Provide 'feedback_list' (list) in JSON body"}), 400

    feedback_list = data["feedback_list"]
    if not isinstance(feedback_list, list) or not feedback_list:
        return jsonify({"error": "'feedback_list' must be a non-empty list"}), 400

    # Fit or update BERTopic model on the batch feedback
    topics, _ = topic_model.fit_transform(feedback_list)

    # Aggregate topic frequencies (skip outliers topic -1)
    topic_freq = {}
    for topic in topics:
        if topic == -1:
            continue
        topic_freq[topic] = topic_freq.get(topic, 0) + 1

    # Get topic representations
    topics_repr = {}
    for topic_num in topic_freq:
        topics_repr[topic_num] = topic_model.get_topic(topic_num)  # list of (word, score)

    # Format output: topic number, frequency, top words
    response = []
    for t_num, freq in topic_freq.items():
        top_words = [word for word, _ in topics_repr[t_num]]
        response.append({
            "topic_num": t_num,
            "frequency": freq,
            "top_words": top_words
        })

    # Sort by frequency descending
    response = sorted(response, key=lambda x: x["frequency"], reverse=True)

    return jsonify(response)

@app.route("/urgent", methods=["POST"])
def urgent():
    data = request.get_json()
    if not data or "feedback_list" not in data:
        return jsonify({"error": "Provide 'feedback_list' (list) in JSON body"}), 400

    feedback_list = data["feedback_list"]
    if not isinstance(feedback_list, list) or not feedback_list:
        return jsonify({"error": "'feedback_list' must be a non-empty list"}), 400

    flagged_feedback = [text for text in feedback_list if flag_urgent(text)]

    return jsonify({"urgent_feedback": flagged_feedback, "count": len(flagged_feedback)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
