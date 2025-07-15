from flask import Flask, request, jsonify
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os

app = Flask(__name__)

# Load sentiment model from HuggingFace Hub
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
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400

    feedbacks = data.get("feedback_text")
    if feedbacks is None:
        return jsonify({"error": "No feedback_text provided"}), 400

    # Accept either single string or list of strings
    if isinstance(feedbacks, str):
        feedbacks = [feedbacks]
    elif not isinstance(feedbacks, list):
        return jsonify({"error": "feedback_text must be a string or list of strings"}), 400

    results = []
    for text in feedbacks:
        result = sentiment_model(text[:512])[0]
        sentiment = map_sentiment(result["label"])
        results.append({"feedback": text, "sentiment": sentiment})

    if len(results) == 1:
        return jsonify(results[0])
    else:
        return jsonify(results)

# Helper: Extract topics using LDA
def extract_topics_lda(feedbacks, n_topics=5, n_top_words=7):
    vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)
    dtm = vectorizer.fit_transform(feedbacks)

    if dtm.shape[0] < n_topics:
        n_topics = max(1, dtm.shape[0])  # Adjust for small datasets

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        topics.append({
            "topic_id": topic_idx,
            "top_words": top_words
        })
    return topics

@app.route("/topics", methods=["POST"])
def topics():
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400

    feedbacks = data.get("feedback_text")
    if not feedbacks or not isinstance(feedbacks, list):
        return jsonify({"error": "feedback_text must be provided as a list of feedback strings"}), 400

    try:
        topics = extract_topics_lda(feedbacks)
        return jsonify({"topics": topics})
    except Exception as e:
        return jsonify({"error": f"Topic extraction failed: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
