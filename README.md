# Sentiment Analysis API

This is a Flask API serving the `nlptown/bert-base-multilingual-uncased-sentiment` model from HuggingFace.

## Usage

Send a POST request to `/predict` with JSON payload:

```json
{
  "feedback_text": "The service was excellent!"
}
