from flask import Flask, request, jsonify
from app.models.model_manager import load_model
from app.utils.similarity import cosine_sim
from app.utils.tokenizer import uz_tokenize
import numpy as np

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Embedding API running"}), 200


# -------- SINGLE EMBEDDING ----------
@app.route("/embed", methods=["POST"])
def embed():

    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "field 'text' is required"}), 400

    text = data["text"].strip()
    model_type = data.get("model", "fasttext-en")

    try:
        model = load_model(model_type)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if model_type.startswith("fasttext"):
        vec = model.get_word_vector(text).tolist()
    else:
        vec = model.get_sentence_vector(text)

    return jsonify({
        "model": model_type,
        "text": text,
        "dim": len(vec),
        "vector": vec
    }), 200


# -------- TOKENIZED EMBEDDING ----------
@app.route("/embed-tokenized", methods=["POST"])
def embed_tokenized():

    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "field 'text' required"}), 400

    text = data["text"].strip()
    model_type = data.get("model", "fasttext-uz")

    model = load_model(model_type)

    tokens = uz_tokenize(text)

    vectors = [model.get_word_vector(t) for t in tokens]
    avg_vector = np.mean(vectors, axis=0).tolist()

    return jsonify({
        "model": model_type,
        "tokens": tokens,
        "dim": len(avg_vector),
        "vector": avg_vector
    }), 200


# -------- TEXT SIMILARITY ----------
@app.route("/compare", methods=["POST"])
def compare():

    data = request.get_json()

    if not data or "text1" not in data or "text2" not in data:
        return jsonify({"error": "text1 and text2 fields required"}), 400

    model_type = data.get("model", "fasttext-en")
    text1, text2 = data["text1"], data["text2"]

    model = load_model(model_type)

    if model_type.startswith("fasttext"):
        v1 = model.get_word_vector(text1)
        v2 = model.get_word_vector(text2)
    else:
        v1 = model.get_sentence_vector(text1)
        v2 = model.get_sentence_vector(text2)

    sim = cosine_sim(v1, v2)

    return jsonify({
        "model": model_type,
        "text1": text1,
        "text2": text2,
        "similarity": sim
    }), 200


# -------- BATCH EMBEDDING ----------
@app.route("/batch/embed", methods=["POST"])
def batch_embed():

    data = request.get_json()

    if not data or "texts" not in data:
        return jsonify({"error": "field 'texts' required"}), 400

    texts = data["texts"]
    model_type = data.get("model", "fasttext-en")

    model = load_model(model_type)

    results = []

    for text in texts:
        if model_type.startswith("fasttext"):
            vec = model.get_word_vector(text).tolist()
        else:
            vec = model.get_sentence_vector(text)

        results.append({"text": text, "vector": vec})

    return jsonify({
        "model": model_type,
        "count": len(texts),
        "results": results
    })


if __name__ == "__main__":
    # debug=False required to prevent double model loading
    app.run(port=5000, debug=False)
