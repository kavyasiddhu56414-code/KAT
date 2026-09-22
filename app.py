import os
import threading
from flask import Flask, render_template, request, jsonify
from ask_kat import ask_kat, get_model, get_articles_and_index

app = Flask(__name__)

# Preload embedding model and FAISS index in the background on startup
def preload_resources():
    try:
        print("[App] Pre-loading model and FAISS index...")
        get_model()
        get_articles_and_index("news.json")
        print("[App] Kat is ready!")
    except Exception as e:
        print(f"[App Warning] Preload encountered: {e}")

threading.Thread(target=preload_resources, daemon=True).start()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
@app.route("/api/ask", methods=["POST"])
def ask():
    # Parse JSON payload
    data = request.get_json(silent=True, force=True) or {}
    question = data.get("question") if isinstance(data, dict) else None

    # Fallback to form data if not sent as JSON
    if not question and request.form:
        question = request.form.get("question")

    if not question or not isinstance(question, str) or not question.strip():
        return jsonify({
            "error": "Question is required.",
            "answer": "Please provide a question so I can search today's news.",
            "sources": []
        }), 400

    try:
        result = ask_kat(question.strip(), json_path="news.json")
        answer = result.get("answer") or result.get("reply", "")
        raw_sources = result.get("sources", [])

        # Return title + url of matched articles as requested
        sources = [
            {
                "title": src.get("title", ""),
                "url": src.get("url", "")
            }
            for src in raw_sources
        ]

        return jsonify({
            "answer": answer,
            "sources": sources
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "answer": f"An error occurred while consulting Kat: {str(e)}",
            "sources": []
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Kat web app on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
