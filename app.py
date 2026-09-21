import os
import threading
from flask import Flask, render_template, request, jsonify
from ask_kat import ask_kat, get_model, get_articles_and_index

app = Flask(__name__)

# Preload embedding model and index in background on startup for immediate chat readiness
def preload_resources():
    try:
        print("[App] Pre-loading model and FAISS index...")
        get_model()
        get_articles_and_index("news.json")
        print("[App] KAT is fully ready!")
    except Exception as e:
        print(f"[App Warning] Preload encountered: {e}")

threading.Thread(target=preload_resources, daemon=True).start()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question") or request.form.get("question", "")
    question = question.strip()

    if not question:
        return jsonify({
            "reply": "Please ask a question so I can look through today's news.",
            "sources": []
        }), 400

    try:
        result = ask_kat(question, json_path="news.json")
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "reply": f"An error occurred while consulting Kat: {str(e)}",
            "sources": []
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Kat Chat Web App on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=True)
