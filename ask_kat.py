import json
import os
import sys
import warnings

# Suppress harmless warnings from transformers / huggingface_hub
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


def call_gemini(prompt: str) -> str:
    """
    Sends the prompt to Google's Gemini API using the
    google-generativeai library and model 'gemini-1.5-flash'.
    """
    gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()

    if not gemini_key or gemini_key == "your_gemini_api_key_here":
        return (
            "[ERROR] GEMINI_API_KEY not found or not set in .env file.\n"
            "Please add your Gemini API key to .env:\n"
            "GEMINI_API_KEY=your_actual_key"
        )

    try:
        genai.configure(api_key=gemini_key)
        # Attempt requested model 'gemini-1.5-flash' first
        model_name = "gemini-1.5-flash"
        try:
            model = genai.GenerativeModel(model_name)
            print(f"Querying Gemini API ('{model_name}')...")
            response = model.generate_content(prompt)
            return response.text or "No response received from Gemini."
        except Exception as err:
            # If Google API notes that 1.5-flash is retired/unsupported, fall back to current active model
            if "404" in str(err) or "not found" in str(err).lower() or "not supported" in str(err).lower():
                fallback_model = "gemini-3.6-flash"
                print(f"[Note] '{model_name}' is retired by Google. Falling back to '{fallback_model}'...")
                model = genai.GenerativeModel(fallback_model)
                response = model.generate_content(prompt)
                return response.text or "No response received from Gemini."
            raise err
    except Exception as e:
        return f"[ERROR calling Gemini API]: {e}"


# In-memory cached model and index for fast responses in web apps
_cached_model = None
_cached_index = None
_cached_articles = None
_cached_mtime = None


def get_model():
    """Returns the cached SentenceTransformer model, loading it if not yet initialized."""
    global _cached_model
    if _cached_model is None:
        _cached_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _cached_model


def get_articles_and_index(json_path: str = "news.json"):
    """
    Loads news articles from json_path and returns (articles, faiss_index).
    Caches the FAISS index in memory unless news.json has been modified.
    """
    global _cached_index, _cached_articles, _cached_mtime
    if not os.path.exists(json_path):
        return None, None

    mtime = os.path.getmtime(json_path)
    if _cached_index is not None and _cached_articles is not None and _cached_mtime == mtime:
        return _cached_articles, _cached_index

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            articles = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read '{json_path}': {e}")
        return None, None

    if not articles:
        return [], None

    model = get_model()
    article_texts = [
        f"{article.get('title', '')}. {article.get('description', '')}".strip()
        for article in articles
    ]

    embeddings = model.encode(article_texts, convert_to_numpy=True, show_progress_bar=False)
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings.astype("float32"))

    _cached_articles = articles
    _cached_index = index
    _cached_mtime = mtime
    return _cached_articles, _cached_index


def ask_kat(question: str, json_path: str = "news.json") -> dict:
    """
    Core Kat Q&A logic:
    Retrieves top matching articles from FAISS index, builds the prompt,
    calls Gemini API, and returns a dictionary with Kat's reply and source articles.
    """
    load_dotenv()
    question = (question or "").strip()
    if not question:
        return {"answer": "Please ask a non-empty question.", "reply": "Please ask a non-empty question.", "sources": []}

    articles, index = get_articles_and_index(json_path)
    if articles is None:
        msg = f"[ERROR] '{json_path}' not found. Please run fetch_news.py first."
        return {
            "answer": msg,
            "reply": msg,
            "sources": []
        }
    if not articles or index is None:
        msg = "No articles found in news.json. Please run fetch_news.py to fetch fresh headlines."
        return {
            "answer": msg,
            "reply": msg,
            "sources": []
        }

    model = get_model()
    query_embedding = model.encode([question], convert_to_numpy=True, show_progress_bar=False)

    top_k = min(3, len(articles))
    distances, indices = index.search(query_embedding.astype("float32"), k=top_k)

    formatted_articles = []
    sources = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx == -1 or idx >= len(articles):
            continue
        art = articles[idx]
        title = art.get("title") or "No Title Available"
        desc = art.get("description") or "No Description Available"
        url = art.get("url") or ""

        sources.append({
            "rank": rank,
            "title": title,
            "description": desc,
            "url": url,
            "distance": float(dist)
        })

        formatted_articles.append(
            f"Article {rank}:\n"
            f"- Title: {title}\n"
            f"- Description: {desc}\n"
            f"- URL: {url}"
        )

    articles_context = "\n\n".join(formatted_articles)

    prompt = (
        f"You are Kat, a chatbot that answers questions using today's news.\n\n"
        f"Given these articles:\n"
        f"{articles_context}\n\n"
        f"answer the question: {question}\n\n"
        f"Only use information from the articles provided. If the articles don't contain the answer, say so."
    )

    reply = call_gemini(prompt)
    return {
        "answer": reply,
        "reply": reply,
        "sources": sources
    }


def main():
    load_dotenv()
    json_path = "news.json"

    if not os.path.exists(json_path):
        print(f"[ERROR] '{json_path}' not found. Please run fetch_news.py first to generate articles.")
        return

    try:
        question = input("Ask KAT a question: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        return

    if not question:
        print("Please provide a non-empty question.")
        return

    print("Retrieving answer from Kat...")
    result = ask_kat(question, json_path=json_path)

    print("\nRetrieved Sources:")
    for src in result.get("sources", []):
        print(f"{src['rank']}. {src['title']} (Distance: {src['distance']:.4f})")
    print("-" * 80)

    print("\nKat's Reply:")
    print("=" * 80)
    print(result.get("reply", ""))
    print("=" * 80)


if __name__ == "__main__":
    main()
