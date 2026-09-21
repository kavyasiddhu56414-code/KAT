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


def main():
    # 1. Load environment variables from .env
    load_dotenv()

    json_path = "news.json"

    # 2. Check if news.json exists
    if not os.path.exists(json_path):
        print(f"[ERROR] '{json_path}' not found. Please run fetch_news.py first to generate articles.")
        return

    # 3. Load news articles
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            articles = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read '{json_path}': {e}")
        return

    if not articles:
        print("No articles found in news.json.")
        return

    # 4. Load SentenceTransformer embedding model
    print("Loading SentenceTransformer model ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 5. Prepare text: combine title and description for each article
    article_texts = [
        f"{article.get('title', '')}. {article.get('description', '')}".strip()
        for article in articles
    ]

    print(f"Embedding {len(article_texts)} articles and building FAISS index...")
    # 6. Generate embeddings
    embeddings = model.encode(article_texts, convert_to_numpy=True, show_progress_bar=False)

    # 7. Build FAISS index
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings.astype("float32"))
    print(f"FAISS index built successfully with {index.ntotal} vectors.\n")

    # 8. Prompt user for question
    try:
        question = input("Ask KAT a question: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        return

    if not question:
        print("Please provide a non-empty question.")
        return

    # 9. Embed the user's question
    query_embedding = model.encode([question], convert_to_numpy=True, show_progress_bar=False)

    # 10. Search FAISS index for top 3 closest articles
    top_k = min(3, len(articles))
    distances, indices = index.search(query_embedding.astype("float32"), k=top_k)

    # 11. Format top articles for the Gemini prompt
    formatted_articles = []
    print(f"\nRetrieved Top {top_k} Matching Articles from FAISS:\n")
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx == -1 or idx >= len(articles):
            continue
        art = articles[idx]
        title = art.get("title") or "No Title Available"
        desc = art.get("description") or "No Description Available"
        url = art.get("url") or ""

        print(f"{rank}. {title} (Distance: {dist:.4f})")

        formatted_articles.append(
            f"Article {rank}:\n"
            f"- Title: {title}\n"
            f"- Description: {desc}\n"
            f"- URL: {url}"
        )
    print("-" * 80)

    articles_context = "\n\n".join(formatted_articles)

    # 12. Construct the prompt for Kat as requested
    prompt = (
        f"You are Kat, a chatbot that answers questions using today's news.\n\n"
        f"Given these articles:\n"
        f"{articles_context}\n\n"
        f"answer the question: {question}\n\n"
        f"Only use information from the articles provided. If the articles don't contain the answer, say so."
    )

    # 13. Send to Gemini and print Kat's reply
    reply = call_gemini(prompt)
    print("\nKat's Reply:")
    print("=" * 80)
    print(reply)
    print("=" * 80)


if __name__ == "__main__":
    main()
