import json
import os
import sys
import requests
from dotenv import load_dotenv

def fetch_top_headlines(country: str = "us", page_size: int = 5, output_file: str = "news.json"):
    """
    Loads API key from .env, calls NewsAPI top-headlines endpoint,
    prints the top articles, and saves them to a JSON file.
    """
    # 1. Load environment variables from .env file
    load_dotenv()

    # 2. Retrieve the API key
    api_key = os.getenv("NEWS_API_KEY")
    if api_key:
        api_key = api_key.strip()

    if not api_key or api_key == "your_news_api_key_here":
        print("[ERROR] NEWS_API_KEY not found or not set.")
        print("Please add your NewsAPI key to your .env file:")
        print("NEWS_API_KEY=your_actual_api_key_here")
        return

    # 3. NewsAPI top-headlines endpoint URL
    url = "https://newsapi.org/v2/top-headlines"

    # 4. Define query parameters for the request
    params = {
        "country": country,
        "pageSize": page_size,
        "apiKey": api_key,
    }

    print(f"Fetching top {page_size} headlines for country: '{country.upper()}'...\n")

    try:
        # 5. Send GET request to NewsAPI
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raises HTTPError for 4xx/5xx responses
        data = response.json()

        # 6. Check the API response status
        if data.get("status") != "ok":
            print(f"[API ERROR] {data.get('message', 'Failed to fetch news.')}")
            return

        articles = data.get("articles", [])
        if not articles:
            print("No articles found for the given criteria.")
            return

        saved_articles = []

        # 7. Print articles to terminal and extract required fields
        for index, article in enumerate(articles[:page_size], start=1):
            title = article.get("title") or "No Title Available"
            description = article.get("description") or "No Description Available"
            article_url = article.get("url") or ""
            published_at = article.get("publishedAt") or ""

            saved_articles.append({
                "title": title,
                "description": description,
                "url": article_url,
                "publishedAt": published_at,
            })

            print(f"{index}. Title: {title}")
            print(f"   Description: {description}")
            print("-" * 80)

        # 8. Save articles to news.json (overwriting each time)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(saved_articles, f, indent=4, ensure_ascii=False)

        print(f"\n[INFO] Successfully saved {len(saved_articles)} articles to '{output_file}'.")

    except requests.exceptions.RequestException as e:
        print(f"[NETWORK ERROR] Failed to connect to NewsAPI: {e}")


if __name__ == "__main__":
    # You can change the country to "in" or "us", or pass it as an argument
    # e.g.: python fetch_news.py in
    selected_country = sys.argv[1].lower() if len(sys.argv) > 1 else "us"
    fetch_top_headlines(country=selected_country, page_size=5)
