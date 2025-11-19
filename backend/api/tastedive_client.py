import requests
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from repo root
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("TASTEDIVE_API_KEY")
BASE_URL = "https://tastedive.com/api/similar"

def search_tastedive(query, media_type):
    """
    Query TasteDive for a given search term and type.
    media_type can be: 'music', 'movie', 'show', 'book', 'author', etc.
    """
    # Ensure we use singular 'movie' instead of 'movies'
    if media_type.lower() == "movies":
        media_type = "movie"

    params = {
        "q": query,  # requests handles URL encoding
        "type": media_type,
        "k": API_KEY,
        "limit": 10
    }

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return response.json()
