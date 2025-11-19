import requests
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from repo root
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class TMDBClient:
    BASE_URL = "https://api.themoviedb.org/3"

    def __init__(self):
        self.api_key = os.getenv("TMDB_API_KEY")
        if not self.api_key:
            raise ValueError("TMDB_API_KEY not found in environment variables.")

    def _get(self, endpoint, params=None):
        if params is None:
            params = {}
        params["api_key"] = self.api_key
        url = f"{self.BASE_URL}{endpoint}"
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_genre_id(self, genre_name):
        data = self._get("/genre/movie/list")
        genres = data.get("genres", [])
        for g in genres:
            if genre_name.lower() in g["name"].lower():
                return g["id"]
        return None

    def search_movies_by_genre(self, genre_name, limit=10):
        genre_id = self.get_genre_id(genre_name)
        if not genre_id:
            return []

        data = self._get("/discover/movie", {
            "with_genres": genre_id,
            "sort_by": "popularity.desc"
        })
        return data.get("results", [])[:limit]

    def search_movie_by_title(self, title):
        data = self._get("/search/movie", {"query": title})
        return data.get("results", [])
