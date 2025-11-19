import requests
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from repo root
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class OMDBClient:
    BASE_URL = "https://www.omdbapi.com/"

    def __init__(self):
        self.api_key = os.getenv("OMDB_API_KEY")
        if not self.api_key:
            raise ValueError("OMDB_API_KEY not found in .env")

    def get_movie_details(self, title):
        params = {
            "t": title,
            "plot": "full",
            "apikey": self.api_key
        }
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        return response.json()

    def get_by_id(self, imdb_id):
        params = {
            "i": imdb_id,
            "plot": "full",
            "apikey": self.api_key
        }
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        return response.json()
