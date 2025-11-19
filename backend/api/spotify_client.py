from dotenv import load_dotenv
from pathlib import Path
import os
import requests
from urllib.parse import urlencode

# Load .env from repo root
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class SpotifyClient:
    AUTH_URL = "https://accounts.spotify.com/authorize"
    TOKEN_URL = "https://accounts.spotify.com/api/token"
    API_BASE = "https://api.spotify.com/v1"

    def __init__(self):
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        self.redirect_uri = os.getenv("REDIRECT_URI")
        if not self.client_id or not self.client_secret or not self.redirect_uri:
            raise ValueError("SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, or REDIRECT_URI not found in environment variables.")

    def get_auth_url(self):
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": "user-read-recently-played"
        }
        return f"{self.AUTH_URL}?{urlencode(params)}"

    def get_token(self, code):
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        response = requests.post(self.TOKEN_URL, data=data)
        response.raise_for_status()
        return response.json()

    def get_recent_tracks(self, access_token):
        headers = {"Authorization": f"Bearer {access_token}"}
        url = f"{self.API_BASE}/me/player/recently-played?limit=10"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
