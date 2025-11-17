# clients/spotify_client.py
import os
import requests
from urllib.parse import urlencode

class SpotifyClient:
    AUTH_URL = "https://accounts.spotify.com/authorize"
    TOKEN_URL = "https://accounts.spotify.com/api/token"
    API_BASE = "https://api.spotify.com/v1"

    def __init__(self):
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.redirect_uri = os.getenv("REDIRECT_URI")

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
        return requests.post(self.TOKEN_URL, data=data).json()

    def get_recent_tracks(self, access_token):
        headers = {"Authorization": f"Bearer {access_token}"}
        url = f"{self.API_BASE}/me/player/recently-played?limit=10"
        return requests.get(url, headers=headers).json()
