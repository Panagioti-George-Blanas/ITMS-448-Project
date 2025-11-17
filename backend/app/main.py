<<<<<<< HEAD
# This code is the start to grab authentication from Spotify to be able to get songs on the website. -Henry
=======
# main.py  ← this becomes your main Flask file
>>>>>>> fef23299b8eebaf73cc3451b89d416714d0af7db

from flask import Flask, redirect, request, session, render_template
from dotenv import load_dotenv
import os

from clients.spotify_client import SpotifyClient
from clients.tmdb_client import TMDBClient
from clients.omdb_client import OMDBClient

load_dotenv()

<<<<<<< HEAD
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

=======
app = Flask(__name__)
app.secret_key = os.getenv("SESSION_SECRET", "fallback_secret")

spotify = SpotifyClient()
tmdb = TMDBClient()
omdb = OMDBClient()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login")
def login():
    # redirect users to Spotify OAuth
    return redirect(spotify.get_auth_url())

@app.route("/callback")
def callback():
    # Spotify sends back a ?code=
    code = request.args.get("code")
    token_info = spotify.get_token(code)

    # store access token in session
    session["access_token"] = token_info["access_token"]
    return redirect("/songs")

@app.route("/songs")
def songs():
    access_token = session.get("access_token")
    recent = spotify.get_recent_tracks(access_token)
    return render_template("results.html", songs=recent)

if __name__ == "__main__":
    app.run(debug=True)
>>>>>>> fef23299b8eebaf73cc3451b89d416714d0af7db
