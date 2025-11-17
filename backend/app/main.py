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

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

