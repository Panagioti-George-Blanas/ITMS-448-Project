# This code is the start to grab authentication from Spotify to be able to get songs on the website. -Henry

import numpy as np
import pandas as pd
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from fuzzywuzzy import process
import spacy
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from typing import Union, List, Dict

load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def word_tokenize(text):
    """Tokenizes text using SpaCy, removes stop words and punctuation."""
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return tokens

class GenreRecommendationSystem:
    def __init__(self, movie_data_path="tmdb_5000_movies.csv", 
                 music_data_path="extended_data_by_genres.csv", 
                 model_path="word2vec.model"):
        self.movie_data_path = movie_data_path
        self.music_data_path = music_data_path
        self.model_path = model_path
        self.nlp = spacy.load("en_core_web_sm")

        # Load datasets
        self.movies_df = pd.read_csv(self.movie_data_path)
        self.music_df = pd.read_csv(self.music_data_path)

        # Preprocess genres
        self.movies_df["processed_genres"] = self.movies_df["genres"].apply(self.preprocess_genres)
        self.music_df["processed_genres"] = self.music_df["genres"].apply(lambda x: x.lower())

        # Load or train Word2Vec model
        self.word2vec_model = self.load_or_train_model()

        # Generate genre embeddings
        self.movies_df["genre_embeddings"] = self.movies_df["processed_genres"].apply(self.genre_vector)
        self.music_df["genre_embeddings"] = self.music_df["processed_genres"].apply(self.genre_vector)

        # Compute TF-IDF similarity matrix
        self.tfidf = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df["processed_genres"])
        self.genre_sim_matrix = cosine_similarity(self.tfidf_matrix)

    def preprocess_genres(self, genres_str):
        """Extracts genre names from JSON string."""
        try:
            genres_list = [genre["name"].lower() for genre in json.loads(genres_str)]
            return " ".join(genres_list)
        except:
            return ""