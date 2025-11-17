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
    def word_tokenize(self, text):
        """Tokenizes input text using SpaCy."""
        doc = self.nlp(text)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
        return tokens

    def load_or_train_model(self):
        """Loads existing Word2Vec model or trains new one."""
        if os.path.exists(self.model_path):
            print("Loaded existing Word2Vec model.")
            return Word2Vec.load(self.model_path)
        
        print("Training new Word2Vec model...")
        all_genres = self.movies_df["processed_genres"].tolist() + self.music_df["processed_genres"].tolist()
        tokenized_genres = [self.word_tokenize(genres) for genres in all_genres]

        model = Word2Vec(sentences=tokenized_genres, vector_size=100, window=5, min_count=1, workers=4)
        model.save(self.model_path)
        print("New Word2Vec model trained and saved.")
        return model
    def genre_vector(self, genre_text):
        """Converts genre text into numerical vector using Word2Vec."""
        words = self.word_tokenize(genre_text)
        vectors = [self.word2vec_model.wv[word] for word in words if word in self.word2vec_model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(100)

    def find_movie(self, movie_title: str) -> Union[None, pd.Series]:
        """Finds movie using fuzzy matching."""
        best_match = process.extractOne(movie_title, self.movies_df["title"].tolist(), score_cutoff=80)
        if best_match:
            return self.movies_df[self.movies_df["title"] == best_match[0]].iloc[0]
        return None

    def recommend_music(self, movie_title, num_recommendations=5):
        """Recommends music tracks based on movie genres."""
        idx = self.movies_df[self.movies_df["title"].str.lower() == movie_title.lower()].index
        if len(idx) == 0:
            return "Movie not found!"
        idx = idx[0]
        movie_vector = self.movies_df.iloc[idx]["genre_embeddings"]

        # Compute similarity with music tracks
        similarities = cosine_similarity([movie_vector], np.stack(self.music_df["genre_embeddings"].values))[0]
        top_indices = np.argsort(similarities)[::-1][:num_recommendations]
        
        return self.music_df.iloc[top_indices]["track_names"].tolist()


class BidirectionalRecommendationSystem(GenreRecommendationSystem):
    def __init__(self, movie_data_path="tmdb_5000_movies.csv", 
                 music_data_path="extended_data_by_genres.csv", 
                 model_path="word2vec.model"):
        super().__init__(movie_data_path, music_data_path, model_path)
        
    def find_track(self, track_name: str) -> Union[None, pd.Series]:
        """Finds track using fuzzy matching."""
        best_match = process.extractOne(track_name, self.music_df["track_names"].tolist(), score_cutoff=80)
        if best_match:
            return self.music_df[self.music_df["track_names"] == best_match[0]].iloc[0]
        return None

    def recommend_movies(self, track_name: str, num_recommendations=5):
        """Recommends movies based on music track genres."""
        idx = self.music_df[self.music_df["track_names"].str.lower() == track_name.lower()].index
        if len(idx) == 0:
            return "Track not found!"
        
        idx = idx[0]
        track_vector = self.music_df.iloc[idx]["genre_embeddings"]

        # Compute similarity with movies
        similarities = cosine_similarity([track_vector], np.stack(self.movies_df["genre_embeddings"].values))[0]
        top_indices = np.argsort(similarities)[::-1][:num_recommendations]
        
        return self.movies_df.iloc[top_indices][["title", "overview", "vote_average", "popularity"]].to_dict('records')
def get_spotify_preview(track_name, limit=1):
    """Fetches Spotify track information."""
    results = sp.search(q=track_name, type='track', limit=limit)
    
    songs = []
    for track in results['tracks']['items']:
        song_data = {
            "track_name": track["name"],
            "artist": ", ".join([artist["name"] for artist in track["artists"]]),
            "spotify_url": track["external_urls"]["spotify"],
            "preview_url": track["preview_url"],
            "album_name": track["album"]["name"],
            "album_cover": track["album"]["images"][0]["url"] if track["album"]["images"] else None
        }
        songs.append(song_data)
    
    return pd.DataFrame(songs)
