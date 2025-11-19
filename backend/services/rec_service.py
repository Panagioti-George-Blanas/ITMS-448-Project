from api.spotify_client import SpotifyClient
from api.tastedive_client import search_tastedive
from api.omdb_client import OMDBClient
from api.tmdb_client import TMDBClient

from models.spotify_track import SpotifyTrack
from models.movie_recs import MovieRecommendation

import urllib.parse  # for URL-safe encoding


class RecommendationService:
    """
    Main logic used to:
    1. Get user's recently played Spotify tracks
    2. Ask TasteDive for movie recommendations based on those tracks
    3. Use TMDB/OMDB to fetch real movie details
    """

    def __init__(self):
        self.spotify = SpotifyClient()
        self.tmdb = TMDBClient()
        self.omdb = OMDBClient()

    # -----------------------------
    # STEP 1: Recent Spotify tracks
    # -----------------------------
    def _get_recent_tracks(self, access_token):
        raw = self.spotify.get_recent_tracks(access_token)
        items = raw.get("items", [])

        tracks = []
        for item in items:
            track = SpotifyTrack.from_spotify_json(item)
            if track.name and track.artist:
                tracks.append(track)

        return tracks

    # -----------------------------
    # STEP 2: TasteDive movie matches
    # -----------------------------
    def _get_tastedive_movies(self, track: SpotifyTrack):
        """
        Query TasteDive for movie recommendations based on a SpotifyTrack.
        """
        query = f"{track.name} {track.artist}".strip()[:100]  # optional length limit
        # DO NOT urlencode manually
        data = search_tastedive(query, "movie")
        print("QUERY:", query)
        print("TasteDive Response:", data)
        return data.get("Similar", {}).get("Results", [])

    # -----------------------------
    # STEP 3: Lookup movies in TMDB → then fallback to OMDB
    # -----------------------------
    def _lookup_movie_details(self, movie_title):
        # First try TMDB
        tmdb_results = self.tmdb.search_movie_by_title(movie_title)
        if tmdb_results:
            return tmdb_results[0]   # Best match

        # Try OMDB second
        omdb_data = self.omdb.get_movie_details(movie_title)
        if omdb_data and omdb_data.get("Response") == "True":
            return omdb_data

        return None  # No match anywhere

    # -----------------------------
    # PUBLIC FUNCTION
    # -----------------------------
    def recommend_movies(self, access_token):
        tracks = self._get_recent_tracks(access_token)
        recommendations = []
        print("TRACKS FOUND:", [ (t.name, t.artist) for t in tracks ])
        for track in tracks:
            td_results = self._get_tastedive_movies(track)
            for entry in td_results:
                movie_title = entry.get("Name")
                print("TasteDive entry:", entry)
                print("Movie title:", movie_title)

                if not movie_title:
                    continue

                movie_data = self._lookup_movie_details(movie_title)
                if not movie_data:
                    continue

                rec = MovieRecommendation(
                    based_on_song=track.name,
                    based_on_artist=track.artist,
                    movie_title=movie_title,
                    details=movie_data
                )

                recommendations.append(rec)

        return recommendations
