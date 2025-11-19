import requests
from api.spotify_client import SpotifyClient
from api.tastedive_client import search_tastedive
from api.omdb_client import OMDBClient
from models.spotify_track import SpotifyTrack
from models.movie_recs import MovieRecommendation


class RecommendationService:
    def __init__(self):
        self.spotify = SpotifyClient()
        self.omdb = OMDBClient()

    def _get_recent_tracks(self, access_token, limit=10):
        raw = self.spotify.get_recent_tracks(access_token, limit=limit)
        items = raw.get("items", [])
        tracks = []
        for item in items:
            track_data = item.get("track", {})
            track = SpotifyTrack.from_spotify_json(track_data)
            if track.name and track.artist:
                tracks.append(track)
        return tracks

    def _get_tastedive_movies(self, track: SpotifyTrack):
        query = f"{track.name} {track.artist}".strip()[:100]
        data = search_tastedive(query, "movie")
        # TasteDive response uses lowercase keys
        return data.get("similar", {}).get("results", [])

    def _lookup_movie_details(self, movie_title):
        if not movie_title:
            return None
        omdb_data = self.omdb.get_movie_details(movie_title)
        if omdb_data and omdb_data.get("Response") == "True":
            # Only keep basic info
            return {
                "title": omdb_data.get("Title"),
                "year": omdb_data.get("Year"),
                "genre": omdb_data.get("Genre"),
                "plot": omdb_data.get("Plot"),
                "poster": omdb_data.get("Poster")
            }
        return None

    def recommend_movies(self, access_token):
        tracks = self._get_recent_tracks(access_token)
        recommendations = []
        td_debug = {}  # Store all TasteDive results per track

        for track in tracks:
            td_results = self._get_tastedive_movies(track)
            td_debug[track.name] = [entry.get("name") for entry in td_results if entry.get("name")]

            if td_results:
                # Take the first movie from TasteDive results
                first_movie = td_results[0].get("name")
                if first_movie:
                    movie_details = self._lookup_movie_details(first_movie)

                    rec = MovieRecommendation(
                        based_on_song=track.name,
                        based_on_artist=track.artist,
                        movie_title=first_movie,
                        details=movie_details
                    )
                    recommendations.append(rec)

        return recommendations, td_debug
