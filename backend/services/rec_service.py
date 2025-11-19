import requests
from api.spotify_client import SpotifyClient
from api.tastedive_client import search_tastedive
from api.omdb_client import OMDBClient
from api.tmdb_client import TMDBClient
from models.spotify_track import SpotifyTrack
from models.movie_recs import MovieRecommendation


class RecommendationService:
    def __init__(self):
        self.spotify = SpotifyClient()
        self.omdb = OMDBClient()
        self.tmdb = TMDBClient()

    #get user recent Spotify tracks
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

    #get movie suggestions from TasteDive
    def _get_tastedive_movies(self, track: SpotifyTrack):
        query = f"{track.name} {track.artist}".strip()[:100]
        data = search_tastedive(query, "movie")
        return data.get("similar", {}).get("results", [])

    # lookup movie details from OMDB
    def _lookup_movie_details_omdb(self, title):
        omdb = self.omdb.get_movie_details(title)
        if omdb and omdb.get("Response") == "True":
            return {
                "title": omdb.get("Title"),
                "year": omdb.get("Year"),
                "genre": omdb.get("Genre"),
                "plot": omdb.get("Plot"),
                "poster": omdb.get("Poster")
            }
        return None

    #lookup movie details from TMDB as  fallback
    def _lookup_movie_details_tmdb(self, title):
        search_results = self.tmdb.search_movie_by_title(title)
        if not search_results:
            return None
        movie = search_results[0]
        poster_path = movie.get("poster_path")
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
        return {
            "title": movie.get("title"),
            "year": movie.get("release_date", "")[:4],
            "genre": ", ".join([g["name"] for g in movie.get("genres", [])]) if movie.get("genres") else "",
            "plot": movie.get("overview"),
            "poster": poster_url
        }

    #try OMDB first, then TMDB if OMDB fails
    def _lookup_movie_details(self, title):
        details = self._lookup_movie_details_omdb(title)
        if not details or not details.get("poster") or details.get("poster") == "N/A":
            details = self._lookup_movie_details_tmdb(title)
        return details

   
    def recommend_movies(self, access_token):
        tracks = self._get_recent_tracks(access_token)
        recommendations = []
        td_debug = {} 

        for track in tracks:
            td_results = self._get_tastedive_movies(track)
            td_debug[track.name] = [entry.get("name") for entry in td_results if entry.get("name")]

            if td_results:
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
