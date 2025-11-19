
class MovieRecommendation:
    def __init__(self, based_on_song, based_on_artist, movie_title, details):
        self.based_on_song = based_on_song
        self.based_on_artist = based_on_artist
        self.movie_title = movie_title
        self.details = details

    def __repr__(self):
        return f"<MovieRecommendation {self.movie_title} (based on {self.based_on_song})>"