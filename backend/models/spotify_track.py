# models/spotify_track.py

class SpotifyTrack:
    def __init__(self, name, artist, album_image=None, spotify_url=None):
        self.name = name
        self.artist = artist
        self.album_image = album_image
        self.spotify_url = spotify_url


#  convert Spotify API track JSON into a SpotifyTrack object.
    @classmethod
    def from_spotify_json(cls, data):

        if not data:
            return None

				# track name
        name = data.get("name")
        
        # Get artist name
        artists = data.get("artists", [])
        artist = artists[0]["name"] if artists else None

        if not name or not artist:
            return None

				# construct and return SpotifyTrack instance
        return cls(name=name, artist=artist, album_image=album_image, spotify_url=spotify_url)
