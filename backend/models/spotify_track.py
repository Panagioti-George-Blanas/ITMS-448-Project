
class SpotifyTrack:
    def __init__(self, name, artist):
        self.name = name
        self.artist = artist

    @classmethod
    def from_spotify_json(cls, json_data):
        track = json_data.get("track", {})
        name = track.get("name")
        artists = track.get("artists", [])
        artist = artists[0].get("name") if artists else None

        return cls(name, artist)

    def __repr__(self):
        return f"<SpotifyTrack {self.name} by {self.artist}>"