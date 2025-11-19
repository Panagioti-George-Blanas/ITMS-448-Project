from flask import Flask, redirect, request, render_template
from api.spotify_client import SpotifyClient
from services.rec_service import RecommendationService

# Create the Flask app instance
app = Flask(__name__)
 
#clients
spotify_client = SpotifyClient()
service = RecommendationService()


@app.route("/")
def index():
    return '<h2>Welcome!</h2><a href="/login">Login with Spotify</a>'


## redirect user to spotify OAuth URL
@app.route("/login")
def login():
    return redirect(spotify_client.get_auth_url())


@app.route("/callback")
def callback():
    # Spotify redirects after login
    code = request.args.get("code")
    token_data = spotify_client.get_token(code)
    access_token = token_data.get("access_token")

    # get the user's recent tracks
    tracks = service._get_recent_tracks(access_token)

    # get movie recommendations & TasteDive info
    recommendations, td_debug = service.recommend_movies(access_token)

    # pass recent tracks, TasteDive results, and final recommendations
    return render_template(
        "recommendations.html",
        tracks=tracks,
        td_debug=td_debug,
        recommendations=recommendations
    )


# start Flask server
if __name__ == "__main__":
    app.run(port=5000, debug=True)
