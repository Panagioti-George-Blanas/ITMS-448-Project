from flask import Flask, request
import webbrowser
from api.spotify_client import SpotifyClient

app = Flask(__name__)
client = SpotifyClient()
access_token = None

# ---------------------------
# Step 1: Start login
# ---------------------------
def start_auth():
    auth_url = client.get_auth_url()
    print("Opening Spotify login in your browser...")
    webbrowser.open(auth_url)

# ---------------------------
# Step 2: Handle redirect
# ---------------------------
@app.route("/callback")
def callback():
    global access_token
    code = request.args.get("code")
    if not code:
        return "Error: no code found in URL!"
    token_data = client.get_token(code)
    access_token = token_data.get("access_token")
    if access_token:
        return f"<h1>Success!</h1><p>Your token is printed in the console.</p>"
    return "Error: could not get access token"

# ---------------------------
# Step 3: Run Flask server
# ---------------------------
if __name__ == "__main__":
    start_auth()
    # Flask runs on 127.0.0.1:5000, same as your redirect URI
    app.run(port=5000)
    print("\n🎉 Your access token:")
    print(access_token)
