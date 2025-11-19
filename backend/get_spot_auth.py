from api.spotify_client import SpotifyClient

def main():
    client = SpotifyClient()
    auth_url = client.get_auth_url()
    print("Go to this URL in your browser and log in:")
    print(auth_url)

if __name__ == "__main__":
    main()