from services.rec_service import RecommendationService

def main():
    token = input("Enter your Spotify access token: ")

    service = RecommendationService()
    recommendations = service.recommend_movies(token)

    
    for rec in recommendations:
        print("\n Movie Recommendation:")
        print("Based on Song:   ", rec.based_on_song)
        print("Artist:          ", rec.based_on_artist)
        print("Movie:           ", rec.movie_title)

        # TMDB uses "overview", OMDB uses "Plot"
        summary = rec.details.get("overview") or rec.details.get("Plot")
        print("Summary:         ", summary)

if __name__ == "__main__":
    main()