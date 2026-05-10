import pandas as pd


def get_recommendations(users_ids, model, movies_df, top_k=10):
    predict_df = pd.DataFrame({"user_id": users_ids})
    preds_ids = model.predict(predict_df, top_k=top_k)

    movie_map = movies_df.set_index("item_id")["title"].to_dict()

    result = []
    for user_id, user_preds in zip(users_ids, preds_ids):
        movie_titles = [movie_map.get(itm, f"unknown ID: {itm}") for itm in user_preds]

        result.append({"user_id": user_id, "recommendations": movie_titles})

    return pd.DataFrame(result)


def print_recommendations(recs_df):
    print("Recommendations\n")

    for idx, row in recs_df.iterrows():
        user_id = row["user_id"]
        movies = row["recommendations"]

        print(f"User ID: {user_id}")
        print(f"Movies for you: {len(movies)}")

        for i, movie in enumerate(movies, start=1):
            print(f"  {i}. {movie}")

        print("-" * 50 + "\n")
