import pandas as pd
from implicit.als import AlternatingLeastSquares

from src.data_prep import prepare_data
from src.model import ImplicitModel
from src.metrics import evaluate_model
from src.inference import get_recommendations, print_recommendations


def main():
    print("1. Loading data")

    ratings_cols = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(
        "data/ml-1m/ratings.dat", sep="::", engine="python", names=ratings_cols
    )

    movies_cols = ["item_id", "title", "genres"]
    movies_df = pd.read_csv(
        "data/ml-1m/movies.dat",
        sep="::",
        engine="python",
        names=movies_cols,
        encoding="latin-1",
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    min_date = df["timestamp"].min()
    df["last_watch_dt"] = (df["timestamp"] - min_date).dt.days
    df = df.drop(columns="timestamp")

    print("2. Filtering data")
    train_df, test_df = prepare_data(df, min_interactions=10, test_days=30)

    print("3. Fitting model")
    als_model = AlternatingLeastSquares(
        factors=64, regularization=0.01, iterations=15, random_state=42
    )
    recsys_model = ImplicitModel(als_model)
    recsys_model.fit(train_df)

    print("4. Evaluate model")
    preds = recsys_model.predict(test_df, top_k=20)
    preds_df = pd.DataFrame(
        {"user_id": test_df["user_id"].unique(), "ials_preds": preds}
    )

    test_df_grouped = (
        test_df.groupby("user_id")
        .apply(lambda x: list(x["item_id"]))
        .reset_index(name="true_test_interactions")
    )
    test_df_grouped = test_df_grouped.merge(preds_df, on="user_id", how="left")

    ials_ndcg, ials_recall = evaluate_model(
        test_df_grouped, "ials_preds", "true_test_interactions", top_k=20
    )
    print(f"   Metrics on Test data:")
    print(f"   NDCG@20: {ials_ndcg:.4f}")
    print(f"   Recall@20: {ials_recall:.4f}\n")

    print("5. Users' Recomendations Example")
    sample_users = [1, 10, 100]
    final_recs = get_recommendations(sample_users, recsys_model, movies_df, top_k=5)
    print_recommendations(final_recs)


if __name__ == "__main__":
    main()
