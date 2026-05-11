import pandas as pd


def prepare_data(df, min_interactions=10, test_frac=0.2, max_test_items=5):
    df = df[df["rating"] >= 4].copy()

    user_rating = df.groupby("user_id").size().reset_index(name="rating_quantity")
    item_rating = df.groupby("item_id").size().reset_index(name="rating_quantity")

    active_users = user_rating.loc[
        user_rating["rating_quantity"] >= min_interactions, "user_id"
    ].unique()
    active_films = item_rating.loc[
        item_rating["rating_quantity"] >= min_interactions, "item_id"
    ].unique()

    mask = (df["user_id"].isin(active_users)) & (df["item_id"].isin(active_films))
    filtered_df = df[mask].copy()

    filtered_df = filtered_df.sort_values(by=["user_id", "last_watch_dt"])

    def get_test_indices(group):
        n_test = max(1, min(max_test_items, int(len(group) * test_frac)))
        return group.index[-n_test:]

    test_indices = filtered_df.groupby("user_id").apply(get_test_indices).explode()

    test_df = filtered_df.loc[test_indices].copy()
    train_df = filtered_df.drop(test_indices).copy()

    train_users_ids = train_df["user_id"].unique()
    train_item_ids = train_df["item_id"].unique()

    test_df = test_df[test_df["user_id"].isin(train_users_ids)].copy()
    test_df = test_df[test_df["item_id"].isin(train_item_ids)].copy()

    return train_df, test_df
