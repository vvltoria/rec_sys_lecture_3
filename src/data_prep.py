import pandas as pd


def prepare_data(df, min_interactions=10, test_days=30):
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

    # max_date = filtered_df["last_watch_dt"].max()
    # cut_off_date = max_date - test_days

    # train_df = filtered_df[filtered_df["last_watch_dt"] < cut_off_date].copy()
    # test_df = filtered_df[filtered_df["last_watch_dt"] >= cut_off_date].copy()

    filtered_df = filtered_df.sort_values(by=["user_id", "last_watch_dt"])

    test_df = filtered_df.groupby("user_id").tail(5)
    train_df = filtered_df.drop(test_df.index)

    train_users_ids = train_df["user_id"].unique()
    train_item_ids = train_df["item_id"].unique()

    test_df = test_df[test_df["user_id"].isin(train_users_ids)].copy()
    test_df = test_df[test_df["item_id"].isin(train_item_ids)].copy()

    return train_df, test_df
