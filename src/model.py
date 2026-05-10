import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder


class ImplicitModel:
    def __init__(self, model):
        self.model = model
        self.trained = False

    def fit(self, train_df):
        self.item_encoder = LabelEncoder()
        self.user_encoder = LabelEncoder()
        self.item_encoder.fit(train_df["item_id"])
        self.user_encoder.fit(train_df["user_id"])

        self.train_ratings = self.create_matrix(train_df, ["item_id", "user_id"])
        self.model.fit(self.train_ratings)
        self.trained = True

    def predict(self, test_df, top_k=10):
        if not self.trained:
            raise ValueError("Model is not fitted")

        users_to_predict = test_df["user_id"].unique()
        encoded_users_to_predict = self.user_encoder.transform(users_to_predict)

        recs = self.model.recommend(
            encoded_users_to_predict,
            self.train_ratings[encoded_users_to_predict],
            N=top_k,
            filter_already_liked_items=True,
        )[0]

        user_recs = [self.item_encoder.inverse_transform(i) for i in recs]

        return user_recs

    def create_matrix(self, df, ids_list):
        encoded_item_ids = self.item_encoder.transform(df[ids_list[0]])
        encoded_user_ids = self.user_encoder.transform(df[ids_list[1]])

        n_items = len(self.item_encoder.classes_)
        n_users = len(self.user_encoder.classes_)

        matrix_shape = (n_users, n_items)

        sparse_matrix = csr_matrix(
            (np.ones(len(encoded_user_ids)), (encoded_user_ids, encoded_item_ids)),
            shape=matrix_shape,
            dtype=np.float32,
        )

        return sparse_matrix
