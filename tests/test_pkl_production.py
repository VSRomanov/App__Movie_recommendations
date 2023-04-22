"""
This script contains a unit test function to test the recommendation_algorithms module, which includes the knn_train and
corr_train functions. The knn_train function trains a k-Nearest Neighbors model on a movie rating dataset in pivot table
format, while the corr_train function trains a Pearson correlation model on the same dataset.

The script imports the necessary modules and defines fixtures that download a small movie dataset and create sample movie
features DataFrames in both pivot table formats. Then it asserts that the outputs of the knn_train and corr_train
functions have the correct format and pickle files are created.

Specifically, the test function asserts that:
- knn_movie_features_df is a pandas DataFrame
- knn_model is an instance of the NearestNeighbors class
- the file "knn_model.pkl" exists after saving the model
- pc_movie_features_df is a pandas DataFrame

To run the test, execute the test_models_training_and_saving function.
"""
import os
import pickle

import pandas as pd
from sklearn.neighbors import NearestNeighbors

from movie_recommend.constants import PKL_DIR
from movie_recommend.utils.get_databases import get_db
from movie_recommend.utils.recommendation_algorithms import knn_train
from movie_recommend.utils.table_formatting import (
    filter_movies_by_rating_count,
    mean_rating_table,
    merged_table,
    pivot_title_x_users,
    pivot_users_x_title,
)


def test_models_training_and_saving():
    # Define settings
    num_rating_threshold = 10
    dataset_size = "small"

    # Get data
    movies_df, rating_df = get_db(dataset_size)
    rating_with_totalRatingCount = merged_table(movies_df, rating_df)
    total_movie_array = movies_df["title"].values
    total_ratings = mean_rating_table(rating_with_totalRatingCount)
    rating_movie_per_user = filter_movies_by_rating_count(
        rating_with_totalRatingCount, num_rating_threshold
    )

    # # Test KNN model training and saving
    knn_movie_features_df = pivot_title_x_users(rating_movie_per_user)
    knn_model = knn_train(knn_movie_features_df)
    assert isinstance(knn_movie_features_df, pd.DataFrame) == True
    assert isinstance(knn_model, NearestNeighbors) == True

    with open(PKL_DIR / "knn_model_test.pkl", "wb") as f:
        pickle.dump(
            (knn_movie_features_df, knn_model, total_ratings, total_movie_array), f
        )
    assert (PKL_DIR / "knn_model_test.pkl").exists()

    # Delete knn_model_test.pkl file
    os.remove(PKL_DIR / "knn_model_test.pkl")

    # Test Pearson correlation model saving
    pc_movie_features_df = pivot_users_x_title(rating_movie_per_user)
    assert isinstance(pc_movie_features_df, pd.DataFrame) == True
    with open(PKL_DIR / "corr_model_test.pkl", "wb") as f:
        pickle.dump((pc_movie_features_df, total_ratings, total_movie_array), f)
    assert (PKL_DIR / "corr_model_test.pkl").exists()
    # Delete knn_model_test.pkl file
    os.remove(PKL_DIR / "corr_model_test.pkl")
