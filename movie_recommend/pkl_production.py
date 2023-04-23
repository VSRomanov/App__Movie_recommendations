"""
The script trains two different models on a movie rating dataset: a k-Nearest Neighbors model and a Pearson
correlation model.

For the k-Nearest Neighbors model, it filters out unpopular movies, trains the model, and saves
the model and a pivot table of movie features to a pickle file. For the Pearson correlation model, it just
filters out unpopular movies and saves a pivot table of movie features to a pickle file. The script imports utility
functions from the movie_recommend.utils module to retrieve and format the movie rating data, train the models,
and save the results to a file. The minimum number of ratings per movie, the size of the dataset, and the type of
model can be configured by modifying the variables at the top of the script.
"""

import pickle

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

if __name__ == "__main__":
    ## Settings: minimum number of ratings per movie, small or full dataset, model type.
    num_rating_threshold = 10
    # num_rating_threshold = 250

    dataset_size = "small"
    # dataset_size = "full"

    model_types = ("k-Nearest Neighbors", "Pearson correlation")

    ## Import
    movies_df, rating_df = get_db(dataset_size)
    total_movie_array = movies_df["title"].values

    ## Information about tables:
    print("\nThe total number of movies in the database:", len(total_movie_array))
    print("The total number of ratings in the database:", len(rating_df.index))

    rating_with_totalRatingCount = merged_table(movies_df, rating_df)

    total_ratings = mean_rating_table(rating_with_totalRatingCount)

    rating_movie_per_user = filter_movies_by_rating_count(
        rating_with_totalRatingCount, num_rating_threshold
    )

    ## Create the output folder if not exist
    PKL_DIR.mkdir(parents=True, exist_ok=True)

    for model_type in model_types:
        print("\nIn preparation:", model_type)

        if model_type == "k-Nearest Neighbors":
            ## Pivot table to the "Title vs Ratings" format
            movie_features_df = pivot_title_x_users(rating_movie_per_user)

            print(
                f"Number of movies with more than {num_rating_threshold} ratings: {len(movie_features_df.index)}"
            )

            ## Train the model
            knn_model = knn_train(movie_features_df)

            pkl_file_name = (
                "knn_model.pkl" if dataset_size == "full" else "knn_model_small.pkl"
            )

            ## Save the results
            with open(PKL_DIR / pkl_file_name, "wb") as f:
                pickle.dump(
                    (movie_features_df, knn_model, total_ratings, total_movie_array), f
                )

        elif model_type == "Pearson correlation":
            ## Pivot table to the "Ratings vs Title" format
            movie_features_df = pivot_users_x_title(rating_movie_per_user)

            print(
                f"Number of movies with more than {num_rating_threshold} ratings: {len(movie_features_df.columns)}"
            )

            pkl_file_name = (
                "corr_model.pkl" if dataset_size == "full" else "corr_model_small.pkl"
            )

            ## Save the results
            with open(PKL_DIR / pkl_file_name, "wb") as f:
                pickle.dump((movie_features_df, total_ratings, total_movie_array), f)

    print("\nDone! Done!")
