"""
This script trains two different models on a movie rating dataset: a k-Nearest Neighbors model and a Pearson
correlation model.

For the k-Nearest Neighbors model, it filters out unpopular movies, trains the model, and saves the model
along with a pivot table of movie features to a pickle file. For the Pearson correlation model, it filters out
unpopular movies and saves a pivot table of movie features to a pickle file.

The script imports utility functions from the `movie_recommend.utils` module to download, retrieve and format
the movie rating data, train the models, and save the results to a file. The minimum number of ratings per
movie, the size of the dataset, and the type of model can be configured by modifying the variables at the top
of the script.
"""

import os
import pickle
import logging
from typing import Tuple

import movie_recommend.constants as c
from movie_recommend.utils.get_databases import get_db
from movie_recommend.utils.recommendation_algorithms import knn_train
from movie_recommend.utils.table_formatting import (
    filter_movies_by_rating_count,
    mean_rating_table,
    merged_table,
    pivot_ratings
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def save_to_pickle(data: Tuple, filename: str) -> None:
    """Save data to a pickle file."""
    with open(os.path.join(c.PKL_DIR, filename), "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    # Settings: minimum number of ratings per movie, small or full dataset, model type.
    #dataset_size = "small"; rating_threshold = 10
    dataset_size = "full"; rating_threshold = 500
    model_types = ("k-Nearest Neighbors", "Pearson correlation")
    logging.info("Starting script with dataset size '%s' and rating threshold %d", dataset_size, rating_threshold)

    # Import
    try:
        movies_df, rating_df = get_db(dataset_size)
    except Exception as e:
        logging.error("Error loading databases: %s", e)
        exit(1)
    total_movie_array = movies_df[c.TITLE].values
    logging.info("The total number of movies in the database: %d", len(total_movie_array))
    logging.info("The total number of ratings in the database: %d", len(rating_df.index))


    try:
        movie_rating_df = merged_table(movies_df, rating_df)
        all_ratings = mean_rating_table(movie_rating_df)
        rating_movie_per_user = filter_movies_by_rating_count(movie_rating_df, rating_threshold)
    except Exception as e:
        logging.error("Error processing data: %s", e)
        exit(1)

    # Create the output folder if it doesn't exist
    os.makedirs(c.PKL_DIR, exist_ok=True)

    for model_type in model_types:
        logging.info("____________________________________")
        logging.info("In preparation: %s", model_type)

        try:
            if model_type == "k-Nearest Neighbors":
                # Pivot table to the "Title vs Ratings" format
                features_df_knn = pivot_ratings(rating_movie_per_user, index=c.TITLE, columns=c.USER_ID).fillna(0)

                logging.info("Number of movies with more than %d ratings: %d", rating_threshold, len(features_df_knn.index))

                # Train the model
                knn_model = knn_train(features_df_knn)

                # Save the results
                pkl_file_name = ("knn_model_full.pkl" if dataset_size == "full" else "knn_model_small.pkl")
                save_to_pickle((features_df_knn, knn_model, all_ratings, total_movie_array), pkl_file_name)

            elif model_type == "Pearson correlation":
                # Pivot table to the "Ratings vs Title" format
                features_df_corr = pivot_ratings(rating_movie_per_user, index=c.USER_ID, columns=c.TITLE)

                # print(f"Number of movies with more than {rating_threshold} ratings: {len(features_df_corr.columns)}")
                logging.info("Number of movies with more than %d ratings: %d", rating_threshold, len(features_df_corr.columns))

                # Save the results
                pkl_file_name = ("corr_model_full.pkl" if dataset_size == "full" else "corr_model_small.pkl")
                placeholder = [1,2,3,4,5,6,7]
                save_to_pickle((features_df_corr, placeholder, all_ratings, total_movie_array), pkl_file_name)

        except Exception as e:
            logging.error("Error processing model %s: %s", model_type, e)

    logging.info("____________________________________")
    logging.info("Done! Pkl files are created")
