import time
from typing import List, Tuple

import logging
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def recommendation_rename_movie(
    movie_to_compare: str, movie_array: List[str], n_recommend: int, all_ratings: pd.DataFrame
) -> Tuple[str, pd.DataFrame]:
    """Recommends alternative movies based on a given movie title."""
    message = f'No "{movie_to_compare}" movie in the database. Try the following titles:'

    # Remove the year at the end of the movie title (to improve suggestions of alternatives)
    movie_to_compare = movie_to_compare.rstrip("(0123456789)").rstrip()

    # Get all titles with similarity score and return the top recommendations
    table_all = pd.DataFrame(data=movie_array, columns=["title"])
    table_all["similarity_score"] = table_all["title"].apply(
        lambda x: fuzz.partial_ratio(movie_to_compare, x)
    )
    table = table_all.sort_values("similarity_score", ascending=False).reset_index(drop=True).head(n_recommend)
    table = table.merge(all_ratings, on="title")
    table = table.reindex(columns=["title", "mean_rating", "totalRatingCount", "similarity_score"])

    return message, table


def knn_train(features_df: pd.DataFrame) -> NearestNeighbors:
    """Trains a k-Nearest Neighbors model on a given dataset of movie features using the cosine distance metric."""
    start_time = time.time()

    logging.info("Training a k-Nearest Neighbors model...")

    # Convert DataFrame to a sparse matrix
    movie_features_matrix = csr_matrix(features_df.values)

    # k-Nearest Neighbors model
    knn_model = NearestNeighbors(metric="cosine", algorithm="brute")
    knn_model.fit(movie_features_matrix)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Done! Time taken to train a k-Nearest Neighbors model: %.2f seconds", elapsed_time)

    return knn_model


def recommendation_knn(
    features_df: pd.DataFrame, model: NearestNeighbors, movie_to_compare: str, n_recommend: int,
    total_ratings: pd.DataFrame
) -> Tuple[str, pd.DataFrame]:
    """Recommends similar movies to a given movie using k-Nearest Neighbors algorithm."""
    # Get the index of the movie to compare
    movie_index = features_df.index.get_loc(movie_to_compare)

    # Using 'model', calculate the distances and indices of the k-Nearest Neighbors relative to 'movie_index'
    distances, indices = model.kneighbors(
        features_df.iloc[movie_index, :].values.reshape(1, -1),
        n_neighbors=n_recommend + 1
    )

    table = pd.DataFrame(columns=["title", "distance"])
    for i in range(1, len(distances.flatten())):
        # Add the recommendation to the DataFrame
        new_row = pd.DataFrame([[features_df.index[indices.flatten()[i]], distances.flatten()[i]]],
                               columns=["title", "distance"])
        table = pd.concat([table, new_row], axis=0, ignore_index=True)
    table = table.join(total_ratings.set_index('title'), on="title")
    table = table.reindex(columns=["title", "mean_rating", "totalRatingCount", "distance"])
    message = f'Recommendations for "{movie_to_compare}":'
    return message, table


def recommendation_corr(
    features_df: pd.DataFrame, movie_to_compare: str, n_recommend: int, total_ratings: pd.DataFrame
) -> Tuple[str, pd.DataFrame]:
    """Recommends top movies based on the Pearson correlation between a specified movie and other movies in the dataset."""

    # Set the minimum number of correlating ratings per movie, depending on the size of the dataset
    min_num_ratings = 20 if len(total_ratings) < 10000 else 150

    # Drop rows with NaN values in the specified movie column
    features_df_nonan = features_df.dropna(subset=[movie_to_compare])

    # Calculate Pearson correlations between 'movie_to_compare' and other movies
    correlations = [
        features_df_nonan[movie_to_compare].corr(features_df_nonan[col], min_periods=min_num_ratings)
        for col in features_df_nonan.columns
    ]

    # Combine correlations with column names and sort by correlation coefficient
    corr_to_my_movie = pd.DataFrame({"title": features_df_nonan.columns, "correlation": correlations})
    corr_to_my_movie.sort_values(by="correlation", ascending=False, inplace=True)

    # Set the index of the DataFrame to the title column
    corr_to_my_movie.set_index("title", inplace=True)

    # Drop the movie being compared from the recommendations and reset the index
    corr_to_my_movie.drop(movie_to_compare, inplace=True)
    corr_to_my_movie.reset_index(inplace=True)

    # Join the correlation DataFrame with the total_ratings DataFrame and rearrange the columns
    table = corr_to_my_movie.iloc[:n_recommend].join(total_ratings.set_index('title'), on="title")
    table = table.reindex(columns=["title", "mean_rating", "totalRatingCount", "correlation"])

    # A header message for the recommendations
    if table["correlation"].isna().all():
        message = f'Not enough ratings for "{movie_to_compare}" to conclude on correlations.\n'
    else:
        message = f'Recommendations for "{movie_to_compare}":'
        table.dropna(subset=["correlation"], inplace=True)

    return message, table
