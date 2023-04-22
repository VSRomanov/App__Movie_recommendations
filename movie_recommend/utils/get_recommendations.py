from typing import List, Tuple

import pandas as pd

from movie_recommend.utils.recommendation_algorithms import (
    recommendation_corr,
    recommendation_knn,
    recommendation_rename_movie,
)


def get_recommendations(
        movie_features_df: pd.DataFrame,
        movie_to_compare: str,
        num_recommendations: int,
        model_type: str,
        model: object,
        total_ratings: pd.DataFrame,
        total_movie_array: List[str],
) -> Tuple[str, pd.DataFrame]:
    """
    Returns a DataFrame of movie recommendations.

    Args:
        movie_features_df (pd.DataFrame): A DataFrame with movie features as columns and movie titles as indices.
        movie_to_compare (str): The name of the movie to compare.
        num_recommendations (int): The number of recommendations to return.
        model_type (str): "knn" for k-nearest neighbors or "corr" for Pearson correlation model.
        model (object): A trained recommendation model object (k-Nearest Neighbors or Pearson correlation).
        total_ratings (pd.DataFrame): A DataFrame with the total rating count and mean rating per movie.
        total_movie_array (List[str]): A list of movie titles in the original dataset.

    Returns:
        tuple: A tuple containing the header message (str) and a pandas DataFrame with recommended movies.
    """

    # List of movies to work with (with number of ratings more than the threshold)
    if model_type == "knn":
        movie_array = movie_features_df.index
    elif model_type == "corr":
        movie_array = movie_features_df.columns

    # If the specified movie is in the original dataset
    if movie_to_compare in total_movie_array:
        # If the specified movie is in the final dataset
        if movie_to_compare in movie_array:
            # Call the appropriate recommendation function based on the model type
            if model_type == "knn":
                first_line, final_table = recommendation_knn(
                    movie_features_df,
                    model,
                    movie_to_compare,
                    num_recommendations,
                    total_ratings,
                )
            elif model_type == "corr":
                first_line, final_table = recommendation_corr(
                    movie_features_df,
                    movie_to_compare,
                    num_recommendations,
                    total_ratings,
                )

        # If the specified movie is not in the final dataset
        else:
            first_line = f'Number of ratings for "{movie_to_compare}" is not enough for the analysis. Try another movie.\n'
            final_table = recommendation_rename_movie(
                movie_to_compare, movie_array, num_recommendations, total_ratings
            )[1]
    # If the specified movie is not in the original dataset
    else:
        first_line, final_table = recommendation_rename_movie(
            movie_to_compare, total_movie_array, num_recommendations, total_ratings
        )

    # Increment the index of the table by 1
    final_table.index += 1

    # Final polishing of the table
    final_table[["mean_rating", "totalRatingCount"]] = final_table[
        ["mean_rating", "totalRatingCount"]
    ].fillna(pd.NA)
    final_table["totalRatingCount"] = final_table["totalRatingCount"].astype("Int64")
    final_table = final_table.astype(str).replace({"nan": "--", "<NA>": "--"})

    return first_line, final_table
