from typing import List, Tuple

import pandas as pd

from movie_recommend.model_types import get_model_type_class_by_name
from movie_recommend.utils.recommendation_algorithms import recommendation_rename_movie


def get_recommendations(
    features_df: pd.DataFrame, movie_to_compare: str, n_recommend: int, model_type: str,
    model: object, all_ratings: pd.DataFrame, total_movie_array: List[str]
) -> Tuple[str, pd.DataFrame]:
    """Returns a message line and a final table of movie recommendations."""
    model_type_class = get_model_type_class_by_name(model_type)()

    # List of movies to work with (with number of ratings more than the threshold)
    movie_array = model_type_class.get_movie_array(features_df)

    # If the specified movie is in the original dataset
    if movie_to_compare in total_movie_array:
        # If the specified movie is in the final dataset
        if movie_to_compare in movie_array:
            # Call the appropriate recommendation function based on the model type
            first_line, final_table = model_type_class.get_recommendations(
                features_df, model, movie_to_compare, n_recommend, all_ratings
            )

        # If the specified movie is not in the final dataset
        else:
            first_line = f'Number of ratings for "{movie_to_compare}" is not enough for the analysis. Try another movie.\n'
            _, final_table = recommendation_rename_movie(movie_to_compare, movie_array, n_recommend, all_ratings)
    # If the specified movie is not in the original dataset
    else:
        first_line, final_table = recommendation_rename_movie(movie_to_compare, total_movie_array, n_recommend, all_ratings)

    # Final polishing of the table
    final_table.index += 1
    final_table[["mean_rating", "totalRatingCount"]] = final_table[["mean_rating", "totalRatingCount"]].fillna(pd.NA)
    final_table["totalRatingCount"] = final_table["totalRatingCount"].astype("Int64")
    final_table = final_table.astype(str).replace({"nan": "--", "<NA>": "--"})

    return first_line, final_table
